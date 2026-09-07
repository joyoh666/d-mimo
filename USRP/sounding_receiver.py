"""Receive, synchronize, and estimate six-branch sounding CSI.

Processing follows the waveform in ``sounding_transmitter.py``:

1. capture one complex stream from the TurtleBot-mounted USRP,
2. locate every 5 ms period by normalized Zadoff-Chu correlation,
3. optionally estimate one capture-wide CFO from the pilot cyclic prefixes,
4. remove each pilot CP and apply a 128-point FFT,
5. estimate one complex CSI value per TX branch with ``H = Y / X``,
6. complex-average the three repeated estimates, and
7. return timestamped CSI and synchronization quality to the caller.

The current pilot allocation measures each TX branch on one distinct
subcarrier.  The saved CSI is therefore ``(snapshot, tx_branch)``, not a
full-band ``(snapshot, tx_branch, subcarrier)`` tensor.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import correlate, find_peaks

if __package__:  # Package imports from the repository root.
    from .helper_function import (
        compensate_cfo,
        estimate_cfo_from_cp,
        get_zc_sequence,
        least_squares_channel_estimate,
        phase_align_channel_estimates,
        sample_timestamps,
    )
    from .sounding_config import DEFAULT_SOUNDING_CONFIG, SoundingConfig
    from .sounding_transmitter import build_transmitted_pilots
else:  # Direct execution from inside USRP/.
    from helper_function import (
        compensate_cfo,
        estimate_cfo_from_cp,
        get_zc_sequence,
        least_squares_channel_estimate,
        phase_align_channel_estimates,
        sample_timestamps,
    )
    from sounding_config import DEFAULT_SOUNDING_CONFIG, SoundingConfig
    from sounding_transmitter import build_transmitted_pilots


ComplexArray = NDArray[np.complex64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReceiverHardwareConfig:
    """Deployment-specific settings for the TurtleBot receiver USRP."""

    address: str
    rx_subdev_spec: str = "A:0"
    rx_antenna: str = "TX/RX"
    rx_channel: int = 0
    clock_source: str = "external"
    time_source: str = "external"
    reference_lock_timeout_s: float = 10.0

    def validate(self) -> None:
        if not self.address.strip():
            raise ValueError("receiver USRP address must not be empty")
        if self.rx_channel < 0:
            raise ValueError("rx_channel must be non-negative")
        if self.reference_lock_timeout_s <= 0:
            raise ValueError("reference_lock_timeout_s must be positive")

    def device_args(self, config: SoundingConfig) -> str:
        self.validate()
        return (
            f"addr={self.address},"
            f"master_clock_rate={config.master_clock_rate_hz:.0f}"
        )


@dataclass(frozen=True, slots=True)
class ReceiverProcessingConfig:
    """Synchronization and capture choices not fixed by the paper."""

    correlation_threshold: float = 0.35
    minimum_peak_distance_fraction: float = 0.8
    compensate_frequency_offset: bool = True

    def validate(self) -> None:
        if not 0.0 < self.correlation_threshold <= 1.0:
            raise ValueError("correlation_threshold must be in (0, 1]")
        if not 0.0 < self.minimum_peak_distance_fraction <= 1.0:
            raise ValueError(
                "minimum_peak_distance_fraction must be in (0, 1]"
            )


@dataclass(frozen=True, slots=True)
class RawCapture:
    """One contiguous hardware-timestamped RX sample buffer."""

    samples: ComplexArray
    first_sample_time_s: float
    requested_start_time_s: float


@dataclass(frozen=True, slots=True)
class SoundingCSIResult:
    """Processed CSI and quality information for all accepted snapshots."""

    csi: ComplexArray
    csi_repetitions: ComplexArray
    csi_repetitions_aligned: ComplexArray
    pilot_received: ComplexArray
    csi_timestamps_usrp_s: FloatArray
    pilot_timestamps_usrp_s: FloatArray
    frame_start_sample_indices: NDArray[np.int64]
    frame_sequence_numbers: NDArray[np.int64]
    timing_error_samples: NDArray[np.int64]
    zc_correlation_scores: NDArray[np.float32]
    pilot_noise_power: NDArray[np.float32]
    pilot_snr_db: NDArray[np.float32]
    repetition_phase_corrections_rad: NDArray[np.float32]
    estimated_cfo_hz: float

    @property
    def num_snapshots(self) -> int:
        return int(self.csi.shape[0])


def _import_uhd() -> Any:
    try:
        import uhd  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "UHD Python bindings are not installed. Install the UHD package "
            "matching the receiver host driver."
        ) from error
    return uhd


def _make_tune_request(uhd: Any, frequency_hz: float) -> Any:
    tune_request = getattr(uhd.types, "TuneRequest", None)
    if tune_request is not None:
        return tune_request(frequency_hz)
    return uhd.libpyuhd.types.tune_request(frequency_hz)


def _wait_for_sensor(
    read_value: Any,
    *,
    description: str,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if bool(read_value()):
            return
        time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for {description}")


def initialize_receiver(
    hardware: ReceiverHardwareConfig,
    config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
    *,
    reset_hardware_time: bool = True,
) -> Any:
    """Open and configure the TurtleBot receiver USRP."""
    hardware.validate()
    uhd = _import_uhd()
    device_args = hardware.device_args(config)
    print(f"Opening receiver USRP: {device_args}")
    usrp = uhd.usrp.MultiUSRP(device_args)
    if usrp.get_num_mboards() != 1:
        raise RuntimeError("the TurtleBot receiver must resolve to one motherboard")

    usrp.set_clock_source(hardware.clock_source, 0)
    usrp.set_time_source(hardware.time_source, 0)
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(hardware.rx_subdev_spec), 0)

    sensor_names = usrp.get_mboard_sensor_names(0)
    if hardware.clock_source == "external" and "ref_locked" in sensor_names:
        _wait_for_sensor(
            lambda: usrp.get_mboard_sensor("ref_locked", 0).to_bool(),
            description="the OctoClock 10 MHz reference lock",
            timeout_s=hardware.reference_lock_timeout_s,
        )

    actual_master_clock = float(usrp.get_master_clock_rate(0))
    if not math.isclose(
        actual_master_clock,
        config.master_clock_rate_hz,
        rel_tol=1e-9,
        abs_tol=1.0,
    ):
        raise RuntimeError(
            f"receiver master clock is {actual_master_clock} Hz, not "
            f"{config.master_clock_rate_hz} Hz"
        )

    if reset_hardware_time:
        if hardware.time_source == "external":
            usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
        else:
            usrp.set_time_now(uhd.types.TimeSpec(0.0), 0)

    if hardware.rx_channel >= usrp.get_rx_num_channels():
        raise RuntimeError(
            f"RX channel {hardware.rx_channel} requested, but UHD exposes "
            f"{usrp.get_rx_num_channels()} channel(s)"
        )

    channel = hardware.rx_channel
    usrp.set_rx_rate(config.sampling_rate_hz, channel)
    usrp.set_rx_freq(_make_tune_request(uhd, config.center_frequency_hz), channel)
    usrp.set_rx_gain(config.rx_gain_db, channel)
    usrp.set_rx_bandwidth(config.nominal_bandwidth_hz, channel)
    usrp.set_rx_antenna(hardware.rx_antenna, channel)

    actual_rate = float(usrp.get_rx_rate(channel))
    if not math.isclose(
        actual_rate,
        config.sampling_rate_hz,
        rel_tol=1e-6,
        abs_tol=1.0,
    ):
        raise RuntimeError(
            f"RX rate is {actual_rate} S/s, not {config.sampling_rate_hz} S/s"
        )

    rx_sensor_names = usrp.get_rx_sensor_names(channel)
    if "lo_locked" in rx_sensor_names:
        _wait_for_sensor(
            lambda: usrp.get_rx_sensor("lo_locked", channel).to_bool(),
            description="the receiver LO lock",
            timeout_s=hardware.reference_lock_timeout_s,
        )

    print(
        f"RX {channel}: rate={actual_rate:.0f} S/s, "
        f"frequency={usrp.get_rx_freq(channel):.0f} Hz, "
        f"gain={usrp.get_rx_gain(channel):.1f} dB, "
        f"clock={hardware.clock_source}, time={hardware.time_source}"
    )
    return usrp


def choose_capture_start_time(usrp: Any, minimum_lead_time_s: float) -> float:
    """Return a future integer UHD second for a timed RX command."""
    now = float(usrp.get_time_now().get_real_secs())
    return float(math.ceil(now + minimum_lead_time_s) + 1)


def capture_samples(
    usrp: Any,
    num_samples: int,
    start_time_s: float,
    hardware: ReceiverHardwareConfig,
    config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
) -> RawCapture:
    """Capture an exact, contiguous, hardware-timestamped RX buffer."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    uhd = _import_uhd()
    now = float(usrp.get_time_now().get_real_secs())
    if start_time_s - now < config.startup_lead_time_s:
        raise ValueError(
            f"RX start must be at least {config.startup_lead_time_s} s in "
            f"the future (now={now:.6f}, requested={start_time_s:.6f})"
        )

    stream_args = uhd.usrp.StreamArgs(
        config.cpu_sample_format,
        config.wire_sample_format,
    )
    stream_args.channels = [hardware.rx_channel]
    streamer = usrp.get_rx_stream(stream_args)

    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.stream_now = False
    stream_cmd.time_spec = uhd.types.TimeSpec(start_time_s)
    stream_cmd.num_samps = num_samples
    streamer.issue_stream_cmd(stream_cmd)

    samples = np.empty(num_samples, dtype=np.complex64)
    receive_buffer = np.empty(
        (1, streamer.get_max_num_samps()),
        dtype=np.complex64,
    )
    metadata = uhd.types.RXMetadata()
    first_sample_time_s: float | None = None
    received_total = 0
    wall_deadline = (
        time.monotonic()
        + max(0.0, start_time_s - now)
        + num_samples / config.sampling_rate_hz
        + 5.0
    )

    while received_total < num_samples:
        received = int(streamer.recv(receive_buffer, metadata))
        error_code = metadata.error_code
        if error_code == uhd.types.RXMetadataErrorCode.timeout:
            if time.monotonic() >= wall_deadline:
                raise TimeoutError("timed out while waiting for RX samples")
            continue
        if error_code != uhd.types.RXMetadataErrorCode.none:
            detail = metadata.strerror() if hasattr(metadata, "strerror") else str(error_code)
            raise RuntimeError(f"UHD RX metadata error: {detail}")
        if received <= 0:
            if time.monotonic() >= wall_deadline:
                raise TimeoutError("UHD returned no RX samples before deadline")
            continue

        if first_sample_time_s is None:
            if not metadata.has_time_spec:
                raise RuntimeError("first RX packet has no UHD timestamp")
            first_sample_time_s = float(metadata.time_spec.get_real_secs())
        elif metadata.has_time_spec:
            packet_time_s = float(metadata.time_spec.get_real_secs())
            expected_time_s = (
                first_sample_time_s
                + received_total / config.sampling_rate_hz
            )
            if not math.isclose(
                packet_time_s,
                expected_time_s,
                rel_tol=0.0,
                abs_tol=0.5 / config.sampling_rate_hz,
            ):
                raise RuntimeError(
                    "non-contiguous RX timestamps detected; refusing to save "
                    "a corrupted CSI time series"
                )

        copied = min(received, num_samples - received_total)
        samples[received_total : received_total + copied] = receive_buffer[
            0,
            :copied,
        ]
        received_total += copied

    if first_sample_time_s is None:
        raise RuntimeError("capture completed without a hardware timestamp")
    return RawCapture(
        samples=samples,
        first_sample_time_s=first_sample_time_s,
        requested_start_time_s=start_time_s,
    )


class ContinuousRXCapture:
    """Continuously read fixed-size chunks on a background UHD RX thread.

    The background reader prevents host-side CSI processing and file writes
    from pausing the hardware stream.  Every returned :class:`RawCapture` is
    contiguous and carries the UHD time of its first sample.
    """

    def __init__(
        self,
        usrp: Any,
        start_time_s: float,
        chunk_samples: int,
        hardware: ReceiverHardwareConfig,
        config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
        *,
        queue_depth: int = 4,
    ) -> None:
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        if queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        self.usrp = usrp
        self.start_time_s = start_time_s
        self.chunk_samples = chunk_samples
        self.hardware = hardware
        self.config = config
        self._queue: queue.Queue[RawCapture | BaseException] = queue.Queue(
            maxsize=queue_depth
        )
        self._stop_event = threading.Event()
        self._armed_event = threading.Event()
        self._thread = threading.Thread(
            target=self._receive_loop,
            name="continuous-usrp-rx",
            daemon=True,
        )

    def start(self) -> None:
        if self._thread.is_alive():
            raise RuntimeError("continuous receiver is already running")
        now = float(self.usrp.get_time_now().get_real_secs())
        if self.start_time_s - now < self.config.startup_lead_time_s:
            raise ValueError(
                f"RX start must be at least {self.config.startup_lead_time_s} "
                f"s in the future"
            )
        self._thread.start()
        if not self._armed_event.wait(timeout=5.0):
            self._raise_queued_error_if_present()
            raise TimeoutError("continuous RX worker did not arm in time")

    def get_chunk(self, timeout_s: float = 10.0) -> RawCapture:
        try:
            item = self._queue.get(timeout=timeout_s)
        except queue.Empty as error:
            raise TimeoutError("timed out waiting for a continuous RX chunk") from error
        if isinstance(item, BaseException):
            raise RuntimeError("continuous UHD RX worker failed") from item
        return item

    def close(self) -> None:
        self._stop_event.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("continuous RX worker did not stop")

    def _raise_queued_error_if_present(self) -> None:
        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            return
        if isinstance(item, BaseException):
            raise RuntimeError("continuous UHD RX worker failed") from item
        self._queue.put_nowait(item)

    def _put(self, item: RawCapture | BaseException) -> None:
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _receive_loop(self) -> None:
        uhd = _import_uhd()
        stream_args = uhd.usrp.StreamArgs(
            self.config.cpu_sample_format,
            self.config.wire_sample_format,
        )
        stream_args.channels = [self.hardware.rx_channel]
        streamer = self.usrp.get_rx_stream(stream_args)
        receive_buffer = np.empty(
            (1, streamer.get_max_num_samps()),
            dtype=np.complex64,
        )
        metadata = uhd.types.RXMetadata()
        expected_next_time_s: float | None = None
        chunk_index = 0

        try:
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
            stream_cmd.stream_now = False
            stream_cmd.time_spec = uhd.types.TimeSpec(self.start_time_s)
            streamer.issue_stream_cmd(stream_cmd)
            self._armed_event.set()

            while not self._stop_event.is_set():
                chunk = np.empty(self.chunk_samples, dtype=np.complex64)
                chunk_first_time_s: float | None = None
                received_total = 0

                while (
                    received_total < self.chunk_samples
                    and not self._stop_event.is_set()
                ):
                    requested = min(
                        receive_buffer.shape[1],
                        self.chunk_samples - received_total,
                    )
                    received = int(
                        streamer.recv(receive_buffer[:, :requested], metadata)
                    )
                    error_code = metadata.error_code
                    if error_code == uhd.types.RXMetadataErrorCode.timeout:
                        continue
                    if error_code != uhd.types.RXMetadataErrorCode.none:
                        detail = (
                            metadata.strerror()
                            if hasattr(metadata, "strerror")
                            else str(error_code)
                        )
                        raise RuntimeError(f"UHD RX metadata error: {detail}")
                    if received <= 0:
                        continue
                    if not metadata.has_time_spec:
                        raise RuntimeError("RX packet has no UHD timestamp")

                    packet_time_s = float(metadata.time_spec.get_real_secs())
                    if chunk_first_time_s is None:
                        chunk_first_time_s = packet_time_s
                    if expected_next_time_s is not None and not math.isclose(
                        packet_time_s,
                        expected_next_time_s,
                        rel_tol=0.0,
                        abs_tol=0.5 / self.config.sampling_rate_hz,
                    ):
                        raise RuntimeError(
                            "non-contiguous RX timestamps detected; the stream "
                            "may have overflowed"
                        )

                    chunk[received_total : received_total + received] = (
                        receive_buffer[0, :received]
                    )
                    received_total += received
                    expected_next_time_s = (
                        packet_time_s
                        + received / self.config.sampling_rate_hz
                    )

                if self._stop_event.is_set():
                    break
                if chunk_first_time_s is None:
                    raise RuntimeError("RX chunk completed without a timestamp")
                self._put(
                    RawCapture(
                        samples=chunk,
                        first_sample_time_s=chunk_first_time_s,
                        requested_start_time_s=(
                            self.start_time_s
                            + chunk_index
                            * self.chunk_samples
                            / self.config.sampling_rate_hz
                        ),
                    )
                )
                chunk_index += 1
        except BaseException as error:
            self._armed_event.set()
            self._put(error)
        finally:
            stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            stop_cmd.stream_now = True
            streamer.issue_stream_cmd(stop_cmd)


def normalized_zc_correlation(
    received_samples: ArrayLike,
    config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
) -> NDArray[np.float32]:
    """Return normalized sliding ZC correlation scores in the range [0, 1]."""
    received = np.asarray(received_samples, dtype=np.complex64).reshape(-1)
    reference = (
        get_zc_sequence(config.zc_length_samples, config.zc_root)
        * config.digital_power_scale
    )
    if received.size < reference.size:
        raise ValueError("received_samples is shorter than the ZC sequence")

    raw = np.abs(correlate(received, reference, mode="valid", method="fft"))
    sample_power = np.abs(received).astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(sample_power)))
    window_energy = cumulative[reference.size :] - cumulative[: -reference.size]
    reference_energy = float(np.vdot(reference, reference).real)
    denominator = np.sqrt(np.maximum(window_energy * reference_energy, 1e-30))
    return np.clip(raw / denominator, 0.0, 1.0).astype(np.float32)


def detect_sounding_frames(
    received_samples: ArrayLike,
    config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
    processing: ReceiverProcessingConfig = ReceiverProcessingConfig(),
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Find complete sounding bursts and return starts and ZC scores."""
    processing.validate()
    received = np.asarray(received_samples, dtype=np.complex64).reshape(-1)
    score = normalized_zc_correlation(received, config)
    minimum_distance = max(
        1,
        round(
            processing.minimum_peak_distance_fraction
            * config.csi_interval_samples
        ),
    )
    peaks, properties = find_peaks(
        score,
        height=processing.correlation_threshold,
        distance=minimum_distance,
    )
    peaks = peaks.astype(np.int64)
    heights = np.asarray(properties["peak_heights"], dtype=np.float32)

    complete = peaks + config.sounding_burst_samples <= received.size
    return peaks[complete], heights[complete]


def estimate_capture_cfo(
    received_samples: ArrayLike,
    frame_starts: ArrayLike,
    config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
) -> float:
    """Estimate one common CFO from every complete received pilot CP."""
    received = np.asarray(received_samples, dtype=np.complex64).reshape(-1)
    starts = np.asarray(frame_starts, dtype=np.int64).reshape(-1)
    estimates: list[float] = []
    for frame_start in starts:
        for repetition in range(config.pilot_repetitions):
            symbol_start = int(frame_start) + config.pilot_symbol_cp_start(
                repetition
            )
            symbol_stop = symbol_start + config.pilot_symbol_samples
            if symbol_stop > received.size:
                continue
            estimates.append(
                estimate_cfo_from_cp(
                    received[symbol_start:symbol_stop],
                    config.fft_size,
                    config.cp_length_samples,
                    config.sampling_rate_hz,
                )
            )
    if not estimates:
        raise ValueError("no complete pilot symbols are available for CFO estimation")
    return float(np.median(np.asarray(estimates, dtype=np.float64)))


def _active_fft_indices(config: SoundingConfig) -> NDArray[np.int64]:
    half = config.num_active_subcarriers // 2
    centered = np.concatenate(
        (
            np.arange(-half, 0, dtype=np.int64),
            np.arange(1, half + 1, dtype=np.int64),
        )
    )
    return centered % config.fft_size


def process_received_samples(
    received_samples: ArrayLike,
    first_sample_time_s: float,
    config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
    processing: ReceiverProcessingConfig = ReceiverProcessingConfig(),
    *,
    max_snapshots: int | None = None,
    cfo_override_hz: float | None = None,
    cfo_phase_reference_time_s: float | None = None,
) -> SoundingCSIResult:
    """Synchronize a contiguous capture and estimate timestamped branch CSI.

    By default, CFO is estimated independently for this capture and its phase
    correction starts at the first supplied sample.  A continuous experiment
    can instead reuse one ``cfo_override_hz`` for every chunk and provide one
    common ``cfo_phase_reference_time_s``.  That keeps the correction phase
    continuous across overlapping processing buffers.
    """
    processing.validate()
    if max_snapshots is not None and max_snapshots <= 0:
        raise ValueError("max_snapshots must be positive or None")

    received = np.asarray(received_samples, dtype=np.complex64).reshape(-1)
    received = (received - np.mean(received)).astype(np.complex64)
    initial_starts, initial_scores = detect_sounding_frames(
        received,
        config,
        processing,
    )
    if initial_starts.size == 0:
        raise RuntimeError(
            "no ZC sequence exceeded the correlation threshold; check TX, "
            "frequency, gain, antenna port, and threshold"
        )

    estimated_cfo_hz = (
        estimate_capture_cfo(received, initial_starts, config)
        if cfo_override_hz is None
        else float(cfo_override_hz)
    )
    if not math.isfinite(estimated_cfo_hz):
        raise ValueError("cfo_override_hz must be finite")
    if processing.compensate_frequency_offset:
        first_sample_index = 0
        if cfo_phase_reference_time_s is not None:
            first_sample_index = round(
                (first_sample_time_s - cfo_phase_reference_time_s)
                * config.sampling_rate_hz
            )
        corrected = compensate_cfo(
            received,
            estimated_cfo_hz,
            config.sampling_rate_hz,
            first_sample_index=first_sample_index,
        )
        frame_starts, correlation_scores = detect_sounding_frames(
            corrected,
            config,
            processing,
        )
    else:
        corrected = received
        frame_starts = initial_starts
        correlation_scores = initial_scores
    if frame_starts.size == 0:
        raise RuntimeError("ZC detection failed after CFO compensation")

    if max_snapshots is not None:
        frame_starts = frame_starts[:max_snapshots]
        correlation_scores = correlation_scores[:max_snapshots]

    transmitted_pilots = build_transmitted_pilots(config)
    num_frames = frame_starts.size
    csi_repetitions = np.empty(
        (num_frames, config.pilot_repetitions, config.num_tx_branches),
        dtype=np.complex64,
    )
    pilot_received = np.empty_like(csi_repetitions)
    noise_power = np.empty(
        (num_frames, config.pilot_repetitions),
        dtype=np.float32,
    )
    pilot_snr_db = np.empty_like(csi_repetitions, dtype=np.float32)

    assigned_fft_indices = np.asarray(
        [index % config.fft_size for index in config.pilot_centered_bins],
        dtype=np.int64,
    )
    noise_fft_indices = np.setdiff1d(
        _active_fft_indices(config),
        assigned_fft_indices,
        assume_unique=True,
    )

    for frame_index, frame_start in enumerate(frame_starts):
        for repetition in range(config.pilot_repetitions):
            useful_start = (
                int(frame_start)
                + config.pilot_symbol_data_start(repetition)
            )
            useful_stop = useful_start + config.fft_size
            useful = corrected[useful_start:useful_stop]
            if useful.size != config.fft_size:
                raise RuntimeError("a detected frame contains an incomplete pilot")

            fft_bins = np.fft.fft(useful, norm="ortho").astype(np.complex64)
            received_pilots = fft_bins[assigned_fft_indices]
            pilot_received[frame_index, repetition] = received_pilots
            csi_repetitions[frame_index, repetition] = (
                least_squares_channel_estimate(
                    received_pilots,
                    transmitted_pilots[repetition],
                )
            )

            estimated_noise = float(
                np.mean(np.abs(fft_bins[noise_fft_indices]) ** 2)
            )
            noise_power[frame_index, repetition] = estimated_noise
            pilot_snr_db[frame_index, repetition] = (
                10.0
                * np.log10(
                    (np.abs(received_pilots) ** 2 + 1e-20)
                    / (estimated_noise + 1e-20)
                )
            ).astype(np.float32)

    csi_repetitions_aligned = np.empty_like(csi_repetitions)
    phase_corrections = np.zeros(
        (num_frames, config.pilot_repetitions),
        dtype=np.float32,
    )
    for frame_index in range(num_frames):
        if config.phase_align_pilot_repetitions:
            aligned, phases = phase_align_channel_estimates(
                csi_repetitions[frame_index]
            )
            csi_repetitions_aligned[frame_index] = aligned
            phase_corrections[frame_index] = phases
        else:
            csi_repetitions_aligned[frame_index] = csi_repetitions[frame_index]

    csi = np.mean(csi_repetitions_aligned, axis=1).astype(np.complex64)

    sequence_numbers = np.rint(
        (frame_starts - frame_starts[0]) / config.csi_interval_samples
    ).astype(np.int64)
    nominal_starts = (
        frame_starts[0] + sequence_numbers * config.csi_interval_samples
    )
    timing_error_samples = (frame_starts - nominal_starts).astype(np.int64)

    csi_offsets = frame_starts + config.csi_timestamp_offset_samples
    csi_timestamps = sample_timestamps(
        first_sample_time_s,
        csi_offsets,
        config.sampling_rate_hz,
    )
    pilot_center_offsets = np.asarray(
        [
            config.pilot_symbol_data_start(repetition)
            + config.fft_size // 2
            for repetition in range(config.pilot_repetitions)
        ],
        dtype=np.int64,
    )
    pilot_offsets = frame_starts[:, None] + pilot_center_offsets[None, :]
    pilot_timestamps = sample_timestamps(
        first_sample_time_s,
        pilot_offsets,
        config.sampling_rate_hz,
    )

    return SoundingCSIResult(
        csi=csi,
        csi_repetitions=csi_repetitions,
        csi_repetitions_aligned=csi_repetitions_aligned,
        pilot_received=pilot_received,
        csi_timestamps_usrp_s=np.asarray(csi_timestamps, dtype=np.float64),
        pilot_timestamps_usrp_s=np.asarray(
            pilot_timestamps,
            dtype=np.float64,
        ),
        frame_start_sample_indices=frame_starts,
        frame_sequence_numbers=sequence_numbers,
        timing_error_samples=timing_error_samples,
        zc_correlation_scores=correlation_scores,
        pilot_noise_power=noise_power,
        pilot_snr_db=pilot_snr_db,
        repetition_phase_corrections_rad=phase_corrections,
        estimated_cfo_hz=estimated_cfo_hz,
    )


__all__ = [
    "ContinuousRXCapture",
    "RawCapture",
    "ReceiverHardwareConfig",
    "ReceiverProcessingConfig",
    "SoundingCSIResult",
    "capture_samples",
    "choose_capture_start_time",
    "detect_sounding_frames",
    "estimate_capture_cfo",
    "initialize_receiver",
    "normalized_zc_correlation",
    "process_received_samples",
]
