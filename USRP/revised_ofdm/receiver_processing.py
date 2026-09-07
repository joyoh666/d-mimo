"""Pure-DSP receive processing and virtual-pilot channel estimation.

This module has no UHD dependency. Raw samples can come directly from
``usrp_capture.transmit_and_receive_ofdm`` or from a previously saved IQ file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:  # Support script execution and package-style imports.
    from .config import DEFAULT_CONFIG, OFDMConfig
    from .synchronization import synchronize_received_frame
except ImportError:
    from USRP.test.revised_ofdm.config import DEFAULT_CONFIG, OFDMConfig
    from USRP.test.revised_ofdm.synchronization import synchronize_received_frame


ComplexArray = NDArray[np.complex64]
FloatArray = NDArray[np.float32]
IndexTuple = tuple[NDArray[np.intp], ...]


@dataclass(frozen=True)
class ReceiverProcessingResult:
    """All synchronization, channel-estimation, and equalization outputs."""

    frame_rcv: ComplexArray
    frame_rcv_timesync: ComplexArray
    frame_rcv_ffosync: ComplexArray
    frame_rcv_rfosync: ComplexArray
    resource_maps_rcv: ComplexArray
    h_fd: ComplexArray
    h_single_fd: ComplexArray
    h_fd_virtual_avg_raw: ComplexArray
    h_fd_virtual_avg_aligned: ComplexArray
    h_virtual_pilots_fd: ComplexArray
    h_virtual_pilots_fd_aligned: ComplexArray
    virtual_pilot_variance_raw: FloatArray
    virtual_pilot_variance_aligned: FloatArray
    virtual_pilot_common_phases_rad: FloatArray
    single_pilot_snr_db: float
    virtual_pilot_snr_db: float
    virtual_pilot_snr_gain_db: float
    virtual_pilot_correction_phase_rad: float
    iq_rcv: ComplexArray
    iq_rcv_single: ComplexArray
    sync_idx: int
    ffo_hz: float
    rfo_hz_per_slot: FloatArray
    timing_correlation_peak: float
    timing_sync_fallback_used: bool

    @property
    def channel_mean_fd(self) -> ComplexArray:
        """One averaged active-subcarrier CSI vector for this capture."""
        return np.mean(self.h_fd, axis=(0, 1)).astype(np.complex64)

    def to_legacy_dict(self) -> dict[str, object]:
        """Return field names compatible with the original receive function."""
        return {
            "frame_rcv": self.frame_rcv,
            "frame_rcv_timesync": self.frame_rcv_timesync,
            "frame_rcv_ffosync": self.frame_rcv_ffosync,
            "frame_rcv_rfosync": self.frame_rcv_rfosync,
            "resource_maps_rcv": self.resource_maps_rcv,
            "h_fd": self.h_fd,
            "h_single_fd": self.h_single_fd,
            "h_fd_virtual_avg_raw": self.h_fd_virtual_avg_raw,
            "h_fd_virtual_avg_aligned": self.h_fd_virtual_avg_aligned,
            "h_virtual_pilots_fd": self.h_virtual_pilots_fd,
            "h_virtual_pilots_fd_aligned": self.h_virtual_pilots_fd_aligned,
            "virtual_pilot_variance_raw": self.virtual_pilot_variance_raw,
            "virtual_pilot_variance_aligned": self.virtual_pilot_variance_aligned,
            "virtual_pilot_common_phases_rad": (
                self.virtual_pilot_common_phases_rad
            ),
            "single_pilot_snr_db": self.single_pilot_snr_db,
            "virtual_pilot_snr_db": self.virtual_pilot_snr_db,
            "virtual_pilot_snr_gain_db": self.virtual_pilot_snr_gain_db,
            "virtual_pilot_correction_phase_rad": (
                self.virtual_pilot_correction_phase_rad
            ),
            "iq_rcv": self.iq_rcv,
            "iq_rcv_single": self.iq_rcv_single,
            "sync_idx": self.sync_idx,
            "ffo_hz": self.ffo_hz,
            "rfo_hz_per_slot": self.rfo_hz_per_slot,
            "timing_correlation_peak": self.timing_correlation_peak,
            "timing_sync_fallback_used": self.timing_sync_fallback_used,
        }


def ofdm_demodulate_to_resource_grid(
    frame_rcv_rfosync: ComplexArray,
    config: OFDMConfig = DEFAULT_CONFIG,
) -> ComplexArray:
    """Remove CPs, perform the FFT, and restore active-subcarrier ordering."""
    frame = np.asarray(frame_rcv_rfosync, dtype=np.complex64).flatten()
    if frame.size != config.frame_length:
        raise ValueError(
            f"Synchronized frame must contain {config.frame_length} samples, "
            f"got {frame.size}"
        )

    num_slots = config.num_subframe_per_frame * config.num_slot_per_subframe
    fft_symbols_rcv = np.reshape(frame, (num_slots, -1))
    fft_symbols_rcv = fft_symbols_rcv[
        :,
        config.first_CP_length - config.normal_CP_length:,
    ]
    fft_symbols_rcv = np.reshape(
        fft_symbols_rcv,
        (num_slots * config.num_symbols_per_slot, -1),
    )
    fft_symbols_rcv = fft_symbols_rcv[:, config.normal_CP_length:]
    fft_symbols_rcv = np.fft.fft(fft_symbols_rcv, axis=-1, norm="ortho")
    fft_symbols_rcv = np.reshape(
        fft_symbols_rcv,
        (
            config.num_subframe_per_frame,
            config.num_slot_per_subframe,
            config.num_symbols_per_slot,
            config.FFT_SIZE,
        ),
    )

    resource_maps_rcv = np.zeros(
        (
            config.num_subframe_per_frame,
            config.num_slot_per_subframe,
            config.num_symbols_per_slot,
            config.N,
        ),
        dtype=np.complex64,
    )
    resource_maps_rcv[..., config.N // 2:] = fft_symbols_rcv[
        ...,
        1:config.N // 2 + 1,
    ]
    resource_maps_rcv[..., :config.N // 2] = fft_symbols_rcv[
        ...,
        -(config.N // 2):,
    ]
    return resource_maps_rcv


def phase_align_channel_estimates(
    h_virtual_pilots: ComplexArray,
) -> tuple[ComplexArray, FloatArray]:
    """Align each repeated-pilot CSI vector to the first pilot's common phase."""
    estimates = np.asarray(h_virtual_pilots, dtype=np.complex64)
    if estimates.ndim != 2 or estimates.shape[0] == 0:
        raise ValueError(
            "h_virtual_pilots must have shape (num_virtual_pilots, subcarriers)"
        )

    reference = estimates[0]
    aligned = [reference.astype(np.complex64)]
    common_phases = [0.0]
    for pilot_index in range(1, estimates.shape[0]):
        estimate = estimates[pilot_index]
        phase = np.angle(np.vdot(reference, estimate))
        aligned.append((estimate * np.exp(-1j * phase)).astype(np.complex64))
        common_phases.append(float(phase))
    return (
        np.stack(aligned, axis=0).astype(np.complex64),
        np.asarray(common_phases, dtype=np.float32),
    )


def pilot_snr_db_from_equalized(
    equalized_pilots_fd: ComplexArray,
    known_ref_seq: ComplexArray,
) -> float:
    """Calculate the same reference-domain SNR proxy used by the original."""
    if equalized_pilots_fd.size == 0:
        return 0.0
    equalized_unit = equalized_pilots_fd / np.asarray(
        known_ref_seq,
        dtype=np.complex64,
    ).reshape(1, -1)
    mse = np.mean(np.abs(equalized_unit - 1.0) ** 2)
    return float(10.0 * np.log10(1.0 / (mse + 1e-15)))


def estimate_channel_and_equalize(
    resource_maps_rcv: ComplexArray,
    known_ref_seq: ComplexArray,
    pdsch_idx: IndexTuple,
    config: OFDMConfig = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Estimate CSI from repeated pilots and equalize the PDSCH resource elements."""
    resource_grid = np.asarray(resource_maps_rcv, dtype=np.complex64)
    expected_shape = (
        config.num_subframe_per_frame,
        config.num_slot_per_subframe,
        config.num_symbols_per_slot,
        config.N,
    )
    if resource_grid.shape != expected_shape:
        raise ValueError(
            f"resource_maps_rcv must have shape {expected_shape}, "
            f"got {resource_grid.shape}"
        )
    known_reference = np.asarray(known_ref_seq, dtype=np.complex64)
    if known_reference.shape != (config.N,):
        raise ValueError(
            f"known_ref_seq must have shape ({config.N},), "
            f"got {known_reference.shape}"
        )
    if len(pdsch_idx) != resource_grid.ndim:
        raise ValueError(
            f"pdsch_idx must contain {resource_grid.ndim} index arrays"
        )

    h_virtual_pilots = []
    virtual_pilot_symbols = []
    positions = config.get_virtual_pilot_positions()
    if not positions:
        raise ValueError("At least one virtual-pilot position is required")
    for subframe_index, slot_index, symbol_index in positions:
        received_pilot = resource_grid[
            subframe_index,
            slot_index,
            symbol_index,
            :,
        ]
        h_virtual_pilots.append(received_pilot / known_reference)
        virtual_pilot_symbols.append(received_pilot)

    h_virtual_pilots_fd = np.stack(h_virtual_pilots, axis=0).astype(np.complex64)
    h_fd_virtual_avg_raw = np.mean(h_virtual_pilots_fd, axis=0).astype(
        np.complex64
    )
    h_virtual_pilots_fd_aligned, common_phases = phase_align_channel_estimates(
        h_virtual_pilots_fd
    )
    if config.PHASE_ALIGN_VIRTUAL_PILOTS:
        h_fd_virtual_avg_aligned = np.mean(
            h_virtual_pilots_fd_aligned,
            axis=0,
        ).astype(np.complex64)
    else:
        h_fd_virtual_avg_aligned = h_fd_virtual_avg_raw.copy()

    variance_raw = np.mean(
        np.abs(h_virtual_pilots_fd - h_fd_virtual_avg_raw[None, :]) ** 2,
        axis=0,
    ).astype(np.float32)
    variance_aligned = np.mean(
        np.abs(
            h_virtual_pilots_fd_aligned
            - h_fd_virtual_avg_aligned[None, :]
        )
        ** 2,
        axis=0,
    ).astype(np.float32)

    channel_shape = (
        config.num_subframe_per_frame,
        config.num_slot_per_subframe,
        config.N,
    )
    h_fd = np.broadcast_to(
        h_fd_virtual_avg_aligned.reshape(1, 1, -1),
        channel_shape,
    ).copy()
    h_single_fd = np.broadcast_to(
        h_virtual_pilots_fd[0].reshape(1, 1, -1),
        channel_shape,
    ).copy()

    resource_maps_rcv_zf = resource_grid / (h_fd[:, :, None, :] + 1e-12)
    iq_rcv = resource_maps_rcv_zf[pdsch_idx]
    resource_maps_rcv_zf_single = resource_grid / (
        h_single_fd[:, :, None, :] + 1e-12
    )
    iq_rcv_single = resource_maps_rcv_zf_single[pdsch_idx]

    reference_subframe, reference_slot, _ = positions[-1]
    pilot_equalized_reference = (
        resource_maps_rcv_zf[reference_subframe, reference_slot, 0, :]
        / known_reference
    )
    correction_phase = float(np.angle(np.mean(pilot_equalized_reference)))
    correction = np.complex64(np.exp(-1j * correction_phase))
    iq_rcv = (iq_rcv * correction).astype(np.complex64)
    iq_rcv_single = (iq_rcv_single * correction).astype(np.complex64)

    virtual_pilot_received = np.stack(virtual_pilot_symbols, axis=0)
    h_gain_raw = virtual_pilot_received / h_fd_virtual_avg_aligned.reshape(1, -1)
    h_gain_single = virtual_pilot_received / h_virtual_pilots_fd[0].reshape(1, -1)
    single_pilot_snr_db = pilot_snr_db_from_equalized(
        h_gain_single,
        known_reference,
    )
    virtual_pilot_snr_db = pilot_snr_db_from_equalized(
        h_gain_raw,
        known_reference,
    )

    return {
        "h_fd": h_fd.astype(np.complex64),
        "h_single_fd": h_single_fd.astype(np.complex64),
        "h_fd_virtual_avg_raw": h_fd_virtual_avg_raw,
        "h_fd_virtual_avg_aligned": h_fd_virtual_avg_aligned,
        "h_virtual_pilots_fd": h_virtual_pilots_fd,
        "h_virtual_pilots_fd_aligned": h_virtual_pilots_fd_aligned,
        "virtual_pilot_variance_raw": variance_raw,
        "virtual_pilot_variance_aligned": variance_aligned,
        "virtual_pilot_common_phases_rad": common_phases,
        "single_pilot_snr_db": single_pilot_snr_db,
        "virtual_pilot_snr_db": virtual_pilot_snr_db,
        "virtual_pilot_snr_gain_db": (
            virtual_pilot_snr_db - single_pilot_snr_db
        ),
        "virtual_pilot_correction_phase_rad": correction_phase,
        "iq_rcv": iq_rcv,
        "iq_rcv_single": iq_rcv_single,
    }


def process_received_frame(
    frame_rcv: NDArray[np.complexfloating],
    ss_td_with_cp: ComplexArray,
    known_ref_seq: ComplexArray,
    pdsch_idx: IndexTuple,
    config: OFDMConfig = DEFAULT_CONFIG,
    *,
    receive_row: int = 0,
    report_timing_fallback: bool = True,
) -> ReceiverProcessingResult:
    """Run synchronization, OFDM demodulation, CSI estimation, and equalization."""
    sync = synchronize_received_frame(
        frame_rcv,
        ss_td_with_cp,
        config,
        receive_row=receive_row,
        report_timing_fallback=report_timing_fallback,
    )
    resource_maps_rcv = ofdm_demodulate_to_resource_grid(
        sync.frame_rcv_rfosync,
        config,
    )
    channel = estimate_channel_and_equalize(
        resource_maps_rcv,
        known_ref_seq,
        pdsch_idx,
        config,
    )

    return ReceiverProcessingResult(
        frame_rcv=sync.frame_rcv,
        frame_rcv_timesync=sync.frame_rcv_timesync,
        frame_rcv_ffosync=sync.frame_rcv_ffosync,
        frame_rcv_rfosync=sync.frame_rcv_rfosync,
        resource_maps_rcv=resource_maps_rcv,
        sync_idx=sync.sync_idx,
        ffo_hz=sync.ffo_hz,
        rfo_hz_per_slot=sync.rfo_hz_per_slot,
        timing_correlation_peak=sync.timing_correlation_peak,
        timing_sync_fallback_used=sync.timing_sync_fallback_used,
        **channel
    )


def active_to_full_spectrum(
    channel_active: ComplexArray,
    config: OFDMConfig = DEFAULT_CONFIG,
) -> ComplexArray:
    """Map active-subcarrier CSI into the complete FFT-bin layout."""
    channel = np.asarray(channel_active, dtype=np.complex64).flatten()
    if channel.shape != (config.N,):
        raise ValueError(
            f"channel_active must have shape ({config.N},), got {channel.shape}"
        )
    full_spectrum = np.zeros(config.FFT_SIZE, dtype=np.complex64)
    full_spectrum[1:config.N // 2 + 1] = channel[config.N // 2:]
    full_spectrum[-(config.N // 2):] = channel[:config.N // 2]
    return full_spectrum


def channel_to_delay_response(
    channel_active: ComplexArray,
    config: OFDMConfig = DEFAULT_CONFIG,
) -> tuple[ComplexArray, ComplexArray]:
    """Return the channel impulse response and its full FFT-bin spectrum."""
    full_spectrum = active_to_full_spectrum(channel_active, config)
    impulse_response = np.fft.ifft(
        full_spectrum,
        n=config.FFT_SIZE,
        norm="ortho",
    )
    return (
        impulse_response.astype(np.complex64),
        full_spectrum.astype(np.complex64),
    )


def delay_residual_power(
    impulse_response: ComplexArray,
    *,
    guard_taps: int = 2,
) -> tuple[float, float, float]:
    """Return main power, residual power, and main/residual ratio in dB."""
    if guard_taps < 0:
        raise ValueError("guard_taps must be non-negative")
    power = np.abs(np.asarray(impulse_response).flatten()) ** 2
    if power.size == 0:
        return 0.0, 0.0, -np.inf
    peak_index = int(np.argmax(power))
    residual_mask = np.ones(power.shape[0], dtype=np.bool_)
    lower = max(0, peak_index - guard_taps)
    upper = min(power.shape[0], peak_index + guard_taps + 1)
    residual_mask[lower:upper] = False

    main_power = float(np.mean(power[~residual_mask]) + 1e-15)
    residual_power = float(np.mean(power[residual_mask]) + 1e-15)
    residual_reduction_db = float(10.0 * np.log10(main_power / residual_power))
    return main_power, residual_power, residual_reduction_db
