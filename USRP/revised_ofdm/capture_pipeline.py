"""Repeated channel-capture orchestration and result aggregation.

This module joins the hardware capture and pure-DSP receiver stages without
owning either responsibility.  It also performs the per-capture delay-domain
post-processing that used to live in the original script's ``main`` function.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

import numpy as np
from numpy.typing import NDArray

if __package__:  # Package-style import (for example, ``python -m USRP...``).
    from .config import DEFAULT_CONFIG, OFDMConfig
    from .ofdm_frame import FrameArtifacts
    from .receiver_processing import (
        ReceiverProcessingResult,
        channel_to_delay_response,
        delay_residual_power,
        process_received_frame,
    )
else:  # Direct execution from the USRP directory.
    from USRP.test.revised_ofdm.config import DEFAULT_CONFIG, OFDMConfig
    from USRP.test.revised_ofdm.ofdm_frame import FrameArtifacts
    from USRP.test.revised_ofdm.receiver_processing import (
        ReceiverProcessingResult,
        channel_to_delay_response,
        delay_residual_power,
        process_received_frame,
    )


ComplexArray = NDArray[np.complex64]
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]
CaptureFunction = Callable[[Any, ComplexArray, OFDMConfig], ComplexArray]
ProcessingFunction = Callable[
    [ComplexArray, ComplexArray, ComplexArray, tuple[NDArray[np.intp], ...], OFDMConfig],
    ReceiverProcessingResult,
]


class CaptureResults(TypedDict):
    """Stacked outputs saved and plotted by the original capture program."""

    channel_estimates_fd: ComplexArray
    channel_estimates_fd_single_pilot: ComplexArray
    channel_estimates_fd_raw_virtual: ComplexArray
    channel_estimates_fd_aligned_virtual: ComplexArray
    channel_mean_fd: ComplexArray
    channel_impulse_response_td: ComplexArray
    channel_full_spectrum_fd: ComplexArray
    channel_impulse_response_td_single_pilot: ComplexArray
    channel_full_spectrum_fd_single_pilot: ComplexArray
    virtual_pilot_variance_raw: FloatArray
    virtual_pilot_variance_aligned: FloatArray
    virtual_pilot_common_phases_rad: FloatArray
    single_pilot_snr_db: FloatArray
    virtual_pilot_snr_db: FloatArray
    virtual_pilot_snr_gain_db: FloatArray
    virtual_pilot_correction_phase_rad: FloatArray
    delay_profile_residual_gain_db: FloatArray
    sync_indices: IntArray
    ffo_estimates_hz: FloatArray
    rfo_estimates_hz: FloatArray
    timing_correlation_peaks: FloatArray
    example_iq_rcv: ComplexArray
    example_iq_rcv_single: ComplexArray
    example_frame_rcv: ComplexArray


def _default_capture_function(
    usrp: Any,
    waveform: ComplexArray,
    config: OFDMConfig,
) -> ComplexArray:
    """Load the UHD boundary only when a real capture is requested."""
    if __package__:
        from .usrp_capture import transmit_and_receive_ofdm
    else:
        from USRP.test.revised_ofdm.usrp_capture import transmit_and_receive_ofdm

    return transmit_and_receive_ofdm(usrp, waveform, config)


def collect_channel_results(
    usrp: Any,
    frame_artifacts: FrameArtifacts,
    known_ref_seq: ComplexArray,
    config: OFDMConfig = DEFAULT_CONFIG,
    *,
    capture_function: CaptureFunction | None = None,
    processing_function: ProcessingFunction = process_received_frame,
) -> CaptureResults:
    """Capture, process, and stack all observations for one bandwidth.

    ``capture_function`` is injectable so the complete aggregation path can be
    verified with loopback or saved IQ data without importing UHD.
    """
    capture = capture_function or _default_capture_function
    waveform = frame_artifacts["waveform"]
    ss_td_with_cp = frame_artifacts["ss_td_with_cp"]
    pdsch_idx = frame_artifacts["pdsch_idx"]

    channel_estimates_fd: list[ComplexArray] = []
    channel_estimates_fd_single_pilot: list[ComplexArray] = []
    channel_estimates_fd_raw_virtual: list[ComplexArray] = []
    channel_estimates_fd_aligned_virtual: list[ComplexArray] = []
    channel_mean_fd: list[ComplexArray] = []
    channel_impulse_response_td: list[ComplexArray] = []
    channel_full_spectrum_fd: list[ComplexArray] = []
    channel_impulse_response_td_single_pilot: list[ComplexArray] = []
    channel_full_spectrum_fd_single_pilot: list[ComplexArray] = []
    virtual_pilot_variance_raw: list[FloatArray] = []
    virtual_pilot_variance_aligned: list[FloatArray] = []
    virtual_pilot_common_phases: list[FloatArray] = []
    single_pilot_snr_db: list[float] = []
    virtual_pilot_snr_db: list[float] = []
    virtual_pilot_snr_gain_db: list[float] = []
    virtual_pilot_correction_phase_rad: list[float] = []
    delay_profile_residual_gain_db: list[float] = []
    sync_indices: list[int] = []
    ffo_estimates_hz: list[float] = []
    rfo_estimates_hz: list[FloatArray] = []
    timing_correlation_peaks: list[float] = []

    example_iq_rcv: ComplexArray | None = None
    example_iq_rcv_single: ComplexArray | None = None
    example_frame_rcv: ComplexArray | None = None

    for capture_index in range(config.NUM_CHANNEL_CAPTURES):
        print(
            f"[BW {config.BANDWIDTH:>5} MHz] Capture "
            f"{capture_index + 1}/{config.NUM_CHANNEL_CAPTURES}"
        )
        raw_samples = capture(usrp, waveform, config)
        received = processing_function(
            raw_samples,
            ss_td_with_cp,
            known_ref_seq,
            pdsch_idx,
            config,
        )

        mean_channel = received.channel_mean_fd
        single_pilot_channel = np.mean(
            received.h_single_fd,
            axis=(0, 1),
        ).astype(np.complex64)
        impulse_response, full_spectrum = channel_to_delay_response(
            mean_channel,
            config,
        )
        single_impulse_response, single_full_spectrum = (
            channel_to_delay_response(single_pilot_channel, config)
        )

        delay_guard_taps = max(1, int(round(config.FFT_SIZE / 200)))
        _, single_residual_power, _ = delay_residual_power(
            single_impulse_response,
            guard_taps=delay_guard_taps,
        )
        _, averaged_residual_power, _ = delay_residual_power(
            impulse_response,
            guard_taps=delay_guard_taps,
        )
        residual_gain_db = 10.0 * np.log10(
            (single_residual_power + 1e-15)
            / (averaged_residual_power + 1e-15)
        )

        channel_estimates_fd.append(received.h_fd.astype(np.complex64))
        channel_estimates_fd_single_pilot.append(
            received.h_single_fd.astype(np.complex64)
        )
        channel_estimates_fd_raw_virtual.append(
            received.h_fd_virtual_avg_raw.astype(np.complex64)
        )
        channel_estimates_fd_aligned_virtual.append(
            received.h_fd_virtual_avg_aligned.astype(np.complex64)
        )
        channel_mean_fd.append(mean_channel.astype(np.complex64))
        channel_impulse_response_td.append(
            impulse_response.astype(np.complex64)
        )
        channel_full_spectrum_fd.append(full_spectrum.astype(np.complex64))
        channel_impulse_response_td_single_pilot.append(
            single_impulse_response.astype(np.complex64)
        )
        channel_full_spectrum_fd_single_pilot.append(
            single_full_spectrum.astype(np.complex64)
        )
        virtual_pilot_variance_raw.append(
            received.virtual_pilot_variance_raw.astype(np.float32)
        )
        virtual_pilot_variance_aligned.append(
            received.virtual_pilot_variance_aligned.astype(np.float32)
        )
        virtual_pilot_common_phases.append(
            received.virtual_pilot_common_phases_rad.astype(np.float32)
        )
        single_pilot_snr_db.append(received.single_pilot_snr_db)
        virtual_pilot_snr_db.append(received.virtual_pilot_snr_db)
        virtual_pilot_snr_gain_db.append(received.virtual_pilot_snr_gain_db)
        virtual_pilot_correction_phase_rad.append(
            received.virtual_pilot_correction_phase_rad
        )
        delay_profile_residual_gain_db.append(float(residual_gain_db))
        sync_indices.append(received.sync_idx)
        ffo_estimates_hz.append(received.ffo_hz)
        rfo_estimates_hz.append(received.rfo_hz_per_slot.astype(np.float32))
        timing_correlation_peaks.append(received.timing_correlation_peak)

        if capture_index == 0:
            example_iq_rcv = received.iq_rcv.astype(np.complex64)
            example_iq_rcv_single = received.iq_rcv_single.astype(np.complex64)
            example_frame_rcv = received.frame_rcv.astype(np.complex64)

    # OFDMConfig rejects zero captures, so these are guaranteed to be set.
    assert example_iq_rcv is not None
    assert example_iq_rcv_single is not None
    assert example_frame_rcv is not None

    return {
        "channel_estimates_fd": np.stack(channel_estimates_fd, axis=0),
        "channel_estimates_fd_single_pilot": np.stack(
            channel_estimates_fd_single_pilot,
            axis=0,
        ),
        "channel_estimates_fd_raw_virtual": np.stack(
            channel_estimates_fd_raw_virtual,
            axis=0,
        ),
        "channel_estimates_fd_aligned_virtual": np.stack(
            channel_estimates_fd_aligned_virtual,
            axis=0,
        ),
        "channel_mean_fd": np.stack(channel_mean_fd, axis=0),
        "channel_impulse_response_td": np.stack(
            channel_impulse_response_td,
            axis=0,
        ),
        "channel_full_spectrum_fd": np.stack(
            channel_full_spectrum_fd,
            axis=0,
        ),
        "channel_impulse_response_td_single_pilot": np.stack(
            channel_impulse_response_td_single_pilot,
            axis=0,
        ),
        "channel_full_spectrum_fd_single_pilot": np.stack(
            channel_full_spectrum_fd_single_pilot,
            axis=0,
        ),
        "virtual_pilot_variance_raw": np.stack(
            virtual_pilot_variance_raw,
            axis=0,
        ),
        "virtual_pilot_variance_aligned": np.stack(
            virtual_pilot_variance_aligned,
            axis=0,
        ),
        "virtual_pilot_common_phases_rad": np.stack(
            virtual_pilot_common_phases,
            axis=0,
        ),
        "single_pilot_snr_db": np.asarray(
            single_pilot_snr_db,
            dtype=np.float32,
        ),
        "virtual_pilot_snr_db": np.asarray(
            virtual_pilot_snr_db,
            dtype=np.float32,
        ),
        "virtual_pilot_snr_gain_db": np.asarray(
            virtual_pilot_snr_gain_db,
            dtype=np.float32,
        ),
        "virtual_pilot_correction_phase_rad": np.asarray(
            virtual_pilot_correction_phase_rad,
            dtype=np.float32,
        ),
        "delay_profile_residual_gain_db": np.asarray(
            delay_profile_residual_gain_db,
            dtype=np.float32,
        ),
        "sync_indices": np.asarray(sync_indices, dtype=np.int32),
        "ffo_estimates_hz": np.asarray(
            ffo_estimates_hz,
            dtype=np.float32,
        ),
        "rfo_estimates_hz": np.stack(rfo_estimates_hz, axis=0).astype(
            np.float32
        ),
        "timing_correlation_peaks": np.asarray(
            timing_correlation_peaks,
            dtype=np.float32,
        ),
        "example_iq_rcv": example_iq_rcv,
        "example_iq_rcv_single": example_iq_rcv_single,
        "example_frame_rcv": example_frame_rcv,
    }
