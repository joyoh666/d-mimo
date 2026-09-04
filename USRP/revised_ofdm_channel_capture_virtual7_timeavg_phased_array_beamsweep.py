"""RU/DFT-beam-resolved downlink CSI capture with a phased array.

The OFDM frame, synchronization, seven-virtual-pilot estimator, and most saved
field names follow ``revised_ofdm_channel_capture_virtual7_timeavg.py``.

Capture order
-------------
For every sweep, each RU is activated in turn.  Every column of the DFT
codebook is then applied to that RU, the array is allowed to settle, and one
OFDM frame is captured.  The saved effective downlink channel is therefore

    H_eff[sweep, ru, beam, subframe, slot, active_subcarrier]

and ``channel_mean_fd`` removes the repeated subframe/slot dimensions:

    H_eff_mean[sweep, ru, beam, active_subcarrier].

Two beamforming backends are supported:

``external``
    A single USRP RF port feeds each analog/hybrid phased array.  A configured
    external command receives a JSON file containing the RU, beam index, DFT
    weights, magnitudes, and phases.  The vendor-specific command must program
    the phase shifters before it exits.

``digital``
    One USRP TX channel is connected to every array element.  The DFT weights
    are multiplied into the baseband waveform directly.  Each RU configuration
    must then list exactly Mx*My*Mz TX channels.

Important: the repository does not contain the phased-array vendor API or its
register protocol.  For ``external`` mode, set ``PHASED_ARRAY_COMMAND`` below
to the real controller program.  The script deliberately refuses to acquire
with an unconfigured controller so that a mislabeled all-omnidirectional data
set cannot be produced silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

import revised_ofdm_channel_capture_virtual7_timeavg as original
from DFT_codebook_generator import UPA_codebook_generator_DFT


# =============================================================================
# User-configurable phased-array/RU settings
# =============================================================================
# "external": one RF feed per array plus a vendor array controller.
# "digital" : one USRP TX channel per antenna element.
BEAMFORMING_MODE = "external"

# Array geometry.  Change this to the physical element layout before capture.
ARRAY_SHAPE = (4, 4, 1)  # (Mx, My, Mz)
CODEBOOK_OVERSAMPLING = (1, 1, 1)
ANTENNA_SPACING_WAVELENGTH = 0.5

# Keep False to use the columns returned by DFT_codebook_generator.py exactly.
# Some TX array controllers use the conjugate steering-vector convention.
CONJUGATE_CODEBOOK_FOR_TX = False
BEAM_SETTLING_TIME_S = 0.020


@dataclass(frozen=True)
class RUConfig:
    """Mapping between a logical RU, USRP TX channel(s), and an array ID."""

    ru_id: str
    tx_channels: tuple[int, ...]
    array_id: str


# This one-RU entry exactly preserves the original script's TX-channel mapping.
# Add one entry per RU, for example in external mode:
#   RUConfig("RU1", (1,), "array_1"),
#   RUConfig("RU2", (2,), "array_2"),
# In digital mode, tx_channels must contain one channel per array element.
RU_CONFIGS = (
    RUConfig("RU0", (0,), "array_0"),
)

RX_CHANNEL = 0

# Same UHD device and RF frontend defaults as the original script.
USRP_DEVICE_ARGS = (
    "addr0=192.168.10.2,second_addr=192.168.11.2,"
    "third_addr=192.168.12.2,fourth_addr=192.168.13.2"
)
TX_SUBDEV_SPECS_BY_MBOARD = {0: "A:0"}
RX_SUBDEV_SPECS_BY_MBOARD = {0: "B:0"}
TX_ANTENNA = "TX/RX"
RX_ANTENNA = "TX/RX"

# Vendor-neutral external-controller bridge.  Each list item is passed to
# subprocess without a shell.  Available placeholders are:
#   {config_path}, {ru_id}, {array_id}, {beam_index}, {beam_x}, {beam_y}, {beam_z}
# Example:
# PHASED_ARRAY_COMMAND = (
#     "python3", "my_array_controller.py", "apply", "--config", "{config_path}",
# )
PHASED_ARRAY_COMMAND: tuple[str, ...] | None = None

# Optional command that disables every array after a sweep/capture session.
# It may use {config_path}.  Leaving this unset is safe only when inactive RUs
# cannot radiate without a scheduled USRP waveform.
PHASED_ARRAY_DISABLE_COMMAND: tuple[str, ...] | None = None

# The original file's capture count is retained, now interpreted as complete
# RU-by-beam sweeps rather than captures of one fixed antenna state.
NUM_BEAM_SWEEPS = original.NUM_CHANNEL_CAPTURES
OUTPUT_DIR = "saved_channel_estimates"
SHOW_PLOTS = True


# =============================================================================
# Beam-controller backends
# =============================================================================
class BeamController:
    def set_beam(
        self,
        ru: RUConfig,
        beam_index: int,
        beam_axis_index: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def disable_all(self) -> None:
        return None


class DigitalBeamController(BeamController):
    """No external action is needed; weights are applied to TX waveforms."""

    def set_beam(
        self,
        ru: RUConfig,
        beam_index: int,
        beam_axis_index: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        return None


class ExternalCommandBeamController(BeamController):
    """Pass one atomic JSON beam command to a vendor-specific executable."""

    def __init__(
        self,
        command_template: Sequence[str] | None,
        disable_command_template: Sequence[str] | None,
        output_dir: str,
    ) -> None:
        if not command_template:
            raise RuntimeError(
                "BEAMFORMING_MODE='external' requires PHASED_ARRAY_COMMAND. "
                "Configure the vendor controller command before RF capture."
            )
        self.command_template = tuple(command_template)
        self.disable_command_template = (
            tuple(disable_command_template) if disable_command_template else None
        )
        self.config_path = Path(output_dir).resolve() / "active_phased_array_beam.json"

    @staticmethod
    def _format_command(template: Sequence[str], values: dict[str, object]) -> list[str]:
        return [str(part).format(**values) for part in template]

    def set_beam(
        self,
        ru: RUConfig,
        beam_index: int,
        beam_axis_index: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        beam_x, beam_y, beam_z = [int(v) for v in beam_axis_index]
        weights = np.asarray(weights, dtype=np.complex64)
        payload = {
            "command": "set_beam",
            "timestamp_unix_s": time.time(),
            "ru_id": ru.ru_id,
            "array_id": ru.array_id,
            "beam_index": int(beam_index),
            "beam_axis_index": [beam_x, beam_y, beam_z],
            "weight_order": "x_fastest_then_y_then_z",
            "weights_real": np.real(weights).astype(float).tolist(),
            "weights_imag": np.imag(weights).astype(float).tolist(),
            "weights_magnitude": np.abs(weights).astype(float).tolist(),
            "weights_phase_rad": np.angle(weights).astype(float).tolist(),
        }

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.config_path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        os.replace(temporary_path, self.config_path)

        values = {
            "config_path": str(self.config_path),
            "ru_id": ru.ru_id,
            "array_id": ru.array_id,
            "beam_index": beam_index,
            "beam_x": beam_x,
            "beam_y": beam_y,
            "beam_z": beam_z,
        }
        subprocess.run(
            self._format_command(self.command_template, values),
            check=True,
        )

    def disable_all(self) -> None:
        if self.disable_command_template is None:
            return
        subprocess.run(
            self._format_command(
                self.disable_command_template,
                {"config_path": str(self.config_path)},
            ),
            check=True,
        )


def create_beam_controller(output_dir: str) -> BeamController:
    if BEAMFORMING_MODE == "digital":
        return DigitalBeamController()
    if BEAMFORMING_MODE == "external":
        return ExternalCommandBeamController(
            PHASED_ARRAY_COMMAND,
            PHASED_ARRAY_DISABLE_COMMAND,
            output_dir,
        )
    raise ValueError(f"Unsupported BEAMFORMING_MODE: {BEAMFORMING_MODE!r}")


# =============================================================================
# Configuration, waveform routing, and USRP setup
# =============================================================================
def generate_tx_codebook() -> tuple[np.ndarray, np.ndarray]:
    Mx, My, Mz = ARRAY_SHAPE
    ox, oy, oz = CODEBOOK_OVERSAMPLING
    codebook, beam_axis_indices = UPA_codebook_generator_DFT(
        Mx,
        My,
        Mz,
        oversampling_x=ox,
        oversampling_y=oy,
        oversampling_z=oz,
        ant_spacing=ANTENNA_SPACING_WAVELENGTH,
    )
    if CONJUGATE_CODEBOOK_FOR_TX:
        codebook = np.conj(codebook)
    return codebook.astype(np.complex64), beam_axis_indices.astype(np.int32)


def validate_configuration(codebook: np.ndarray) -> None:
    if BEAMFORMING_MODE not in {"external", "digital"}:
        raise ValueError("BEAMFORMING_MODE must be 'external' or 'digital'")
    if not RU_CONFIGS:
        raise ValueError("RU_CONFIGS must contain at least one RU")
    if len({ru.ru_id for ru in RU_CONFIGS}) != len(RU_CONFIGS):
        raise ValueError("Every RUConfig.ru_id must be unique")
    if len({ru.array_id for ru in RU_CONFIGS}) != len(RU_CONFIGS):
        raise ValueError("Every RUConfig.array_id must be unique")
    if RX_CHANNEL < 0:
        raise ValueError("RX_CHANNEL must be non-negative")

    num_elements = codebook.shape[0]
    for ru in RU_CONFIGS:
        if not ru.tx_channels or any(channel < 0 for channel in ru.tx_channels):
            raise ValueError(f"Invalid TX channels for {ru.ru_id}: {ru.tx_channels}")
        required_channels = num_elements if BEAMFORMING_MODE == "digital" else 1
        if len(ru.tx_channels) != required_channels:
            raise ValueError(
                f"{ru.ru_id} needs {required_channels} TX channel(s) in "
                f"{BEAMFORMING_MODE!r} mode, got {len(ru.tx_channels)}"
            )


def configure_usrp(usrp) -> None:
    num_mboards = usrp.get_num_mboards()
    for mboard, spec in TX_SUBDEV_SPECS_BY_MBOARD.items():
        if mboard >= num_mboards:
            raise ValueError(f"TX subdevice mboard {mboard} does not exist")
        usrp.set_tx_subdev_spec(original.uhd.usrp.SubdevSpec(spec), mboard)
    for mboard, spec in RX_SUBDEV_SPECS_BY_MBOARD.items():
        if mboard >= num_mboards:
            raise ValueError(f"RX subdevice mboard {mboard} does not exist")
        usrp.set_rx_subdev_spec(original.uhd.usrp.SubdevSpec(spec), mboard)

    tx_channels = sorted({channel for ru in RU_CONFIGS for channel in ru.tx_channels})
    if tx_channels[-1] >= usrp.get_tx_num_channels():
        raise ValueError(
            f"Configured TX channel {tx_channels[-1]} but UHD exposes only "
            f"{usrp.get_tx_num_channels()} channel(s)"
        )
    if RX_CHANNEL >= usrp.get_rx_num_channels():
        raise ValueError(
            f"Configured RX channel {RX_CHANNEL} but UHD exposes only "
            f"{usrp.get_rx_num_channels()} channel(s)"
        )

    for channel in tx_channels:
        usrp.set_tx_antenna(TX_ANTENNA, channel)
    usrp.set_rx_antenna(RX_ANTENNA, RX_CHANNEL)


def waveform_for_beam(
    base_waveform: np.ndarray,
    beam_weights: np.ndarray,
) -> np.ndarray:
    if BEAMFORMING_MODE == "external":
        return base_waveform.copy().astype(np.complex64)

    # The DFT codewords have unit L2 norm, so this preserves total digital
    # transmit energy across elements relative to the original one-port signal.
    beam_weights = np.asarray(beam_weights, dtype=np.complex64)
    return (beam_weights[:, None] * base_waveform[0][None, :]).astype(np.complex64)


# =============================================================================
# Original receive/CSI processing with selectable TX/RX channel routing
# =============================================================================
def receive_and_process_ru_beam(
    usrp,
    params: dict[str, object],
    waveform: np.ndarray,
    tx_channels: Sequence[int],
    rx_channel: int,
    ss_td_with_cp: np.ndarray,
    pdsch_idx: tuple[np.ndarray, ...],
    known_ref_seq: np.ndarray,
) -> dict[str, object]:
    """Run the original estimator for one RU/beam capture event."""
    N = params["N"]
    FFT_SIZE = params["FFT_SIZE"]
    sampling_rate = params["sampling_rate"]
    frame_length = params["frame_length"]
    slot_length = params["slot_length"]
    first_CP_length = params["first_CP_length"]
    normal_CP_length = params["normal_CP_length"]

    frame_rcv = original.sendAndReceive(
        usrp,
        waveform,
        1,
        original.carrier_frequency,
        sampling_rate,
        original.Tx_gain,
        original.Rx_gain,
        list(tx_channels),
        [rx_channel],
        wait_time=0.2,
        tx_delay_samples=int(sampling_rate * 1e-4),
        rx_trailing_samples=int(sampling_rate * 1e-3),
        otw_format="sc16",
    )
    frame_rcv = frame_rcv[:, 10:]
    frame_rcv -= np.mean(frame_rcv)
    frame_rcv = frame_rcv[0].flatten()

    frame_rcv_ifosync = frame_rcv
    corr = np.abs(correlate(frame_rcv_ifosync, ss_td_with_cp, "valid", "fft"))
    ss_start_idx = FFT_SIZE * 5 + first_CP_length + normal_CP_length * 4
    sync_idx = int(np.argmax(corr) - ss_start_idx)
    if sync_idx + frame_length > frame_rcv.size or sync_idx < 0:
        print("sync not found, using sync_idx = 0")
        sync_idx = 0

    frame_rcv_timesync = frame_rcv_ifosync[sync_idx:sync_idx + frame_length]
    if frame_rcv_timesync.size < frame_length:
        frame_rcv_timesync = np.concatenate(
            [
                frame_rcv_timesync.astype(np.complex64),
                np.zeros(frame_length - frame_rcv_timesync.size, dtype=np.complex64),
            ]
        )

    td_symbols_rcv = np.reshape(
        frame_rcv_timesync,
        (original.num_subframe_per_frame * original.num_slot_per_subframe, -1),
    )
    first_CPs = td_symbols_rcv[:, :first_CP_length]
    first_signal_for_CPs = td_symbols_rcv[:, FFT_SIZE:FFT_SIZE + first_CP_length]

    td_symbols_rcv_wo_first = td_symbols_rcv[:, FFT_SIZE + first_CP_length:]
    td_symbols_rcv_wo_first = np.reshape(
        td_symbols_rcv_wo_first,
        (
            original.num_subframe_per_frame
            * original.num_slot_per_subframe
            * (original.num_symbols_per_slot - 1),
            -1,
        ),
    )
    other_CPs = td_symbols_rcv_wo_first[:, :normal_CP_length]
    other_signal_for_CPs = td_symbols_rcv_wo_first[:, -normal_CP_length:]
    phase_diff = np.angle(
        np.sum(np.conj(other_CPs) * other_signal_for_CPs)
        + np.sum(np.conj(first_CPs) * first_signal_for_CPs)
    )
    ffo = phase_diff * sampling_rate / (2 * np.pi * FFT_SIZE)
    sample_index = np.arange(frame_rcv_timesync.size)
    compensation_ffo = np.exp(
        -1j * ffo * sample_index / sampling_rate * 2 * np.pi
    ).astype(np.complex64)
    frame_rcv_ffosync = frame_rcv_timesync * compensation_ffo

    refsym_rcv_td = np.reshape(
        frame_rcv_ffosync,
        (original.num_subframe_per_frame * original.num_slot_per_subframe, -1),
    )
    refsym_rcv_td = refsym_rcv_td[:, first_CP_length:first_CP_length + FFT_SIZE]
    phase_diff = np.angle(
        np.sum(np.conj(refsym_rcv_td[:-1]) * refsym_rcv_td[1:], axis=-1)
    )
    rfo = phase_diff * sampling_rate / (2 * np.pi * slot_length)
    rfo = np.expand_dims(np.concatenate([rfo, rfo[-1:]], axis=0), axis=1)
    slot_sample_index = np.expand_dims(np.arange(slot_length), axis=0)
    compensation_rfo = np.exp(
        -1j * rfo * slot_sample_index / sampling_rate * 2 * np.pi
    ).astype(np.complex64)
    frame_rcv_rfosync = frame_rcv_ffosync * compensation_rfo.flatten()

    fft_symbols_rcv = np.reshape(
        frame_rcv_rfosync,
        (original.num_subframe_per_frame * original.num_slot_per_subframe, -1),
    )
    fft_symbols_rcv = fft_symbols_rcv[:, first_CP_length - normal_CP_length:]
    fft_symbols_rcv = np.reshape(
        fft_symbols_rcv,
        (
            original.num_subframe_per_frame
            * original.num_slot_per_subframe
            * original.num_symbols_per_slot,
            -1,
        ),
    )
    fft_symbols_rcv = np.fft.fft(
        fft_symbols_rcv[:, normal_CP_length:], axis=-1, norm="ortho"
    )
    fft_symbols_rcv = np.reshape(
        fft_symbols_rcv,
        (
            original.num_subframe_per_frame,
            original.num_slot_per_subframe,
            original.num_symbols_per_slot,
            FFT_SIZE,
        ),
    )

    resource_maps_rcv = np.zeros(
        (
            original.num_subframe_per_frame,
            original.num_slot_per_subframe,
            original.num_symbols_per_slot,
            N,
        ),
        dtype=np.complex64,
    )
    resource_maps_rcv[..., N // 2:] = fft_symbols_rcv[..., 1:N // 2 + 1]
    resource_maps_rcv[..., :N // 2] = fft_symbols_rcv[..., -(N // 2):]

    h_virtual_pilots = []
    virtual_pilot_symbols = []
    virtual_pilot_positions_used = None
    for sf_idx, slot_idx, sym_idx in original.get_virtual_pilot_positions():
        if not (
            0 <= sf_idx < original.num_subframe_per_frame
            and 0 <= slot_idx < original.num_slot_per_subframe
            and 0 <= sym_idx < original.num_symbols_per_slot
        ):
            continue
        h_virtual_pilots.append(
            resource_maps_rcv[sf_idx, slot_idx, sym_idx, :] / known_ref_seq
        )
        virtual_pilot_symbols.append(resource_maps_rcv[sf_idx, slot_idx, sym_idx, :])
        virtual_pilot_positions_used = (sf_idx, slot_idx, sym_idx)

    if not h_virtual_pilots:
        raise RuntimeError("No valid virtual-pilot position is configured")

    h_virtual_pilots = np.stack(h_virtual_pilots, axis=0).astype(np.complex64)
    h_fd_virtual_avg_raw = np.mean(h_virtual_pilots, axis=0).astype(np.complex64)
    h_virtual_pilots_aligned, common_phases = original.phase_align_channel_estimates(
        h_virtual_pilots
    )
    if original.PHASE_ALIGN_VIRTUAL_PILOTS:
        h_fd_virtual_avg = np.mean(h_virtual_pilots_aligned, axis=0).astype(np.complex64)
    else:
        h_fd_virtual_avg = h_fd_virtual_avg_raw.copy()

    variance_raw = np.mean(
        np.abs(h_virtual_pilots - h_fd_virtual_avg_raw[None, :]) ** 2,
        axis=0,
    ).astype(np.float32)
    variance_aligned = np.mean(
        np.abs(h_virtual_pilots_aligned - h_fd_virtual_avg[None, :]) ** 2,
        axis=0,
    ).astype(np.float32)

    h_fd = np.broadcast_to(
        h_fd_virtual_avg.reshape(1, 1, -1),
        (original.num_subframe_per_frame, original.num_slot_per_subframe, N),
    ).copy()
    h_single_fd = h_virtual_pilots[0]
    h_single_fd_full = np.broadcast_to(
        h_single_fd.reshape(1, 1, -1),
        (original.num_subframe_per_frame, original.num_slot_per_subframe, N),
    ).copy()

    resource_maps_rcv_zf = resource_maps_rcv / (h_fd[:, :, None, :] + 1e-12)
    iq_rcv = resource_maps_rcv_zf[pdsch_idx]
    resource_maps_rcv_zf_single = resource_maps_rcv / (
        h_single_fd_full[:, :, None, :] + 1e-12
    )
    iq_rcv_single = resource_maps_rcv_zf_single[pdsch_idx]

    sfn_ref, slot_ref, _ = virtual_pilot_positions_used
    pilot_eq_ref = resource_maps_rcv_zf[sfn_ref, slot_ref, 0, :] / known_ref_seq
    correction_phase = float(np.angle(np.mean(pilot_eq_ref)))
    correction = np.complex64(np.exp(-1j * correction_phase))
    iq_rcv *= correction
    iq_rcv_single *= correction

    virtual_pilot_received = np.stack(virtual_pilot_symbols, axis=0)
    h_gain_raw = virtual_pilot_received / h_fd_virtual_avg.reshape(1, -1)
    h_gain_single = virtual_pilot_received / h_single_fd.reshape(1, -1)
    single_snr_db = original._pilot_snr_db_from_equalized(
        h_gain_single, known_ref_seq
    )
    virtual_snr_db = original._pilot_snr_db_from_equalized(h_gain_raw, known_ref_seq)

    return {
        "frame_rcv": frame_rcv.astype(np.complex64),
        "h_fd": h_fd.astype(np.complex64),
        "h_single_fd": h_single_fd_full.astype(np.complex64),
        "h_fd_virtual_avg_raw": h_fd_virtual_avg_raw.astype(np.complex64),
        "h_fd_virtual_avg_aligned": h_fd_virtual_avg.astype(np.complex64),
        "h_virtual_pilots_fd": h_virtual_pilots.astype(np.complex64),
        "h_virtual_pilots_fd_aligned": h_virtual_pilots_aligned.astype(np.complex64),
        "virtual_pilot_variance_raw": variance_raw,
        "virtual_pilot_variance_aligned": variance_aligned,
        "virtual_pilot_common_phases_rad": common_phases.astype(np.float32),
        "single_pilot_snr_db": float(single_snr_db),
        "virtual_pilot_snr_db": float(virtual_snr_db),
        "virtual_pilot_snr_gain_db": float(virtual_snr_db - single_snr_db),
        "virtual_pilot_correction_phase_rad": correction_phase,
        "iq_rcv": iq_rcv.astype(np.complex64),
        "iq_rcv_single": iq_rcv_single.astype(np.complex64),
        "sync_idx": sync_idx,
        "ffo_hz": float(ffo),
        "rfo_hz_per_slot": rfo.flatten().astype(np.float32),
        "timing_correlation_peak": float(np.max(corr)) if corr.size else 0.0,
    }


# =============================================================================
# Sweep collection and array assembly
# =============================================================================
def _stack_record_grid(
    records: list[list[list[dict[str, object]]]],
    key: str,
    dtype=None,
) -> np.ndarray:
    array = np.stack(
        [
            np.stack(
                [np.stack([beam[key] for beam in ru], axis=0) for ru in sweep],
                axis=0,
            )
            for sweep in records
        ],
        axis=0,
    )
    return array.astype(dtype) if dtype is not None else array


def collect_sweeps(
    usrp,
    controller: BeamController,
    params: dict[str, object],
    tx_objects: dict[str, object],
    known_ref_seq: np.ndarray,
    codebook: np.ndarray,
    beam_axis_indices: np.ndarray,
    num_sweeps: int,
) -> dict[str, np.ndarray]:
    records: list[list[list[dict[str, object]]]] = []
    num_events = num_sweeps * len(RU_CONFIGS) * codebook.shape[1]
    event_index = 0

    try:
        controller.disable_all()
        for sweep_index in range(num_sweeps):
            sweep_records = []
            for ru_index, ru in enumerate(RU_CONFIGS):
                ru_records = []
                for beam_index in range(codebook.shape[1]):
                    event_index += 1
                    axis_index = beam_axis_indices[beam_index]
                    weights = codebook[:, beam_index]
                    print(
                        f"[BW {params['bandwidth_mhz']:>5} MHz] "
                        f"sweep={sweep_index + 1}/{num_sweeps} "
                        f"RU={ru.ru_id} ({ru_index + 1}/{len(RU_CONFIGS)}) "
                        f"beam={beam_index + 1}/{codebook.shape[1]} "
                        f"axis={axis_index.tolist()} event={event_index}/{num_events}"
                    )

                    controller.set_beam(ru, beam_index, axis_index, weights)
                    if BEAMFORMING_MODE == "external" and BEAM_SETTLING_TIME_S > 0:
                        time.sleep(BEAM_SETTLING_TIME_S)

                    capture_time = time.time()
                    tx_waveform = waveform_for_beam(tx_objects["waveform"], weights)
                    rx = receive_and_process_ru_beam(
                        usrp=usrp,
                        params=params,
                        waveform=tx_waveform,
                        tx_channels=ru.tx_channels,
                        rx_channel=RX_CHANNEL,
                        ss_td_with_cp=tx_objects["ss_td_with_cp"],
                        pdsch_idx=tx_objects["pdsch_idx"],
                        known_ref_seq=known_ref_seq,
                    )

                    h_mean = np.mean(rx["h_fd"], axis=(0, 1)).astype(np.complex64)
                    h_single = np.mean(rx["h_single_fd"], axis=(0, 1)).astype(
                        np.complex64
                    )
                    h_delay, h_full = original.channel_to_delay_response(
                        h_mean, int(params["N"]), int(params["FFT_SIZE"])
                    )
                    h_delay_single, h_full_single = original.channel_to_delay_response(
                        h_single, int(params["N"]), int(params["FFT_SIZE"])
                    )
                    _, residual_single, _ = original._delay_residual_power_db(
                        h_delay_single,
                        guard_taps=max(1, int(round(int(params["FFT_SIZE"]) / 200))),
                    )
                    _, residual_avg, _ = original._delay_residual_power_db(
                        h_delay,
                        guard_taps=max(1, int(round(int(params["FFT_SIZE"]) / 200))),
                    )

                    rx["channel_mean_fd"] = h_mean
                    rx["channel_impulse_response_td"] = h_delay
                    rx["channel_full_spectrum_fd"] = h_full
                    rx["channel_impulse_response_td_single_pilot"] = h_delay_single
                    rx["channel_full_spectrum_fd_single_pilot"] = h_full_single
                    rx["delay_profile_residual_gain_db"] = float(
                        10.0
                        * np.log10((residual_single + 1e-15) / (residual_avg + 1e-15))
                    )
                    rx["capture_event_unix_s"] = capture_time
                    ru_records.append(rx)
                sweep_records.append(ru_records)
            records.append(sweep_records)
    finally:
        controller.disable_all()

    complex_keys = {
        "channel_estimates_fd": "h_fd",
        "channel_estimates_fd_single_pilot": "h_single_fd",
        "channel_estimates_fd_raw_virtual": "h_fd_virtual_avg_raw",
        "channel_estimates_fd_aligned_virtual": "h_fd_virtual_avg_aligned",
        "channel_estimates_fd_per_virtual_pilot": "h_virtual_pilots_fd",
        "channel_estimates_fd_per_virtual_pilot_aligned": "h_virtual_pilots_fd_aligned",
        "channel_mean_fd": "channel_mean_fd",
        "channel_impulse_response_td": "channel_impulse_response_td",
        "channel_full_spectrum_fd": "channel_full_spectrum_fd",
        "channel_impulse_response_td_single_pilot": "channel_impulse_response_td_single_pilot",
        "channel_full_spectrum_fd_single_pilot": "channel_full_spectrum_fd_single_pilot",
    }
    results = {
        output_key: _stack_record_grid(records, record_key, np.complex64)
        for output_key, record_key in complex_keys.items()
    }

    float_keys = {
        "virtual_pilot_variance_raw": "virtual_pilot_variance_raw",
        "virtual_pilot_variance_aligned": "virtual_pilot_variance_aligned",
        "virtual_pilot_common_phases_rad": "virtual_pilot_common_phases_rad",
        "single_pilot_snr_db": "single_pilot_snr_db",
        "virtual_pilot_snr_db": "virtual_pilot_snr_db",
        "virtual_pilot_snr_gain_db": "virtual_pilot_snr_gain_db",
        "virtual_pilot_correction_phase_rad": "virtual_pilot_correction_phase_rad",
        "delay_profile_residual_gain_db": "delay_profile_residual_gain_db",
        # Retain the original archive field names.
        "ffo_estimates_hz": "ffo_hz",
        "rfo_estimates_hz": "rfo_hz_per_slot",
        "timing_correlation_peaks": "timing_correlation_peak",
    }
    for output_key, record_key in float_keys.items():
        results[output_key] = _stack_record_grid(records, record_key, np.float32)
    results["sync_indices"] = _stack_record_grid(records, "sync_idx", np.int32)
    results["capture_event_unix_s"] = _stack_record_grid(
        records, "capture_event_unix_s", np.float64
    )

    first = records[0][0][0]
    results["example_rx_constellation"] = np.asarray(
        first["iq_rcv"], dtype=np.complex64
    )
    results["example_rx_constellation_single_pilot"] = np.asarray(
        first["iq_rcv_single"], dtype=np.complex64
    )
    results["example_rx_frame"] = np.asarray(
        first["frame_rcv"], dtype=np.complex64
    )

    beam_power_linear = np.mean(np.abs(results["channel_mean_fd"]) ** 2, axis=-1)
    best_beam_indices = np.argmax(beam_power_linear, axis=2).astype(np.int32)
    sweep_grid = np.arange(num_sweeps)[:, None]
    ru_grid = np.arange(len(RU_CONFIGS))[None, :]
    results["beam_power_linear"] = beam_power_linear.astype(np.float32)
    results["best_beam_indices"] = best_beam_indices
    results["best_beam_axis_indices"] = beam_axis_indices[best_beam_indices]
    results["best_beam_channel_mean_fd"] = results["channel_mean_fd"][
        sweep_grid, ru_grid, best_beam_indices, :
    ].astype(np.complex64)
    results["mean_best_beam_indices_per_ru"] = np.argmax(
        np.mean(beam_power_linear, axis=0), axis=1
    ).astype(np.int32)
    return results


# =============================================================================
# Saving and plotting
# =============================================================================
def save_results(
    output_dir: str,
    params: dict[str, object],
    known_ref_seq: np.ndarray,
    codebook: np.ndarray,
    beam_axis_indices: np.ndarray,
    results: dict[str, np.ndarray],
) -> tuple[str, str, str, dict[str, object]]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bw_tag = f"BW_{str(params['bandwidth_mhz']).replace('.', 'p')}MHz"
    prefix = f"beam_channel_dataset_{bw_tag}_{timestamp}"
    npz_path = os.path.join(output_dir, f"{prefix}.npz")
    json_path = os.path.join(output_dir, f"{prefix}_metadata.json")
    fig_path = os.path.join(output_dir, f"beam_channel_plots_{bw_tag}_{timestamp}.png")

    codebook_hash = hashlib.sha256(
        np.ascontiguousarray(codebook).view(np.uint8)
    ).hexdigest()
    channel_shape = list(results["channel_estimates_fd"].shape)
    mean_shape = list(results["channel_mean_fd"].shape)
    delay_shape = list(results["channel_impulse_response_td"].shape)
    num_sweeps, num_rus, num_beams = results["beam_power_linear"].shape
    reference_seed = original.REFERENCE_SEQUENCE_SEED + int(
        10 * float(params["bandwidth_mhz"])
    )

    metadata: dict[str, object] = {
        # Original waveform and estimator metadata.
        "timestamp": timestamp,
        "bandwidth_mhz": params["bandwidth_mhz"],
        "subcarrier_spacing_hz": params["delta_f"],
        "num_active_subcarriers": params["N"],
        "fft_size": params["FFT_SIZE"],
        "sampling_rate_hz": params["sampling_rate"],
        "carrier_frequency_hz": params["carrier_frequency"],
        "tx_gain_db": params["Tx_gain"],
        "rx_gain_db": params["Rx_gain"],
        "num_symbols_per_slot": params["num_symbols_per_slot"],
        "num_slots_per_subframe": params["num_slot_per_subframe"],
        "num_subframes_per_frame": params["num_subframe_per_frame"],
        "num_symbols_per_frame": params["num_symbols_frame"],
        "normal_cp_length_samples": params["normal_CP_length"],
        "first_cp_length_samples": params["first_CP_length"],
        "slot_length_samples": params["slot_length"],
        "frame_length_samples": params["frame_length"],
        "num_channel_captures": int(num_sweeps),
        "samples_per_channel_vector": params["N"],
        "samples_per_delay_response": params["FFT_SIZE"],
        "reference_sequence_type": "seeded_unit_power_qpsk",
        "reference_sequence_seed": original.REFERENCE_SEQUENCE_SEED,
        "effective_reference_sequence_seed": reference_seed,
        "virtual_pilot_count": params["num_virtual_pilots"],
        "virtual_pilot_positions": params["virtual_pilot_positions"],
        "phase_align_virtual_pilots": bool(original.PHASE_ALIGN_VIRTUAL_PILOTS),
        "expected_channel_estimation_snr_gain_db": float(
            10 * np.log10(int(params["num_virtual_pilots"]))
        ),
        "achieved_channel_estimation_snr_gain_db_mean": float(
            np.mean(results["virtual_pilot_snr_gain_db"])
        ),
        "mean_single_to_averaged_residual_delay_gain_db": float(
            np.mean(results["delay_profile_residual_gain_db"])
        ),
        "modulation_order": params["modulation_order"],
        "digital_power_scale": params["POWER"],
        # RU and phased-array additions.
        "measurement_type": "effective_downlink_channel_after_tx_beamforming",
        "sweep_order": "sweep_then_ru_then_beam",
        "beamforming_mode": BEAMFORMING_MODE,
        "num_beam_sweeps": int(num_sweeps),
        "num_rus": int(num_rus),
        "num_dft_beams": int(num_beams),
        "num_capture_events": int(num_sweeps * num_rus * num_beams),
        "ru_configs": [asdict(ru) for ru in RU_CONFIGS],
        "rx_channel": RX_CHANNEL,
        "array_shape_xyz": list(ARRAY_SHAPE),
        "codebook_oversampling_xyz": list(CODEBOOK_OVERSAMPLING),
        "antenna_spacing_wavelength": ANTENNA_SPACING_WAVELENGTH,
        "codebook_generator": "UPA_codebook_generator_DFT",
        "codebook_element_order": "x_fastest_then_y_then_z",
        "conjugate_codebook_for_tx": CONJUGATE_CODEBOOK_FOR_TX,
        "dft_codebook_shape": list(codebook.shape),
        "dft_codebook_sha256": codebook_hash,
        "beam_axis_indices_shape": list(beam_axis_indices.shape),
        "beam_settling_time_s": BEAM_SETTLING_TIME_S,
        "saved_channel_axis_order": [
            "sweep",
            "ru",
            "beam",
            "subframe",
            "slot",
            "active_subcarrier",
        ],
        "saved_channel_tensor_shape": channel_shape,
        "saved_mean_channel_axis_order": [
            "sweep",
            "ru",
            "beam",
            "active_subcarrier",
        ],
        "saved_mean_channel_shape": mean_shape,
        "saved_delay_response_axis_order": [
            "sweep",
            "ru",
            "beam",
            "delay_sample",
        ],
        "saved_delay_response_shape": delay_shape,
        "saved_single_pilot_delay_response_shape": list(
            results["channel_impulse_response_td_single_pilot"].shape
        ),
        "capture_event_time_axis_order": ["sweep", "ru", "beam"],
        "best_beam_metric": "mean_active_subcarrier_power",
        "complex_phase_reference": (
            "independent timing/CFO/virtual-pilot phase correction per RU-beam event; "
            "cross-beam absolute phase requires a shared calibration reference"
        ),
        "mean_best_beam_indices_per_ru": results[
            "mean_best_beam_indices_per_ru"
        ].astype(int).tolist(),
    }

    np.savez_compressed(
        npz_path,
        **results,
        dft_codebook=codebook.astype(np.complex64),
        beam_axis_indices=beam_axis_indices.astype(np.int32),
        ru_ids=np.asarray([ru.ru_id for ru in RU_CONFIGS]),
        known_reference_sequence=known_ref_seq.astype(np.complex64),
        metadata_json=json.dumps(metadata),
    )
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return npz_path, json_path, fig_path, metadata


def plot_beam_results(
    fig_path: str,
    params: dict[str, object],
    results: dict[str, np.ndarray],
) -> None:
    num_rus = len(RU_CONFIGS)
    fig, axes = plt.subplots(
        num_rus,
        2,
        figsize=(16, max(5, 4 * num_rus)),
        squeeze=False,
    )
    channel_power = np.mean(np.abs(results["channel_mean_fd"]) ** 2, axis=0)
    beam_power_db = 10 * np.log10(np.mean(channel_power, axis=-1) + 1e-15)

    for ru_index, ru in enumerate(RU_CONFIGS):
        spectrum_db = 10 * np.log10(channel_power[ru_index] + 1e-15)
        image = axes[ru_index, 0].imshow(
            spectrum_db,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        axes[ru_index, 0].set_title(f"{ru.ru_id}: beam-resolved |H[k]|^2")
        axes[ru_index, 0].set_xlabel("Active subcarrier index")
        axes[ru_index, 0].set_ylabel("DFT beam index")
        fig.colorbar(image, ax=axes[ru_index, 0], label="Power (dB)")

        best_beam = int(results["mean_best_beam_indices_per_ru"][ru_index])
        colors = ["tab:blue"] * beam_power_db.shape[1]
        colors[best_beam] = "tab:red"
        axes[ru_index, 1].bar(
            np.arange(beam_power_db.shape[1]), beam_power_db[ru_index], color=colors
        )
        axes[ru_index, 1].set_title(
            f"{ru.ru_id}: average beam power (best={best_beam})"
        )
        axes[ru_index, 1].set_xlabel("DFT beam index")
        axes[ru_index, 1].set_ylabel("Mean subcarrier power (dB)")
        axes[ru_index, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"RU/beam downlink CSI | BW={params['bandwidth_mhz']} MHz | "
        f"sweeps={results['channel_mean_fd'].shape[0]}"
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# =============================================================================
# Main execution
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweeps",
        type=int,
        default=NUM_BEAM_SWEEPS,
        help="number of complete RU-by-beam sweeps",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print the codebook/RU layout without opening a USRP",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="save CSI without generating summary figures",
    )
    args = parser.parse_args()
    if args.sweeps <= 0:
        parser.error("--sweeps must be positive")
    return args


def main() -> None:
    args = parse_args()
    codebook, beam_axis_indices = generate_tx_codebook()
    validate_configuration(codebook)
    print(
        f"DFT codebook ready: elements={codebook.shape[0]}, "
        f"beams={codebook.shape[1]}, RUs={len(RU_CONFIGS)}, "
        f"mode={BEAMFORMING_MODE}"
    )
    print("Saved CSI axis order: sweep, RU, beam, subframe, slot, subcarrier")
    if args.validate_only:
        print("Configuration validation completed; RF capture was not started.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    controller = create_beam_controller(args.output_dir)
    usrp = original.uhd.usrp.MultiUSRP(USRP_DEVICE_ARGS)
    usrp.set_time_unknown_pps(original.uhd.types.TimeSpec(0.0))
    configure_usrp(usrp)
    print("USRP loaded. Session Ready.")

    for bandwidth_mhz in original.CAPTURE_BANDWIDTHS:
        print("=" * 80)
        print(f"Running bandwidth setting: {bandwidth_mhz} MHz")
        params = original.get_system_params(bandwidth_mhz)
        effective_seed = original.REFERENCE_SEQUENCE_SEED + int(10 * bandwidth_mhz)
        known_ref_seq = original.generate_known_reference_sequence(
            int(params["N"]), seed=effective_seed
        )
        tx_objects = original.build_frame(params, known_ref_seq)

        results = collect_sweeps(
            usrp=usrp,
            controller=controller,
            params=params,
            tx_objects=tx_objects,
            known_ref_seq=known_ref_seq,
            codebook=codebook,
            beam_axis_indices=beam_axis_indices,
            num_sweeps=args.sweeps,
        )
        npz_path, json_path, fig_path, metadata = save_results(
            output_dir=args.output_dir,
            params=params,
            known_ref_seq=known_ref_seq,
            codebook=codebook,
            beam_axis_indices=beam_axis_indices,
            results=results,
        )
        if not args.no_plots:
            plot_beam_results(fig_path, params, results)

        print("Saved results:")
        print(f"  NPZ  : {npz_path}")
        print(f"  JSON : {json_path}")
        if not args.no_plots:
            print(f"  Plot : {fig_path}")
        print(f"  CSI shape: {metadata['saved_channel_tensor_shape']}")
        print(
            "  Mean best beam per RU: "
            f"{metadata['mean_best_beam_indices_per_ru']}"
        )


if __name__ == "__main__":
    main()
