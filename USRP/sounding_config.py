"""Configuration for the paper-compatible D-MIMO sounding experiment.

The dataclass separates values stated by the paper from implementation choices
that the paper excerpt does not specify.  Assumptions are intentionally kept
as fields so that every recorded dataset can store the exact configuration and
later experiments can change one choice without editing DSP code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gcd, isclose
from typing import Any


@dataclass(frozen=True, slots=True)
class SoundingConfig:
    """Complete waveform, hardware, and dataset configuration.

    Paper-specified values are grouped first.  Everything under
    ``Implementation assumptions`` must be reported as an experimental design
    choice rather than as a value taken from the paper.
    """

    # ------------------------------------------------------------------
    # Paper-specified parameters
    # ------------------------------------------------------------------
    center_frequency_hz: float = 2.2e9
    nominal_bandwidth_hz: float = 1.4e6
    subcarrier_spacing_hz: float = 15e3
    fft_size: int = 128
    csi_interval_s: float = 5e-3
    num_rus: int = 3
    antennas_per_ru: int = 2
    num_rx_antennas: int = 1
    pilot_repetitions: int = 3

    # ------------------------------------------------------------------
    # Implementation assumptions not specified by the paper excerpt
    # ------------------------------------------------------------------

    # LTE-like 1.4 MHz allocation.  Seventy-two active subcarriers occupy
    # 1.08 MHz; 1.4 MHz is the nominal RF channel bandwidth.
    num_active_subcarriers: int = 72

    # CP and synchronization layout inherited from the earlier prototype.
    cp_length_samples: int = 10
    zc_root: int = 25
    zc_length_samples: int = 139
    guard_length_samples: int = 16
    sync_tx_branch: int = 0

    # Centered FFT-bin indices: negative frequency, DC=0, positive frequency.
    # Each of the six TX branches owns one distinct pilot bin.
    pilot_centered_bins: tuple[int, ...] = (-30, -18, -6, 6, 18, 30)
    pilot_modulation: str = "qpsk"
    pilot_seed: int = 2026

    # RF/UHD operating assumptions.  Device addresses and serial numbers are
    # deployment-specific and should be supplied by the hardware launcher.
    tx_gain_db: float = 15.0
    rx_gain_db: float = 30.0
    digital_power_scale: float = 1.0 / 8.0
    cpu_sample_format: str = "fc32"
    wire_sample_format: str = "sc16"
    # 184.32 MHz / 96 = 1.92 MS/s exactly.  The OctoClock supplies the
    # 10 MHz reference from which each X310 synthesizes this master clock.
    master_clock_rate_hz: float = 184.32e6
    clock_source: str = "external"
    time_source: str = "external"
    startup_lead_time_s: float = 0.2

    # Dataset conventions.  The representative CSI time is the mean of the
    # useful-symbol centres of all repeated pilots.  Phase alignment between
    # different 5 ms observations is disabled to preserve physical evolution.
    csi_timestamp_reference: str = "mean_pilot_useful_centers"
    phase_align_pilot_repetitions: bool = False
    save_raw_iq: bool = False
    dataset_chunk_duration_s: float = 1.0
    output_directory: str = "sounding_datasets"

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    # Derived waveform and timing quantities
    # ------------------------------------------------------------------
    @property
    def num_tx_branches(self) -> int:
        return self.num_rus * self.antennas_per_ru

    @property
    def sampling_rate_hz(self) -> float:
        return self.fft_size * self.subcarrier_spacing_hz

    @property
    def sample_period_s(self) -> float:
        return 1.0 / self.sampling_rate_hz

    @property
    def useful_symbol_duration_s(self) -> float:
        return self.fft_size / self.sampling_rate_hz

    @property
    def cp_duration_s(self) -> float:
        return self.cp_length_samples / self.sampling_rate_hz

    @property
    def occupied_bandwidth_hz(self) -> float:
        return self.num_active_subcarriers * self.subcarrier_spacing_hz

    @property
    def csi_interval_samples(self) -> int:
        return round(self.csi_interval_s * self.sampling_rate_hz)

    @property
    def pilot_symbol_samples(self) -> int:
        return self.cp_length_samples + self.fft_size

    @property
    def pilot_block_samples(self) -> int:
        return self.pilot_repetitions * self.pilot_symbol_samples

    @property
    def pilot_block_cp_start(self) -> int:
        return self.zc_length_samples + self.guard_length_samples

    @property
    def pilot_block_data_start(self) -> int:
        return self.pilot_block_cp_start + self.cp_length_samples

    @property
    def sounding_burst_samples(self) -> int:
        return self.pilot_block_cp_start + self.pilot_block_samples

    @property
    def zero_padding_samples(self) -> int:
        """Unused samples after the sounding burst in each 5 ms period."""
        return self.csi_interval_samples - self.sounding_burst_samples

    @property
    def csi_timestamp_offset_samples(self) -> int:
        """Representative CSI sample offset within one 5 ms period."""
        first_useful_center = self.pilot_block_data_start + self.fft_size / 2
        last_useful_center = first_useful_center + (
            (self.pilot_repetitions - 1) * self.pilot_symbol_samples
        )
        return round((first_useful_center + last_useful_center) / 2)

    @property
    def dataset_chunk_csi_samples(self) -> int:
        """Number of CSI observations written per dataset chunk."""
        return round(self.dataset_chunk_duration_s / self.csi_interval_s)

    @property
    def branch_labels(self) -> tuple[str, ...]:
        return tuple(
            f"ru{ru_index}_ant{antenna_index}"
            for ru_index in range(self.num_rus)
            for antenna_index in range(self.antennas_per_ru)
        )

    def pilot_symbol_cp_start(self, repetition: int) -> int:
        """Return one repeated pilot symbol's CP-start sample offset."""
        if not 0 <= repetition < self.pilot_repetitions:
            raise IndexError("pilot repetition is outside the valid range")
        return self.pilot_block_cp_start + repetition * self.pilot_symbol_samples

    def pilot_symbol_data_start(self, repetition: int) -> int:
        """Return one repeated pilot symbol's useful-data sample offset."""
        return self.pilot_symbol_cp_start(repetition) + self.cp_length_samples

    # ------------------------------------------------------------------
    # Validation and serializable metadata
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Reject internally inconsistent or unsafe experiment settings."""
        positive_float_fields = {
            "center_frequency_hz": self.center_frequency_hz,
            "nominal_bandwidth_hz": self.nominal_bandwidth_hz,
            "subcarrier_spacing_hz": self.subcarrier_spacing_hz,
            "csi_interval_s": self.csi_interval_s,
            "master_clock_rate_hz": self.master_clock_rate_hz,
            "startup_lead_time_s": self.startup_lead_time_s,
            "dataset_chunk_duration_s": self.dataset_chunk_duration_s,
        }
        for name, value in positive_float_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        positive_integer_fields = {
            "fft_size": self.fft_size,
            "num_active_subcarriers": self.num_active_subcarriers,
            "num_rus": self.num_rus,
            "antennas_per_ru": self.antennas_per_ru,
            "num_rx_antennas": self.num_rx_antennas,
            "pilot_repetitions": self.pilot_repetitions,
            "cp_length_samples": self.cp_length_samples,
            "zc_length_samples": self.zc_length_samples,
        }
        for name, value in positive_integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.num_active_subcarriers % 2:
            raise ValueError("num_active_subcarriers must be even")
        if self.num_active_subcarriers >= self.fft_size:
            raise ValueError("num_active_subcarriers must be smaller than fft_size")
        if self.cp_length_samples >= self.fft_size:
            raise ValueError("cp_length_samples must be smaller than fft_size")
        if self.guard_length_samples < 0:
            raise ValueError("guard_length_samples must be non-negative")
        if gcd(self.zc_root, self.zc_length_samples) != 1:
            raise ValueError("zc_root and zc_length_samples must be coprime")

        interval_samples_float = self.csi_interval_s * self.sampling_rate_hz
        if not isclose(
            interval_samples_float,
            self.csi_interval_samples,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "csi_interval_s must correspond to an integer number of samples"
            )
        if self.sounding_burst_samples > self.csi_interval_samples:
            raise ValueError("the ZC, guard, and pilots do not fit in 5 ms")

        if len(self.pilot_centered_bins) != self.num_tx_branches:
            raise ValueError(
                "pilot_centered_bins must contain one bin per TX branch"
            )
        if len(set(self.pilot_centered_bins)) != len(self.pilot_centered_bins):
            raise ValueError("pilot_centered_bins must be mutually distinct")

        active_half = self.num_active_subcarriers // 2
        if any(
            bin_index == 0
            or bin_index < -active_half
            or bin_index > active_half
            for bin_index in self.pilot_centered_bins
        ):
            raise ValueError(
                "pilot bins must be non-DC bins inside the active allocation"
            )
        if not 0 <= self.sync_tx_branch < self.num_tx_branches:
            raise ValueError("sync_tx_branch is outside the TX branch range")
        if not 0 < self.digital_power_scale <= 1.0:
            raise ValueError("digital_power_scale must be in (0, 1]")
        if self.csi_timestamp_reference != "mean_pilot_useful_centers":
            raise ValueError(
                "csi_timestamp_reference must be "
                "'mean_pilot_useful_centers'"
            )
        if self.dataset_chunk_csi_samples <= 0:
            raise ValueError(
                "dataset_chunk_duration_s is shorter than one CSI interval"
            )

    def paper_parameters(self) -> dict[str, Any]:
        """Return only values attributed to the paper."""
        return {
            "center_frequency_hz": self.center_frequency_hz,
            "nominal_bandwidth_hz": self.nominal_bandwidth_hz,
            "subcarrier_spacing_hz": self.subcarrier_spacing_hz,
            "fft_size": self.fft_size,
            "csi_interval_s": self.csi_interval_s,
            "num_rus": self.num_rus,
            "antennas_per_ru": self.antennas_per_ru,
            "num_tx_branches": self.num_tx_branches,
            "num_rx_antennas": self.num_rx_antennas,
            "pilot_repetitions": self.pilot_repetitions,
            "pilot_orthogonality": "distinct_subcarriers",
        }

    def implementation_assumptions(self) -> dict[str, Any]:
        """Return values chosen because the paper does not specify them."""
        paper_keys = set(self.paper_parameters())
        values = asdict(self)
        return {
            key: value
            for key, value in values.items()
            if key not in paper_keys
        }

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable experiment metadata dictionary."""
        return {
            "schema_version": 1,
            "paper_parameters": self.paper_parameters(),
            "implementation_assumptions": self.implementation_assumptions(),
            "derived_parameters": {
                "num_tx_branches": self.num_tx_branches,
                "sampling_rate_hz": self.sampling_rate_hz,
                "sample_period_s": self.sample_period_s,
                "useful_symbol_duration_s": self.useful_symbol_duration_s,
                "cp_duration_s": self.cp_duration_s,
                "occupied_bandwidth_hz": self.occupied_bandwidth_hz,
                "csi_interval_samples": self.csi_interval_samples,
                "pilot_symbol_samples": self.pilot_symbol_samples,
                "pilot_block_samples": self.pilot_block_samples,
                "pilot_block_cp_start": self.pilot_block_cp_start,
                "sounding_burst_samples": self.sounding_burst_samples,
                "zero_padding_samples": self.zero_padding_samples,
                "csi_timestamp_offset_samples": (
                    self.csi_timestamp_offset_samples
                ),
                "dataset_chunk_csi_samples": self.dataset_chunk_csi_samples,
                "branch_labels": self.branch_labels,
            },
        }


DEFAULT_SOUNDING_CONFIG = SoundingConfig()


__all__ = ["DEFAULT_SOUNDING_CONFIG", "SoundingConfig"]
