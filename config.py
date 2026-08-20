from __future__ import annotations

from dataclasses import dataclass
from math import gcd


@dataclass(frozen=True)
class SoundingConfig:
    """Parameters for the D-MIMO prototype sounding waveform.

    The carrier, bandwidth, SCS, FFT size, 5 ms period, six branches, and
    three-symbol CSI averaging are stated in the manuscript. CP length, ZC
    design, pilot-bin locations, and channel impairments are simulation
    assumptions because the manuscript does not give their exact values.
    """

    center_frequency_hz: float = 2.2e9
    nominal_bandwidth_hz: float = 1.4e6
    subcarrier_spacing_hz: float = 15e3
    nfft: int = 128

    # 5 ms CSI sampling period: 1.92 MS/s * 5 ms = 9,600 samples.
    pilot_period_s: float = 5e-3

    # The paper averages three consecutive pilot symbols per CSI estimate. (동일한 파일럿 3번 전송해서 평균을 취하면 잡음 전력이 1/3로 줄어듦)
    pilot_repetitions: int = 3

    # Simulation assumptions not specified in the manuscript.
    # This CP is longer than the default simulated channel delay spread.
    cp_len: int = 10

    zc_root: int = 25
    zc_len: int = 139
    guard_len: int = 16

    # Six branches use mutually exclusive centered FFT bins.
    # Bin 0 is DC and is intentionally unused.
    pilot_bins: tuple[int, ...] = (-30, -18, -6, 6, 18, 30)
    sync_tx_branch: int = 0

    # Simulation-only channel parameters.
    timing_offset_samples: int = 250
    cfo_hz: float = 120.0
    snr_db: float = 30.0
    random_seed: int = 7

    @property
    def sample_rate_hz(self) -> float:
        return self.nfft * self.subcarrier_spacing_hz

    @property
    def sample_period_s(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def useful_symbol_duration_s(self) -> float:
        return self.nfft / self.sample_rate_hz

    @property
    def period_samples(self) -> int:
        return round(self.pilot_period_s * self.sample_rate_hz)

    @property
    def num_tx_branches(self) -> int:
        return len(self.pilot_bins)

    @property
    def pilot_cp_start(self) -> int:
        return self.zc_len + self.guard_len

    @property
    def pilot_data_start(self) -> int:
        return self.pilot_cp_start + self.cp_len

    @property
    def pilot_symbol_samples(self) -> int:
        return self.cp_len + self.nfft

    @property
    def pilot_block_samples(self) -> int:
        return self.pilot_repetitions * self.pilot_symbol_samples

    @property
    def burst_samples(self) -> int:
        return self.pilot_cp_start + self.pilot_block_samples

    def pilot_symbol_cp_start(self, repetition: int) -> int:
        if not 0 <= repetition < self.pilot_repetitions:
            raise IndexError("Pilot repetition is outside the valid range.")
        return self.pilot_cp_start + repetition * self.pilot_symbol_samples

    def pilot_symbol_data_start(self, repetition: int) -> int:
        return self.pilot_symbol_cp_start(repetition) + self.cp_len

    def validate(self) -> None:
        if self.period_samples < self.burst_samples:
            raise ValueError("The ZC and repeated OFDM pilots do not fit.")
        if self.pilot_repetitions < 1:
            raise ValueError("pilot_repetitions must be positive.")
        if not 0 < self.cp_len < self.nfft:
            raise ValueError("cp_len must be between zero and nfft.")
        if gcd(self.zc_root, self.zc_len) != 1:
            raise ValueError("zc_root and zc_len must be coprime.")
        if not 0 <= self.sync_tx_branch < self.num_tx_branches:
            raise ValueError("sync_tx_branch is outside the branch range.")
        if len(set(self.pilot_bins)) != len(self.pilot_bins):
            raise ValueError("Pilot bins must be unique.")
        if any(k == 0 or not (-self.nfft // 2 <= k < self.nfft // 2)
               for k in self.pilot_bins):
            raise ValueError("Pilot bins must be valid centered non-DC bins.")

        half_bandwidth = self.nominal_bandwidth_hz / 2.0
        if any(
            abs(k * self.subcarrier_spacing_hz) > half_bandwidth
            for k in self.pilot_bins
        ):
            raise ValueError("Pilot bins exceed the nominal sounding band.")

