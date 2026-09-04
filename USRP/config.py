"""Shared configuration for the OFDM channel-capture pipeline.

The original capture script remains unchanged. New modules should share one
OFDMConfig instance instead of redeclaring its many module-level values.
Legacy-style field names are retained to simplify moving the original code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import ClassVar, Iterator, Mapping


BandwidthTable = Mapping[float, tuple[int, int]]

# BW [MHz] -> (number of active subcarriers, FFT size)
BANDWIDTH_OPTIONS: BandwidthTable = MappingProxyType(
    {
        1.4: (72, 128),
        3: (180, 256),
        5: (300, 512),
        10: (600, 1024),
        15: (900, 1536),
        20: (1200, 2048),
        50: (3000, 4096),
        400: (24000, 32768),
    }
)


@dataclass(frozen=True)
class OFDMConfig:
    """Complete immutable configuration for one OFDM capture experiment."""

    bandwidth_options: ClassVar[BandwidthTable] = BANDWIDTH_OPTIONS

    # RF and capture settings
    carrier_frequency: float = 5.35e9
    Tx_gain: int = 30
    Rx_gain: int = 30
    BANDWIDTH: float = 10
    CAPTURE_BANDWIDTHS: tuple[float, ...] = (10,)
    modulation_order: int = 4
    POWER: float = 4
    NUM_CHANNEL_CAPTURES: int = 50
    REFERENCE_SEQUENCE_SEED: int = 2026
    OUTPUT_DIR: str = "saved_channel_estimates"

    # Repeated virtual-pilot settings
    NUM_VIRTUAL_PILOTS: int = 7
    VIRTUAL_PILOT_SUBFRAME: int = 0
    VIRTUAL_PILOT_SLOT: int = 1
    VIRTUAL_PILOT_SYMBOL_START: int = 0
    PHASE_ALIGN_VIRTUAL_PILOTS: bool = True

    # LTE-like OFDM structure
    DELTA_F: float = 180e3
    normal_CP_time: float = 4.7e-6
    first_CP_time: float = 5.2e-6
    N_PSS: int = 62
    num_symbols_per_slot: int = 7
    num_slot_per_subframe: int = 2
    num_subframe_per_frame: int = 10

    # UHD settings previously embedded in main()/receive processing
    USRP_DEVICE_ARGS: str = (
        "addr0=192.168.10.2,second_addr=192.168.11.2,"
        "third_addr=192.168.12.2,fourth_addr=192.168.13.2"
    )
    TX_SUBDEV_SPEC: str = "A:0"
    RX_SUBDEV_SPEC: str = "B:0"
    TX_ANTENNA: str = "TX/RX"
    RX_ANTENNA: str = "TX/RX"
    TX_CHANNELS: tuple[int, ...] = (0,)
    RX_CHANNELS: tuple[int, ...] = (0,)
    WAIT_TIME_S: float = 0.2
    TX_DELAY_TIME_S: float = 1e-4
    RX_TRAILING_TIME_S: float = 1e-3
    RX_DISCARD_SAMPLES: int = 10
    OTW_FORMAT: str = "sc16"

    def __post_init__(self) -> None:
        if self.BANDWIDTH not in self.bandwidth_options:
            raise ValueError(
                f"Unsupported BANDWIDTH={self.BANDWIDTH}; "
                f"choose from {tuple(self.bandwidth_options)}"
            )
        invalid_bandwidths = [
            bandwidth
            for bandwidth in self.CAPTURE_BANDWIDTHS
            if bandwidth not in self.bandwidth_options
        ]
        if invalid_bandwidths:
            raise ValueError(f"Unsupported CAPTURE_BANDWIDTHS: {invalid_bandwidths}")
        if self.NUM_CHANNEL_CAPTURES <= 0:
            raise ValueError("NUM_CHANNEL_CAPTURES must be positive")
        if self.NUM_VIRTUAL_PILOTS <= 0:
            raise ValueError("NUM_VIRTUAL_PILOTS must be positive")
        if self.VIRTUAL_PILOT_SYMBOL_START < 0:
            raise ValueError("VIRTUAL_PILOT_SYMBOL_START must be non-negative")
        if (
            self.VIRTUAL_PILOT_SYMBOL_START + self.NUM_VIRTUAL_PILOTS
            > self.num_symbols_per_slot
        ):
            raise ValueError("Virtual-pilot symbols do not fit in one slot")
        if not 0 <= self.VIRTUAL_PILOT_SUBFRAME < self.num_subframe_per_frame:
            raise ValueError("VIRTUAL_PILOT_SUBFRAME is outside the frame")
        if not 0 <= self.VIRTUAL_PILOT_SLOT < self.num_slot_per_subframe:
            raise ValueError("VIRTUAL_PILOT_SLOT is outside the subframe")
        if self.DELTA_F <= 0 or self.normal_CP_time <= 0 or self.first_CP_time <= 0:
            raise ValueError("OFDM frequency/time values must be positive")
        if self.N_PSS <= 0 or self.N_PSS >= self.N:
            raise ValueError("N_PSS must be positive and smaller than N")
        if self.modulation_order < 2 or (
            self.modulation_order & (self.modulation_order - 1)
        ):
            raise ValueError("modulation_order must be a power of two")

    @property
    def N(self) -> int:
        """Number of active subcarriers for BANDWIDTH."""
        return self.bandwidth_options[self.BANDWIDTH][0]

    @property
    def FFT_SIZE(self) -> int:
        return self.bandwidth_options[self.BANDWIDTH][1]

    @property
    def sampling_rate(self) -> float:
        return self.DELTA_F * self.FFT_SIZE

    @property
    def T(self) -> float:
        return 1.0 / self.DELTA_F

    @property
    def normal_CP_length(self) -> int:
        return round(self.normal_CP_time * self.sampling_rate)

    @property
    def first_CP_length(self) -> int:
        return round(self.first_CP_time * self.sampling_rate)

    @property
    def slot_length(self) -> int:
        return (
            self.FFT_SIZE * self.num_symbols_per_slot
            + self.normal_CP_length * (self.num_symbols_per_slot - 1)
            + self.first_CP_length
        )

    @property
    def frame_length(self) -> int:
        return (
            self.slot_length
            * self.num_slot_per_subframe
            * self.num_subframe_per_frame
        )

    @property
    def num_symbols_frame(self) -> int:
        return (
            self.num_symbols_per_slot
            * self.num_slot_per_subframe
            * self.num_subframe_per_frame
        )

    @property
    def VIRTUAL_PILOT_SYMBOLS(self) -> tuple[int, ...]:
        start = self.VIRTUAL_PILOT_SYMBOL_START
        return tuple(range(start, start + self.NUM_VIRTUAL_PILOTS))

    def get_virtual_pilot_positions(self) -> list[tuple[int, int, int]]:
        return [
            (self.VIRTUAL_PILOT_SUBFRAME, self.VIRTUAL_PILOT_SLOT, symbol)
            for symbol in self.VIRTUAL_PILOT_SYMBOLS
        ]

    def effective_reference_sequence_seed(self) -> int:
        """Reproduce the bandwidth-dependent seed used in original main()."""
        return self.REFERENCE_SEQUENCE_SEED + int(10 * self.BANDWIDTH)

    def with_bandwidth(self, bandwidth_mhz: float) -> "OFDMConfig":
        """Create a configuration for another bandwidth without mutation."""
        return replace(self, BANDWIDTH=bandwidth_mhz)

    def get_system_params(self) -> dict[str, object]:
        """Return the original script's parameter dictionary and key names."""
        return {
            "bandwidth_mhz": self.BANDWIDTH,
            "N": self.N,
            "FFT_SIZE": self.FFT_SIZE,
            "delta_f": self.DELTA_F,
            "sampling_rate": self.sampling_rate,
            "T": self.T,
            "normal_CP_length": self.normal_CP_length,
            "first_CP_length": self.first_CP_length,
            "slot_length": self.slot_length,
            "frame_length": self.frame_length,
            "carrier_frequency": self.carrier_frequency,
            "Tx_gain": self.Tx_gain,
            "Rx_gain": self.Rx_gain,
            "modulation_order": self.modulation_order,
            "POWER": self.POWER,
            "num_symbols_per_slot": self.num_symbols_per_slot,
            "num_slot_per_subframe": self.num_slot_per_subframe,
            "num_subframe_per_frame": self.num_subframe_per_frame,
            "num_symbols_frame": self.num_symbols_frame,
            "N_PSS": self.N_PSS,
            "num_virtual_pilots": self.NUM_VIRTUAL_PILOTS,
            "virtual_pilot_positions": self.get_virtual_pilot_positions(),
        }


DEFAULT_CONFIG = OFDMConfig()


def iter_capture_configs(config: OFDMConfig = DEFAULT_CONFIG) -> Iterator[OFDMConfig]:
    """Yield one immutable configuration per requested capture bandwidth."""
    for bandwidth_mhz in config.CAPTURE_BANDWIDTHS:
        yield config.with_bandwidth(bandwidth_mhz)
