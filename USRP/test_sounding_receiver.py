"""Offline tests for ZC synchronization, LS CSI, timestamps, and storage."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from USRP.main import CSIChunkWriter, ClockAnchor
from USRP.sounding_config import DEFAULT_SOUNDING_CONFIG
from USRP.sounding_receiver import (
    RawCapture,
    ReceiverHardwareConfig,
    ReceiverProcessingConfig,
    process_received_samples,
)
from USRP.sounding_transmitter import TransmitterHardwareConfig
from USRP.sounding_transmitter import build_sounding_period


class SoundingReceiverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DEFAULT_SOUNDING_CONFIG
        self.true_csi = np.asarray(
            [
                0.8 + 0.2j,
                0.6 - 0.1j,
                -0.3 + 0.7j,
                0.4 + 0.5j,
                -0.5 - 0.2j,
                0.2 - 0.6j,
            ],
            dtype=np.complex64,
        )
        transmitted = build_sounding_period(self.config)
        self.received_period = np.sum(
            self.true_csi[:, None] * transmitted,
            axis=0,
        ).astype(np.complex64)
        self.prefix_samples = 317
        self.first_sample_time_s = 100.0

    def make_capture(
        self,
        *,
        periods: int = 5,
        cfo_hz: float = 0.0,
    ) -> np.ndarray:
        samples = np.concatenate(
            (
                np.zeros(self.prefix_samples, dtype=np.complex64),
                np.tile(self.received_period, periods),
                np.zeros(
                    self.config.sounding_burst_samples,
                    dtype=np.complex64,
                ),
            )
        )
        sample_index = np.arange(samples.size, dtype=np.float64)
        samples *= np.exp(
            1j
            * 2.0
            * np.pi
            * cfo_hz
            * sample_index
            / self.config.sampling_rate_hz
        ).astype(np.complex64)
        return samples

    def test_zc_sync_ls_csi_and_timestamps(self) -> None:
        result = process_received_samples(
            self.make_capture(),
            self.first_sample_time_s,
            self.config,
            ReceiverProcessingConfig(compensate_frequency_offset=False),
        )
        np.testing.assert_array_equal(
            result.frame_start_sample_indices,
            self.prefix_samples
            + self.config.csi_interval_samples * np.arange(5),
        )
        np.testing.assert_allclose(
            result.csi,
            np.broadcast_to(self.true_csi, result.csi.shape),
            atol=2e-6,
        )
        np.testing.assert_allclose(
            np.diff(result.csi_timestamps_usrp_s),
            self.config.csi_interval_s,
            atol=1e-12,
        )
        first_expected_time = self.first_sample_time_s + (
            self.prefix_samples + self.config.csi_timestamp_offset_samples
        ) / self.config.sampling_rate_hz
        self.assertAlmostEqual(
            result.csi_timestamps_usrp_s[0],
            first_expected_time,
            places=12,
        )

    def test_capture_wide_cfo_is_estimated_and_compensated(self) -> None:
        result = process_received_samples(
            self.make_capture(cfo_hz=350.0),
            self.first_sample_time_s,
            self.config,
            ReceiverProcessingConfig(compensate_frequency_offset=True),
        )
        self.assertAlmostEqual(result.estimated_cfo_hz, 350.0, delta=0.1)
        np.testing.assert_allclose(
            result.csi,
            np.broadcast_to(self.true_csi, result.csi.shape),
            # A tiny median CP-CFO quantization error accumulates over the
            # synthetic 25 ms capture; it remains well below one percent.
            atol=2e-3,
        )

    def test_dataset_contains_csi_time_and_metadata(self) -> None:
        capture_samples = self.make_capture(periods=2)
        result = process_received_samples(
            capture_samples,
            self.first_sample_time_s,
            self.config,
            ReceiverProcessingConfig(compensate_frequency_offset=False),
        )
        raw_capture = RawCapture(
            samples=capture_samples,
            first_sample_time_s=self.first_sample_time_s,
            requested_start_time_s=self.first_sample_time_s,
        )
        anchor = ClockAnchor(
            usrp_time_s=99.0,
            host_unix_time_s=1_700_000_000.0,
            host_bracket_s=1e-4,
        )
        hardware = ReceiverHardwareConfig(address="192.0.2.1")
        processing = ReceiverProcessingConfig(
            compensate_frequency_offset=False
        )

        with tempfile.TemporaryDirectory() as directory:
            writer = CSIChunkWriter(
                directory,
                self.config,
                TransmitterHardwareConfig(),
                hardware,
                processing,
                anchor,
            )
            npz_path = writer.write_chunk(
                result,
                raw_capture,
                processing_buffer_first_time_s=self.first_sample_time_s,
                snapshot_metadata={
                    "turtlebot_linear_velocity_mps": np.linspace(
                        0.1,
                        0.2,
                        result.num_snapshots,
                        dtype=np.float32,
                    )
                },
            )
            json_path = writer.run_directory / "run_metadata.json"
            self.assertTrue(npz_path.exists())
            self.assertTrue(json_path.exists())
            with np.load(npz_path) as saved:
                np.testing.assert_array_equal(saved["csi"], result.csi)
                np.testing.assert_array_equal(
                    saved["csi_timestamps_usrp_s"],
                    result.csi_timestamps_usrp_s,
                )
                np.testing.assert_allclose(
                    saved["csi_timestamps_host_unix_s"],
                    anchor.host_unix_time_s
                    + result.csi_timestamps_usrp_s
                    - anchor.usrp_time_s,
                )
                np.testing.assert_allclose(
                    saved["turtlebot_linear_velocity_mps"],
                    [0.1, 0.2],
                )
                self.assertNotIn("raw_iq", saved.files)
            metadata = json.loads(Path(json_path).read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["timestamp_convention"]["primary_domain"],
                "receiver_usrp_hardware_time",
            )
            self.assertEqual(
                metadata["timestamp_convention"]["mapped_domain"],
                "host_unix_time_approximate",
            )
            manifest = json.loads(
                (writer.run_directory / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["total_snapshots"], 2)


if __name__ == "__main__":
    unittest.main()
