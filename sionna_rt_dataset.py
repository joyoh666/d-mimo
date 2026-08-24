"""Generate Sionna RT CSI episodes compatible with ``csi_collector.py``.

The generated NPZ files use the same core arrays as the hardware collector:

* ``csi``: ``[time, ru, antenna]`` complex CSI at 5 ms intervals
* ``per_pilot_csi``: three noisy pilot observations before complex averaging
* 25 ms predictor tokens and 100 ms scheduler gain views
* identical time indices, validity masks, quality fields, and metadata JSON

Sionna RT additionally provides the noiseless ``csi_ground_truth``, UE
trajectory, velocity, RT update boundaries, and per-RU path counts. By default,
the geometry is ray-traced every 100 ms and ``Paths.cfr()`` evolves the channel
at 5 ms intervals inside each geometry segment using path Doppler. Set
``--geometry-update-ms 5`` for one ray-tracing solve per CSI sample.

Example::

    .venv/bin/python sionna_rt_dataset.py \
        --scene box \
        --episodes 75 \
        --dataset-split train \
        --episode-prefix rt_train \
        --snr-db 20

Controlled antenna/scattering sweep::

    .venv/bin/python sionna_rt_dataset.py \
        --scene box \
        --no-los \
        --tx-patterns iso,dipole,tr38901 \
        --scattering-coefficients 0,0.2,0.5 \
        --scattering-patterns lambertian,directive \
        --xpd-coefficients 0,0.2 \
        --samples-per-src 5000 \
        --output-dir output/csi_rt_sweep

Each Cartesian-product variant is stored in its own directory. All variants
with the same episode index share an identical UE trajectory, RT seed, and
standard-normal receiver-noise realization for controlled comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from config import SoundingConfig
from csi_collector import (
    EpisodeData,
    SCHEMA_VERSION,
    SAMPLES_PER_PREDICTOR_BLOCK,
    SAMPLES_PER_SCHEDULER_SEGMENT,
    make_model_views,
    make_time_indices,
    save_episode,
)


Complex64Array = NDArray[np.complex64]
Float32Array = NDArray[np.float32]

DEFAULT_RU_POSITIONS = (
    (-1.5, -1.0, 1.5),
    (1.5, -1.0, 1.5),
    (0.0, 1.6, 1.5),
)
DEFAULT_UE_BOUNDS = (-0.9, 0.9, -0.6, 0.8)
SUPPORTED_ANTENNA_PATTERNS = ("iso", "dipole", "hw_dipole", "tr38901")
SUPPORTED_SINGLE_POLARIZATIONS = ("V", "H")
SUPPORTED_SCATTERING_PATTERNS = (
    "lambertian",
    "backscattering",
    "directive",
)


@dataclass(frozen=True)
class SyntheticRuntimeConfig:
    """Configuration for one family of Sionna RT episodes."""

    scene_spec: str
    duration_s: float
    episodes: int
    episode_prefix: str
    dataset_split: str
    output_dir: str
    seed: int
    snr_db: float
    geometry_update_ms: float
    ru_positions_m: tuple[tuple[float, float, float], ...]
    ue_start_m: tuple[float, float, float] | None
    ue_bounds_m: tuple[float, float, float, float]
    ue_height_m: float
    speed_mps: float
    turn_std_deg: float
    tx_array_spacing_wavelength: float
    tx_pattern: str
    rx_pattern: str
    tx_polarization: str
    rx_polarization: str
    scattering_coefficient: float
    xpd_coefficient: float
    scattering_pattern: str
    scattering_alpha_r: int
    scattering_alpha_i: int
    scattering_lambda: float
    variant_id: str
    max_depth: int
    max_num_paths_per_src: int
    samples_per_src: int
    los: bool
    specular_reflection: bool
    diffuse_reflection: bool
    refraction: bool
    diffraction: bool
    edge_diffraction: bool
    environment_label: str | None
    notes: str | None
    overwrite: bool
    quiet: bool


@dataclass(frozen=True)
class SionnaModules:
    sionna: Any
    rt: Any
    rt_scene: Any
    mitsuba: Any
    drjit: Any
    antenna_pattern_registry: Any
    polarization_registry: Any
    scattering_pattern_registry: Any


@dataclass(frozen=True)
class RayTracingResult:
    csi_ground_truth: Complex64Array
    rt_update_start_sample: NDArray[np.int32]
    rt_update_sample_count: NDArray[np.int32]
    rt_path_count_per_ru: NDArray[np.int32]
    link_has_path: NDArray[np.bool_]


def parse_vector3(text: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(item.strip()) for item in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected x,y,z numeric values.") from exc
    if len(values) != 3 or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("Expected three finite values: x,y,z.")
    return values


def parse_ru_positions(
    text: str,
) -> tuple[tuple[float, float, float], ...]:
    try:
        positions = tuple(parse_vector3(item) for item in text.split(";"))
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(
            "RU positions must be 'x,y,z;x,y,z;x,y,z'."
        ) from exc
    if len(positions) != 3:
        raise argparse.ArgumentTypeError("Exactly three RU positions are required.")
    return positions


def parse_bounds(text: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(item.strip()) for item in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Bounds must be xmin,xmax,ymin,ymax."
        ) from exc
    if len(values) != 4 or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError(
            "Bounds must contain four finite numbers."
        )
    xmin, xmax, ymin, ymax = values
    if xmin >= xmax or ymin >= ymax:
        raise argparse.ArgumentTypeError("Each lower bound must be below its upper bound.")
    return values


def parse_name_list(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated name.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Sweep values must be unique.")
    return values


def parse_float_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated numeric values."
        ) from exc
    if not values or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("Sweep values must be finite.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Sweep values must be unique.")
    return values


def utc_now_string() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def import_sionna_modules() -> SionnaModules:
    """Import the Sionna 2.x RT stack lazily and report version mismatches."""
    try:
        import drjit
        import mitsuba
        import sionna
        import sionna.rt as rt
        import sionna.rt.scene as rt_scene
        from sionna.rt.antenna_pattern import (
            antenna_pattern_registry,
            polarization_registry,
        )
        from sionna.rt.radio_materials import scattering_pattern_registry
    except ImportError as exc:
        raise RuntimeError(
            "Sionna RT is unavailable. Install this project's dependencies "
            "and run with the same Python environment."
        ) from exc

    version = str(getattr(sionna, "__version__", "0"))
    try:
        major = int(version.split(".", maxsplit=1)[0])
    except ValueError as exc:
        raise RuntimeError(f"Cannot parse Sionna version {version!r}.") from exc
    if major < 2:
        raise RuntimeError(
            f"This generator uses the Sionna RT 2.x API; found {version}. "
            "Use Sionna >=2.0 or adapt PathSolver/Paths.cfr calls."
        )
    return SionnaModules(
        sionna,
        rt,
        rt_scene,
        mitsuba,
        drjit,
        antenna_pattern_registry,
        polarization_registry,
        scattering_pattern_registry,
    )


def resolve_scene_path(
    scene_spec: str,
    modules: SionnaModules,
) -> tuple[str | None, str]:
    """Resolve ``empty``, a bundled scene name, or an XML path."""
    if scene_spec.lower() == "empty":
        return None, "empty"

    bundled = getattr(modules.rt_scene, scene_spec, None)
    if isinstance(bundled, str) and bundled.endswith(".xml"):
        return str(Path(bundled).resolve()), f"bundled:{scene_spec}"

    path = Path(scene_spec).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Scene {scene_spec!r} is neither a bundled Sionna scene nor a file."
        )
    return str(path), "custom"


def sha256_file(path: str | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def float_slug(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def make_variant_id(
    *,
    tx_pattern: str,
    rx_pattern: str,
    tx_polarization: str,
    rx_polarization: str,
    scattering_coefficient: float,
    xpd_coefficient: float,
    scattering_pattern: str,
) -> str:
    scattering_label = (
        "none" if scattering_coefficient == 0.0 else scattering_pattern
    )
    return (
        f"tx-{tx_pattern}-{tx_polarization}_"
        f"rx-{rx_pattern}-{rx_polarization}_"
        f"sc-{scattering_label}-s{float_slug(scattering_coefficient)}-"
        f"xpd{float_slug(xpd_coefficient)}"
    )


def make_scattering_pattern(
    runtime: SyntheticRuntimeConfig,
    modules: SionnaModules,
) -> Any:
    factory = modules.scattering_pattern_registry.get(runtime.scattering_pattern)
    if runtime.scattering_pattern == "lambertian":
        return factory()
    if runtime.scattering_pattern == "directive":
        return factory(alpha_r=runtime.scattering_alpha_r)
    return factory(
        alpha_r=runtime.scattering_alpha_r,
        alpha_i=runtime.scattering_alpha_i,
        lambda_=runtime.scattering_lambda,
    )


def configure_material_scattering(
    scene: Any,
    runtime: SyntheticRuntimeConfig,
    modules: SionnaModules,
) -> tuple[str, ...]:
    """Apply one scattering model to every radio material in the scene."""
    material_names = tuple(scene.radio_materials)
    if runtime.scattering_coefficient > 0.0 and not material_names:
        raise ValueError(
            "A positive scattering coefficient requires a scene with radio "
            "materials; the empty scene has no scattering surfaces."
        )
    for material in scene.radio_materials.values():
        material.scattering_coefficient = runtime.scattering_coefficient
        material.xpd_coefficient = runtime.xpd_coefficient
        material.scattering_pattern = make_scattering_pattern(runtime, modules)
    return material_names


def generate_trajectory(
    *,
    num_samples: int,
    samples_per_rt_update: int,
    sample_interval_s: float,
    bounds: tuple[float, float, float, float],
    height_m: float,
    speed_mps: float,
    turn_std_deg: float,
    rng: np.random.Generator,
    start_position: tuple[float, float, float] | None,
) -> tuple[Float32Array, Float32Array]:
    """Generate a bounded, piecewise-linear random TurtleBot trajectory."""
    if num_samples <= 0 or samples_per_rt_update <= 0:
        raise ValueError("Trajectory sizes must be positive.")
    xmin, xmax, ymin, ymax = bounds
    if start_position is None:
        position = np.array(
            [rng.uniform(xmin, xmax), rng.uniform(ymin, ymax), height_m],
            dtype=np.float64,
        )
    else:
        position = np.asarray(start_position, dtype=np.float64).copy()
        if not (xmin <= position[0] <= xmax and ymin <= position[1] <= ymax):
            raise ValueError("UE start position lies outside --ue-bounds.")
        position[2] = height_m

    positions = np.empty((num_samples, 3), dtype=np.float32)
    velocities = np.empty((num_samples, 3), dtype=np.float32)
    heading = rng.uniform(-np.pi, np.pi)
    turn_std_rad = np.deg2rad(turn_std_deg)

    for start in range(0, num_samples, samples_per_rt_update):
        count = min(samples_per_rt_update, num_samples - start)
        heading += rng.normal(0.0, turn_std_rad)
        velocity = np.array(
            [speed_mps * np.cos(heading), speed_mps * np.sin(heading), 0.0],
            dtype=np.float64,
        )
        horizon_s = count * sample_interval_s
        projected = position + velocity * horizon_s

        if projected[0] < xmin or projected[0] > xmax:
            velocity[0] *= -1.0
        if projected[1] < ymin or projected[1] > ymax:
            velocity[1] *= -1.0
        heading = math.atan2(velocity[1], velocity[0]) if speed_mps else heading

        offsets = np.arange(count, dtype=np.float64)[:, None] * sample_interval_s
        segment_positions = position[None, :] + offsets * velocity[None, :]
        segment_positions[:, 0] = np.clip(segment_positions[:, 0], xmin, xmax)
        segment_positions[:, 1] = np.clip(segment_positions[:, 1], ymin, ymax)
        segment_positions[:, 2] = height_m
        positions[start : start + count] = segment_positions
        velocities[start : start + count] = velocity

        position = position + velocity * horizon_s
        position[0] = np.clip(position[0], xmin, xmax)
        position[1] = np.clip(position[1], ymin, ymax)
        position[2] = height_m

    return positions, velocities


def configure_scene(
    *,
    scene_path: str | None,
    cfg: SoundingConfig,
    runtime: SyntheticRuntimeConfig,
    modules: SionnaModules,
    initial_ue_position: NDArray[np.floating[Any]],
) -> Any:
    """Load a scene and install the 3x2 RU array plus single-antenna UE."""
    scene = modules.rt.load_scene(scene_path)
    scene.frequency = cfg.center_frequency_hz
    scene.bandwidth = cfg.nominal_bandwidth_hz
    configure_material_scattering(scene, runtime, modules)

    for name in tuple(scene.transmitters):
        scene.remove(name)
    for name in tuple(scene.receivers):
        scene.remove(name)

    scene.tx_array = modules.rt.PlanarArray(
        num_rows=1,
        num_cols=2,
        horizontal_spacing=runtime.tx_array_spacing_wavelength,
        pattern=runtime.tx_pattern,
        polarization=runtime.tx_polarization,
    )
    scene.rx_array = modules.rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern=runtime.rx_pattern,
        polarization=runtime.rx_polarization,
    )

    service_center = np.mean(np.asarray(runtime.ru_positions_m), axis=0)
    service_center[2] = runtime.ue_height_m
    for ru_index, position in enumerate(runtime.ru_positions_m):
        transmitter = modules.rt.Transmitter(
            name=f"ru{ru_index + 1}",
            position=position,
        )
        transmitter.look_at = tuple(float(value) for value in service_center)
        scene.add(transmitter)

    receiver = modules.rt.Receiver(
        name="ue",
        position=tuple(float(value) for value in initial_ue_position),
        velocity=(0.0, 0.0, 0.0),
    )
    if np.linalg.norm(service_center - initial_ue_position) > 1e-6:
        receiver.look_at = tuple(float(value) for value in service_center)
    scene.add(receiver)
    return scene


def count_paths_per_ru(paths: Any) -> NDArray[np.int32]:
    """Count valid geometric paths for each RU in a synthetic-array solve."""
    _, delays = paths.cir(normalize_delays=False, out_type="numpy")
    delays = np.asarray(delays)
    if delays.ndim != 3 or delays.shape[:2] != (1, 3):
        raise RuntimeError(f"Unexpected Sionna delay shape: {delays.shape}.")
    return np.sum(delays[0] >= 0.0, axis=-1, dtype=np.int32)


def trace_episode(
    *,
    scene: Any,
    positions: Float32Array,
    velocities: Float32Array,
    cfg: SoundingConfig,
    runtime: SyntheticRuntimeConfig,
    modules: SionnaModules,
    episode_seed: int,
) -> RayTracingResult:
    """Ray-trace one trajectory and sample six assigned pilot subcarriers."""
    num_samples = positions.shape[0]
    samples_per_update = round(
        runtime.geometry_update_ms / (cfg.pilot_period_s * 1e3)
    )
    update_starts = np.arange(0, num_samples, samples_per_update, dtype=np.int32)
    update_counts = np.minimum(
        samples_per_update,
        num_samples - update_starts,
    ).astype(np.int32)

    csi = np.zeros((num_samples, 3, 2), dtype=np.complex64)
    path_counts = np.zeros((update_starts.size, 3), dtype=np.int32)
    link_has_path = np.zeros((num_samples, 3, 2), dtype=bool)
    pilot_offsets_hz = (
        np.asarray(cfg.pilot_bins, dtype=np.float64)
        * cfg.subcarrier_spacing_hz
    )
    solver = modules.rt.PathSolver()
    receiver = scene.get("ue")

    for update_index, (start, count) in enumerate(
        zip(update_starts, update_counts)
    ):
        stop = int(start + count)
        receiver.position = tuple(float(value) for value in positions[start])
        receiver.velocity = tuple(float(value) for value in velocities[start])
        paths = solver(
            scene,
            max_depth=runtime.max_depth,
            max_num_paths_per_src=runtime.max_num_paths_per_src,
            samples_per_src=runtime.samples_per_src,
            synthetic_array=True,
            los=runtime.los,
            specular_reflection=runtime.specular_reflection,
            diffuse_reflection=runtime.diffuse_reflection,
            refraction=runtime.refraction,
            diffraction=runtime.diffraction,
            edge_diffraction=runtime.edge_diffraction,
            seed=episode_seed,
        )
        response = np.asarray(
            paths.cfr(
                frequencies=pilot_offsets_hz,
                sampling_frequency=1.0 / cfg.pilot_period_s,
                num_time_steps=int(count),
                normalize_delays=False,
                normalize=False,
                out_type="numpy",
            ),
            dtype=np.complex64,
        )
        expected_shape = (1, 1, 3, 2, int(count), cfg.num_tx_branches)
        if response.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected Sionna CFR shape {response.shape}; "
                f"expected {expected_shape}."
            )

        for branch in range(cfg.num_tx_branches):
            ru_index, antenna_index = divmod(branch, 2)
            csi[start:stop, ru_index, antenna_index] = response[
                0,
                0,
                ru_index,
                antenna_index,
                :,
                branch,
            ]

        path_counts[update_index] = count_paths_per_ru(paths)
        link_has_path[start:stop] = np.repeat(
            (path_counts[update_index] > 0)[:, None],
            2,
            axis=1,
        )
        if not runtime.quiet:
            print(
                f"  RT update {update_index + 1}/{update_starts.size}: "
                f"samples {int(start)}:{stop}, paths "
                f"{path_counts[update_index].tolist()}"
            )

    return RayTracingResult(
        csi_ground_truth=csi,
        rt_update_start_sample=update_starts,
        rt_update_sample_count=update_counts,
        rt_path_count_per_ru=path_counts,
        link_has_path=link_has_path,
    )


def add_pilot_noise(
    csi_ground_truth: Complex64Array,
    *,
    repetitions: int,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[Complex64Array, Complex64Array, Float32Array]:
    """Create independent LS-domain pilot noise and coherently average it."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    branch_power = np.mean(
        np.abs(csi_ground_truth.astype(np.complex128)) ** 2,
        axis=0,
    )
    if np.isposinf(snr_db):
        noise_variance = np.zeros_like(branch_power, dtype=np.float64)
    else:
        noise_variance = branch_power / (10.0 ** (snr_db / 10.0))

    shape = (
        csi_ground_truth.shape[0],
        repetitions,
        csi_ground_truth.shape[1],
        csi_ground_truth.shape[2],
    )
    noise = (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) * np.sqrt(noise_variance[None, None, :, :] / 2.0)
    per_pilot_csi = (
        csi_ground_truth[:, None, :, :] + noise
    ).astype(np.complex64)
    averaged_csi = np.mean(per_pilot_csi, axis=1, dtype=np.complex64)
    return (
        averaged_csi,
        per_pilot_csi,
        noise_variance.astype(np.float32),
    )


def make_metadata(
    *,
    cfg: SoundingConfig,
    runtime: SyntheticRuntimeConfig,
    modules: SionnaModules,
    scene_path: str | None,
    scene_origin: str,
    scene_hash: str | None,
    episode_id: str,
    trajectory_id: str,
    episode_index: int,
    episode_seed: int,
    output_path: Path,
    num_samples: int,
) -> dict[str, Any]:
    runtime_dict = asdict(runtime)
    metadata_snr_db: float | str
    if np.isposinf(runtime.snr_db):
        metadata_snr_db = "+inf"
        runtime_dict["snr_db"] = metadata_snr_db
    else:
        metadata_snr_db = runtime.snr_db

    branch_map = []
    for branch, pilot_bin in enumerate(cfg.pilot_bins):
        branch_map.append(
            {
                "branch_index": branch,
                "ru_index": branch // 2,
                "ru_label": f"RU{branch // 2 + 1}",
                "antenna_index": branch % 2,
                "antenna_label": f"ANT{branch % 2 + 1}",
                "sionna_transmitter": f"ru{branch // 2 + 1}",
                "sionna_tx_array_element": branch % 2,
                "pilot_centered_bin": pilot_bin,
                "pilot_offset_hz": pilot_bin * cfg.subcarrier_spacing_hz,
                "pilot_frequency_hz": (
                    cfg.center_frequency_hz
                    + pilot_bin * cfg.subcarrier_spacing_hz
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": "synthetic_sionna_rt",
        "dataset_variant_id": runtime.variant_id,
        "created_utc": utc_now_string(),
        "episode_id": episode_id,
        "episode_index": episode_index,
        "episode_seed": episode_seed,
        "output_path": str(output_path.resolve()),
        "source": {
            "generator": "sionna_rt_dataset.py",
            "sionna_version": str(modules.sionna.__version__),
            "mitsuba_version": str(getattr(modules.mitsuba, "__version__", "unknown")),
            "drjit_version": str(getattr(modules.drjit, "__version__", "unknown")),
            "scene_spec": runtime.scene_spec,
            "scene_origin": scene_origin,
            "scene_xml_path": scene_path,
            "scene_xml_sha256": scene_hash,
            "channel_definition": (
                "Sionna RT complex baseband CFR evaluated at each antenna's "
                "assigned pilot subcarrier with absolute path delays preserved."
            ),
            "normalize_delays": False,
            "normalize_channel_energy": False,
            "geometry_time_evolution": (
                "Paths are recomputed at each RT update. Within an update, "
                "Paths.cfr applies Doppler at 5 ms intervals with frozen path "
                "geometry, amplitudes, and delays."
            ),
        },
        "paper_alignment": {
            "num_rus": 3,
            "antennas_per_ru": 2,
            "num_ue_antennas": 1,
            "center_frequency_hz": cfg.center_frequency_hz,
            "nominal_bandwidth_hz": cfg.nominal_bandwidth_hz,
            "subcarrier_spacing_hz": cfg.subcarrier_spacing_hz,
            "nfft": cfg.nfft,
            "csi_interval_ms": cfg.pilot_period_s * 1e3,
            "pilot_repetitions": cfg.pilot_repetitions,
            "predictor_block_samples": SAMPLES_PER_PREDICTOR_BLOCK,
            "predictor_block_ms": 25.0,
            "scheduler_segment_samples": SAMPLES_PER_SCHEDULER_SEGMENT,
            "scheduler_segment_ms": 100.0,
        },
        "array_axes": {
            "csi": ["csi_sample", "ru", "antenna"],
            "csi_ground_truth": ["csi_sample", "ru", "antenna"],
            "per_pilot_csi": [
                "csi_sample",
                "pilot_repetition",
                "ru",
                "antenna",
            ],
            "predictor_tokens": [
                "predictor_block",
                "ru",
                "antenna",
                "gain_5_then_cos_phase_5_then_sin_phase_5",
            ],
            "ru_channel_norm": ["csi_sample", "ru"],
            "scheduler_segment_mean_gain": ["scheduler_segment", "ru"],
            "ue_position_m": ["csi_sample", "xyz"],
            "ue_velocity_mps": ["csi_sample", "xyz"],
            "rt_path_count_per_ru": ["rt_update", "ru"],
            "link_has_path": ["csi_sample", "ru", "antenna"],
        },
        "sounding_config": asdict(cfg),
        "synthetic_runtime_config": runtime_dict,
        "scene": {
            "ru_positions_m": runtime.ru_positions_m,
            "tx_array_shape": [1, 2],
            "tx_array_spacing_wavelength": runtime.tx_array_spacing_wavelength,
            "tx_pattern": runtime.tx_pattern,
            "tx_polarization": runtime.tx_polarization,
            "rx_array_shape": [1, 1],
            "rx_pattern": runtime.rx_pattern,
            "rx_polarization": runtime.rx_polarization,
            "scattering_material_scope": "all scene radio materials",
            "scattering_coefficient": runtime.scattering_coefficient,
            "xpd_coefficient": runtime.xpd_coefficient,
            "scattering_pattern": runtime.scattering_pattern,
            "scattering_pattern_parameters": {
                "alpha_r": runtime.scattering_alpha_r,
                "alpha_i": runtime.scattering_alpha_i,
                "lambda": runtime.scattering_lambda,
            },
        },
        "ray_tracing": {
            "geometry_update_ms": runtime.geometry_update_ms,
            "max_depth": runtime.max_depth,
            "max_num_paths_per_src": runtime.max_num_paths_per_src,
            "samples_per_src": runtime.samples_per_src,
            "synthetic_array": True,
            "los": runtime.los,
            "specular_reflection": runtime.specular_reflection,
            "diffuse_reflection": runtime.diffuse_reflection,
            "refraction": runtime.refraction,
            "diffraction": runtime.diffraction,
            "edge_diffraction": runtime.edge_diffraction,
        },
        "noise": {
            "domain": "complex LS channel coefficient",
            "configured_per_pilot_snr_db": metadata_snr_db,
            "independent_pilot_noise": True,
            "complex_average_repetitions": cfg.pilot_repetitions,
            "averaging_snr_gain_db": 10.0 * np.log10(cfg.pilot_repetitions),
            "csi_is_noisy_average": True,
            "csi_ground_truth_is_noiseless_rt": True,
        },
        "collection": {
            "num_csi_samples": num_samples,
            "episode_duration_s": num_samples * cfg.pilot_period_s,
            "device_timestamps_applicable": False,
            "sync_cfo_fields_applicable": False,
        },
        "quality_fields": {
            "valid": "Finite synthetic channel estimate at this CSI sample.",
            "link_has_path": (
                "Whether Sionna RT found at least one path for the RU during "
                "the containing geometry-update interval."
            ),
            "pilot_snr_db": "Configured per-pilot LS-domain SNR.",
            "sync_start_sample": "Always -1; not applicable to synthetic CSI.",
            "sync_offset_samples": "Always -1; not applicable to synthetic CSI.",
            "sync_metric_peak": "NaN; not applicable to synthetic CSI.",
            "cfo_estimate_hz": "NaN; not applicable to synthetic CSI.",
        },
        "episode_labels": {
            "dataset_split": runtime.dataset_split,
            "trajectory_id": trajectory_id,
            "environment_label": (
                runtime.environment_label or runtime.scene_spec
            ),
            "robot_speed_mps": runtime.speed_mps,
            "notes": runtime.notes,
        },
        "branch_map": branch_map,
    }


def build_episode_data(
    *,
    rt_result: RayTracingResult,
    positions: Float32Array,
    velocities: Float32Array,
    cfg: SoundingConfig,
    snr_db: float,
    rng: np.random.Generator,
    metadata: dict[str, Any],
) -> EpisodeData:
    averaged_csi, per_pilot_csi, noise_variance = add_pilot_noise(
        rt_result.csi_ground_truth,
        repetitions=cfg.pilot_repetitions,
        snr_db=snr_db,
        rng=rng,
    )
    valid = np.all(np.isfinite(averaged_csi), axis=(1, 2))
    indices = make_time_indices(averaged_csi.shape[0])
    model_views = make_model_views(averaged_csi, valid)
    elapsed_time_s = (
        indices["sample_index"].astype(np.float64) * cfg.pilot_period_s
    )
    pilot_snr_db = np.full(
        averaged_csi.shape,
        snr_db,
        dtype=np.float32,
    )

    metadata["collection"]["valid_csi_samples"] = int(np.count_nonzero(valid))
    metadata["collection"]["invalid_csi_samples"] = int(
        valid.size - np.count_nonzero(valid)
    )
    metadata["ray_tracing"]["total_paths_per_ru_over_updates"] = (
        np.sum(rt_result.rt_path_count_per_ru, axis=0).astype(int).tolist()
    )

    return EpisodeData(
        csi=averaged_csi,
        per_pilot_csi=per_pilot_csi,
        csi_gain=np.abs(averaged_csi).astype(np.float32),
        csi_phase_rad=np.angle(averaged_csi).astype(np.float32),
        predictor_tokens=np.asarray(
            model_views["predictor_tokens"], dtype=np.float32
        ),
        predictor_block_valid=np.asarray(
            model_views["predictor_block_valid"], dtype=bool
        ),
        ru_channel_norm=np.asarray(
            model_views["ru_channel_norm"], dtype=np.float32
        ),
        scheduler_segment_mean_gain=np.asarray(
            model_views["scheduler_segment_mean_gain"], dtype=np.float32
        ),
        scheduler_segment_valid=np.asarray(
            model_views["scheduler_segment_valid"], dtype=bool
        ),
        pilot_snr_db=pilot_snr_db,
        elapsed_time_s=elapsed_time_s,
        device_time_s=np.full(averaged_csi.shape[0], np.nan, dtype=np.float64),
        host_time_unix_s=np.full(
            averaged_csi.shape[0], np.nan, dtype=np.float64
        ),
        sample_index=np.asarray(indices["sample_index"], dtype=np.int32),
        predictor_block_index=np.asarray(
            indices["predictor_block_index"], dtype=np.int32
        ),
        sample_in_predictor_block=np.asarray(
            indices["sample_in_predictor_block"], dtype=np.int8
        ),
        scheduler_segment_index=np.asarray(
            indices["scheduler_segment_index"], dtype=np.int32
        ),
        sample_in_scheduler_segment=np.asarray(
            indices["sample_in_scheduler_segment"], dtype=np.int8
        ),
        sync_start_sample=np.full(
            averaged_csi.shape[0], -1, dtype=np.int64
        ),
        sync_offset_samples=np.full(
            averaged_csi.shape[0], -1, dtype=np.int32
        ),
        sync_metric_peak=np.full(
            averaged_csi.shape[0], np.nan, dtype=np.float32
        ),
        cfo_estimate_hz=np.full(
            averaged_csi.shape[0], np.nan, dtype=np.float32
        ),
        valid=valid,
        metadata=metadata,
        extra_arrays={
            "csi_ground_truth": rt_result.csi_ground_truth,
            "csi_ground_truth_gain": np.abs(
                rt_result.csi_ground_truth
            ).astype(np.float32),
            "csi_ground_truth_phase_rad": np.angle(
                rt_result.csi_ground_truth
            ).astype(np.float32),
            "ue_position_m": positions,
            "ue_velocity_mps": velocities,
            "rt_update_start_sample": rt_result.rt_update_start_sample,
            "rt_update_sample_count": rt_result.rt_update_sample_count,
            "rt_path_count_per_ru": rt_result.rt_path_count_per_ru,
            "link_has_path": rt_result.link_has_path,
            "pilot_noise_variance_per_branch": noise_variance,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Sionna RT per-antenna CSI episodes with the same NPZ "
            "schema as csi_collector.py."
        )
    )
    parser.add_argument(
        "--scene",
        default="box",
        help="Bundled Sionna scene name, 'empty', or a Mitsuba XML path.",
    )
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-prefix", default="synthetic_rt")
    parser.add_argument(
        "--dataset-split",
        choices=("train", "validation", "test", "unspecified"),
        default="unspecified",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/csi_rt"))
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--geometry-update-ms", type=float, default=100.0)
    parser.add_argument(
        "--ru-positions",
        type=parse_ru_positions,
        default=DEFAULT_RU_POSITIONS,
        help="Three positions: 'x,y,z;x,y,z;x,y,z'.",
    )
    parser.add_argument("--ue-start", type=parse_vector3, default=None)
    parser.add_argument(
        "--ue-bounds",
        type=parse_bounds,
        default=DEFAULT_UE_BOUNDS,
        help="xmin,xmax,ymin,ymax.",
    )
    parser.add_argument("--ue-height-m", type=float, default=1.0)
    parser.add_argument("--speed-mps", type=float, default=1.0)
    parser.add_argument("--turn-std-deg", type=float, default=20.0)
    parser.add_argument("--tx-array-spacing-wavelength", type=float, default=0.5)
    parser.add_argument(
        "--tx-patterns",
        type=parse_name_list,
        default=("iso",),
        help=(
            "Comma-separated RU antenna patterns to sweep: "
            "iso,dipole,hw_dipole,tr38901."
        ),
    )
    parser.add_argument(
        "--rx-patterns",
        type=parse_name_list,
        default=("iso",),
        help=(
            "Comma-separated UE antenna patterns to sweep: "
            "iso,dipole,hw_dipole,tr38901."
        ),
    )
    parser.add_argument(
        "--tx-polarizations",
        type=parse_name_list,
        default=("V",),
        help="Comma-separated RU polarizations to sweep: V,H.",
    )
    parser.add_argument(
        "--rx-polarizations",
        type=parse_name_list,
        default=("V",),
        help="Comma-separated UE polarizations to sweep: V,H.",
    )
    parser.add_argument(
        "--scattering-coefficients",
        type=parse_float_list,
        default=(0.0,),
        help="Comma-separated material scattering coefficients in [0,1].",
    )
    parser.add_argument(
        "--xpd-coefficients",
        type=parse_float_list,
        default=(0.0,),
        help=(
            "Comma-separated cross-polarization discrimination coefficients "
            "in [0,1]."
        ),
    )
    parser.add_argument(
        "--scattering-patterns",
        type=parse_name_list,
        default=("lambertian",),
        help=(
            "Comma-separated diffuse scattering patterns to sweep: "
            "lambertian,backscattering,directive."
        ),
    )
    parser.add_argument(
        "--scattering-alpha-r",
        type=int,
        default=10,
        help="Directive/backscattering forward-lobe exponent.",
    )
    parser.add_argument(
        "--scattering-alpha-i",
        type=int,
        default=10,
        help="Backscattering incident-direction lobe exponent.",
    )
    parser.add_argument(
        "--scattering-lambda",
        type=float,
        default=0.7,
        help="Backscattering forward/backward lobe mixture in [0,1].",
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-num-paths-per-src", type=int, default=100_000)
    parser.add_argument("--samples-per-src", type=int, default=2_000)
    parser.add_argument(
        "--los",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--specular-reflection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--diffuse-reflection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--refraction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--diffraction",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--edge-diffraction",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--environment-label", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate parameters and scene resolution without ray tracing.",
    )
    return parser


def validate_runtimes(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    cfg: SoundingConfig,
) -> tuple[SyntheticRuntimeConfig, ...]:
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be positive.")
    num_samples = round(args.duration_s / cfg.pilot_period_s)
    if not np.isclose(
        num_samples * cfg.pilot_period_s,
        args.duration_s,
        rtol=0.0,
        atol=1e-12,
    ):
        parser.error("--duration-s must be an integer multiple of 5 ms.")
    if args.episodes <= 0:
        parser.error("--episodes must be positive.")
    if not args.episode_prefix.strip():
        parser.error("--episode-prefix must not be empty.")
    if not (np.isfinite(args.snr_db) or np.isposinf(args.snr_db)):
        parser.error("--snr-db must be finite or +inf.")
    if args.geometry_update_ms <= 0.0:
        parser.error("--geometry-update-ms must be positive.")
    samples_per_update = round(
        args.geometry_update_ms / (cfg.pilot_period_s * 1e3)
    )
    if not np.isclose(
        samples_per_update * cfg.pilot_period_s * 1e3,
        args.geometry_update_ms,
        rtol=0.0,
        atol=1e-9,
    ):
        parser.error("--geometry-update-ms must be a multiple of 5 ms.")
    if args.speed_mps < 0.0:
        parser.error("--speed-mps must be non-negative.")
    max_segment_displacement = args.speed_mps * args.geometry_update_ms / 1e3
    xmin, xmax, ymin, ymax = args.ue_bounds
    if max_segment_displacement > min(xmax - xmin, ymax - ymin):
        parser.error(
            "UE displacement within one RT update exceeds the smallest "
            "trajectory-bound dimension; reduce speed or geometry-update-ms."
        )
    if args.turn_std_deg < 0.0:
        parser.error("--turn-std-deg must be non-negative.")
    if args.tx_array_spacing_wavelength <= 0.0:
        parser.error("--tx-array-spacing-wavelength must be positive.")
    if args.max_depth < 0:
        parser.error("--max-depth must be non-negative.")
    if args.max_num_paths_per_src <= 0 or args.samples_per_src <= 0:
        parser.error("Ray sampling limits must be positive.")
    unknown_tx_patterns = set(args.tx_patterns) - set(SUPPORTED_ANTENNA_PATTERNS)
    unknown_rx_patterns = set(args.rx_patterns) - set(SUPPORTED_ANTENNA_PATTERNS)
    if unknown_tx_patterns or unknown_rx_patterns:
        supported = ",".join(SUPPORTED_ANTENNA_PATTERNS)
        parser.error(
            "Unknown antenna pattern(s): "
            f"{sorted(unknown_tx_patterns | unknown_rx_patterns)}. "
            f"Supported patterns are {supported}."
        )
    unknown_polarizations = (
        set(args.tx_polarizations) | set(args.rx_polarizations)
    ) - set(SUPPORTED_SINGLE_POLARIZATIONS)
    if unknown_polarizations:
        parser.error(
            "Only single-port V or H polarization is supported because the "
            "saved CSI schema is fixed to two RU antenna branches and one UE "
            f"antenna; got {sorted(unknown_polarizations)}."
        )
    unknown_scattering_patterns = set(args.scattering_patterns) - set(
        SUPPORTED_SCATTERING_PATTERNS
    )
    if unknown_scattering_patterns:
        supported = ",".join(SUPPORTED_SCATTERING_PATTERNS)
        parser.error(
            "Unknown scattering pattern(s): "
            f"{sorted(unknown_scattering_patterns)}. "
            f"Supported patterns are {supported}."
        )
    if any(value < 0.0 or value > 1.0 for value in args.scattering_coefficients):
        parser.error("--scattering-coefficients values must lie in [0,1].")
    if any(value < 0.0 or value > 1.0 for value in args.xpd_coefficients):
        parser.error("--xpd-coefficients values must lie in [0,1].")
    if args.scattering_alpha_r < 1 or args.scattering_alpha_i < 1:
        parser.error("Scattering alpha exponents must be positive integers.")
    if not 0.0 <= args.scattering_lambda <= 1.0:
        parser.error("--scattering-lambda must lie in [0,1].")
    scattering_enabled = any(
        coefficient > 0.0 for coefficient in args.scattering_coefficients
    )
    if (
        args.diffuse_reflection or scattering_enabled
    ) and args.samples_per_src < 1_000:
        parser.error(
            "Diffuse reflection requires at least 1,000 --samples-per-src."
        )

    runtimes = []
    antenna_combinations = itertools.product(
        args.tx_patterns,
        args.rx_patterns,
        args.tx_polarizations,
        args.rx_polarizations,
    )
    for tx_pattern, rx_pattern, tx_polarization, rx_polarization in (
        antenna_combinations
    ):
        for scattering_coefficient in args.scattering_coefficients:
            # XPD and the angular scattering pattern have no channel effect
            # when the scattering coefficient is zero. Collapse those cases
            # so the sweep does not write duplicate datasets.
            if scattering_coefficient == 0.0:
                scattering_cases = ((0.0, args.scattering_patterns[0]),)
            else:
                scattering_cases = itertools.product(
                    args.xpd_coefficients,
                    args.scattering_patterns,
                )
            for xpd_coefficient, scattering_pattern in scattering_cases:
                variant_id = make_variant_id(
                    tx_pattern=tx_pattern,
                    rx_pattern=rx_pattern,
                    tx_polarization=tx_polarization,
                    rx_polarization=rx_polarization,
                    scattering_coefficient=scattering_coefficient,
                    xpd_coefficient=xpd_coefficient,
                    scattering_pattern=scattering_pattern,
                )
                runtimes.append(
                    SyntheticRuntimeConfig(
                        scene_spec=args.scene,
                        duration_s=args.duration_s,
                        episodes=args.episodes,
                        episode_prefix=args.episode_prefix.strip(),
                        dataset_split=args.dataset_split,
                        output_dir=str(args.output_dir.expanduser().resolve()),
                        seed=args.seed,
                        snr_db=args.snr_db,
                        geometry_update_ms=args.geometry_update_ms,
                        ru_positions_m=args.ru_positions,
                        ue_start_m=args.ue_start,
                        ue_bounds_m=args.ue_bounds,
                        ue_height_m=args.ue_height_m,
                        speed_mps=args.speed_mps,
                        turn_std_deg=args.turn_std_deg,
                        tx_array_spacing_wavelength=(
                            args.tx_array_spacing_wavelength
                        ),
                        tx_pattern=tx_pattern,
                        rx_pattern=rx_pattern,
                        tx_polarization=tx_polarization,
                        rx_polarization=rx_polarization,
                        scattering_coefficient=scattering_coefficient,
                        xpd_coefficient=xpd_coefficient,
                        scattering_pattern=scattering_pattern,
                        scattering_alpha_r=args.scattering_alpha_r,
                        scattering_alpha_i=args.scattering_alpha_i,
                        scattering_lambda=args.scattering_lambda,
                        variant_id=variant_id,
                        max_depth=args.max_depth,
                        max_num_paths_per_src=args.max_num_paths_per_src,
                        samples_per_src=args.samples_per_src,
                        los=args.los,
                        specular_reflection=args.specular_reflection,
                        diffuse_reflection=(
                            args.diffuse_reflection
                            or scattering_coefficient > 0.0
                        ),
                        refraction=args.refraction,
                        diffraction=args.diffraction,
                        edge_diffraction=args.edge_diffraction,
                        environment_label=(
                            (args.environment_label or "").strip() or None
                        ),
                        notes=(args.notes or "").strip() or None,
                        overwrite=args.overwrite,
                        quiet=args.quiet,
                    )
                )

    variant_ids = [runtime.variant_id for runtime in runtimes]
    if len(set(variant_ids)) != len(variant_ids):
        parser.error(
            "The requested sweep produced duplicate variant identifiers. "
            "Remove numerically equivalent sweep values."
        )
    return tuple(runtimes)


def print_plan(
    *,
    cfg: SoundingConfig,
    runtime: SyntheticRuntimeConfig,
    modules: SionnaModules,
    scene_path: str | None,
    scene_origin: str,
) -> None:
    num_samples = round(runtime.duration_s / cfg.pilot_period_s)
    updates = math.ceil(
        num_samples
        / round(runtime.geometry_update_ms / (cfg.pilot_period_s * 1e3))
    )
    print("=== Sionna RT D-MIMO dataset plan ===")
    print(f"Sionna version      : {modules.sionna.__version__}")
    print(f"Scene               : {runtime.scene_spec} ({scene_origin})")
    print(f"Scene XML           : {scene_path or '(empty scene)'}")
    print(f"Episodes            : {runtime.episodes}")
    print(f"Episode duration    : {runtime.duration_s:.3f} s")
    print(f"CSI tensor/episode  : ({num_samples}, 3, 2) complex64")
    print(f"Predictor blocks    : {num_samples // 5} x 25 ms")
    print(f"Scheduler segments  : {num_samples // 20} x 100 ms")
    print(f"RT updates/episode  : {updates} every {runtime.geometry_update_ms:g} ms")
    print(f"Pilot SNR           : {runtime.snr_db:g} dB")
    print(f"UE speed            : {runtime.speed_mps:g} m/s")
    print(f"Output directory    : {runtime.output_dir}")
    print(f"RU positions        : {runtime.ru_positions_m}")
    print(f"Pilot bins          : {cfg.pilot_bins}")


def print_variants(runtimes: Sequence[SyntheticRuntimeConfig]) -> None:
    print(f"Dataset variants    : {len(runtimes)}")
    for index, runtime in enumerate(runtimes, start=1):
        print(
            f"  [{index:02d}] {runtime.variant_id} | "
            f"TX={runtime.tx_pattern}/{runtime.tx_polarization}, "
            f"RX={runtime.rx_pattern}/{runtime.rx_polarization}, "
            f"scattering={runtime.scattering_coefficient:g}/"
            f"{runtime.scattering_pattern}, XPD={runtime.xpd_coefficient:g}, "
            f"diffuse={runtime.diffuse_reflection}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = SoundingConfig()
    cfg.validate()
    runtimes = validate_runtimes(parser, args, cfg)
    reference_runtime = runtimes[0]
    modules = import_sionna_modules()
    scene_path, scene_origin = resolve_scene_path(
        reference_runtime.scene_spec,
        modules,
    )
    scene_hash = sha256_file(scene_path)
    print_plan(
        cfg=cfg,
        runtime=reference_runtime,
        modules=modules,
        scene_path=scene_path,
        scene_origin=scene_origin,
    )
    print_variants(runtimes)
    if args.dry_run:
        print("Dry run complete. No ray tracing or dataset writes were performed.")
        return 0

    output_dir = Path(reference_runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = round(reference_runtime.duration_s / cfg.pilot_period_s)
    samples_per_update = round(
        reference_runtime.geometry_update_ms / (cfg.pilot_period_s * 1e3)
    )

    for episode_index in range(reference_runtime.episodes):
        episode_seed = reference_runtime.seed + episode_index
        trajectory_id = (
            f"{reference_runtime.episode_prefix}_trajectory_"
            f"{episode_index:04d}_seed{episode_seed}"
        )
        trajectory_rng = np.random.default_rng(episode_seed)
        positions, velocities = generate_trajectory(
            num_samples=num_samples,
            samples_per_rt_update=samples_per_update,
            sample_interval_s=cfg.pilot_period_s,
            bounds=reference_runtime.ue_bounds_m,
            height_m=reference_runtime.ue_height_m,
            speed_mps=reference_runtime.speed_mps,
            turn_std_deg=reference_runtime.turn_std_deg,
            rng=trajectory_rng,
            start_position=reference_runtime.ue_start_m,
        )
        for variant_index, runtime in enumerate(runtimes, start=1):
            variant_dir = output_dir / runtime.variant_id
            variant_dir.mkdir(parents=True, exist_ok=True)
            episode_id = (
                f"{runtime.episode_prefix}_{runtime.variant_id}_"
                f"{episode_index:04d}_seed{episode_seed}"
            )
            file_name = (
                f"{runtime.episode_prefix}_{episode_index:04d}_"
                f"seed{episode_seed}.npz"
            )
            output_path = variant_dir / file_name
            if output_path.exists() and not runtime.overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite {output_path}. "
                    "Use --overwrite if intended."
                )
            if not runtime.quiet:
                print(
                    f"Generating episode {episode_index + 1}/"
                    f"{runtime.episodes}, variant {variant_index}/"
                    f"{len(runtimes)}: {runtime.variant_id}"
                )

            scene = configure_scene(
                scene_path=scene_path,
                cfg=cfg,
                runtime=runtime,
                modules=modules,
                initial_ue_position=positions[0],
            )
            rt_result = trace_episode(
                scene=scene,
                positions=positions,
                velocities=velocities,
                cfg=cfg,
                runtime=runtime,
                modules=modules,
                episode_seed=episode_seed,
            )
            metadata = make_metadata(
                cfg=cfg,
                runtime=runtime,
                modules=modules,
                scene_path=scene_path,
                scene_origin=scene_origin,
                scene_hash=scene_hash,
                episode_id=episode_id,
                trajectory_id=trajectory_id,
                episode_index=episode_index,
                episode_seed=episode_seed,
                output_path=output_path,
                num_samples=num_samples,
            )
            # Resetting the noise RNG per variant gives every channel variant
            # the same standard-normal realization, which makes controlled
            # comparisons less noisy while retaining power-scaled LS noise.
            noise_rng = np.random.default_rng(episode_seed + 1_000_000)
            episode = build_episode_data(
                rt_result=rt_result,
                positions=positions,
                velocities=velocities,
                cfg=cfg,
                snr_db=runtime.snr_db,
                rng=noise_rng,
                metadata=metadata,
            )
            save_episode(output_path, episode)
            print(
                f"Saved {output_path.resolve()} "
                f"({np.count_nonzero(episode.valid)}/"
                f"{episode.valid.size} valid CSI samples)"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Sionna RT generation interrupted.", file=sys.stderr)
        raise SystemExit(130)
