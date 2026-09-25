#!/usr/bin/env python3
"""PUNet-manifest PointFilter training with QNet-led EMA-normalized guidance.

Optimization data come from an effective shape-disjoint training split derived
from ``shape_split_seed2026.json``. This variant can move explicitly requested
shapes into Train before any loader is constructed. QNet is frozen, while gradients through QNet to
the predicted coordinates are retained. Quality-guidance ROIs are rebuilt every
epoch with coverage-aware FPS until each shape×sigma parent cloud reaches the
requested unique source-point coverage (95% by default). The quality stream supports
batched ROIs plus multi-worker prefetch so the GPU is not starved by CPU patch
construction. The guided objective is

  L = L_PF + beta * [eta * L_Q/EMA(L_Q) + (1-eta) * L_geo/EMA(L_geo)].

For ``quality_guided_full``, the quality stream is sampled online from complete
Train parent clouds.  Each epoch builds an epoch-dependent FPS schedule for
every Train shape x sigma pair and consumes the ENTIRE schedule exactly once.
QNet guidance is processed one ROI at a time (quality batch size = 1), while
multiple ROIs may contribute sequentially to one optimizer step so the full
spatial schedule can be covered without holding many QNet graphs in memory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Hundreds of tiny 3-D PCA calls are more stable and faster with one BLAS
# thread per process.  Set these before NumPy/scipy/torch are imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import scipy.spatial as sp
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path("/data/zhangzy/zzyy")
CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT))

import train_pointfilter_structure_guidance_v2 as pf_base  # noqa: E402


GAUSSIAN_ROOT = ROOT / "PUNet/Gaussion"
DEFAULT_OUTPUT_ROOT = ROOT / "PointFilter_frozen_quality_stage3"
# Clean-aligned shared patches: {shape}/{anchor}.npz, holding both "points_ref"
# (clean) and the matching noisy "points_ref" under noisy/sigma_X/.
PATCH_ROOT_CLEAN = GAUSSIAN_ROOT / "shared_patches/clean"
DEFAULT_QNET_SUMMARY = pf_base.DEFAULT_QNET.parent / "summary.json"
DIMENSIONS = (
    "cleanliness",
    "structure_fidelity",
    "surface_fidelity",
    "sampling_regularity",
    "overall",
)

# Locked optimization weights.  PLCC remains an evaluation statistic only;
# it does not determine the guidance strength.
LOCKED_GUIDANCE_WEIGHTS = {
    "cleanliness": 1.0 / 6.0,
    "structure_fidelity": 0.0,
    "surface_fidelity": 1.0 / 6.0,
    "sampling_regularity": 1.0 / 6.0,
    "overall": 0.5,
}
LOCKED_ZEROED_HEADS = ("structure_fidelity",)
LOCKED_BETA_MATCHED = 1.0
LOCKED_GAMMA_RANK = 0.25
LOCKED_RANK_MARGIN = 0.02
DEFAULT_PUNET_PF_TRAIN = GAUSSIAN_ROOT / "pointfilter_train_seed2026_train_test_overlap"
# 30-shape fair-step protocol: dino/cup/gargoyle may appear in both Train and Test;\n# Validation stays disjoint. PF sampling uses 7466 patches/prepared-shape so\n# the optimizer-step budget is approximately matched to the former 28-shape\n# setting with 8000 patches/prepared-shape. Learning rate remains 1e-4.\n

class DetachedLossEMA:
    """Detached scalar EMA used only to put two losses on comparable scales."""

    def __init__(self, decay: float = 0.99, eps: float = 1e-8):
        self.decay = float(decay)
        self.eps = float(eps)
        self.value: float | None = None

    def update(self, loss: torch.Tensor) -> float:
        current = float(loss.detach().cpu())
        if not math.isfinite(current):
            raise RuntimeError(f"Cannot update EMA from non-finite loss {current}")
        self.value = current if self.value is None else self.decay * self.value + (1.0 - self.decay) * current
        return max(float(self.value), self.eps)

    def state_dict(self) -> dict[str, float | None]:
        return {"decay": self.decay, "eps": self.eps, "value": self.value}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.eps = float(state["eps"])
        self.value = None if state.get("value") is None else float(state["value"])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_shape_csv(value: str) -> list[str]:
    result, seen = [], set()
    for part in str(value).split(","):
        name = part.strip()
        if name and name not in seen:
            result.append(name); seen.add(name)
    return result


def build_effective_shape_split(args: argparse.Namespace) -> dict[str, Any]:
    """Put requested shapes in BOTH Train and Test, while keeping Validation disjoint.

    This is intentional train/test overlap for seen-shape evaluation.  It is useful
    for measuring performance on selected known shapes, but those overlapping test
    samples are NOT an unseen-generalization test.

    The original split file is never edited; a run-local effective split is written
    under --output-dir.
    """
    original_path = Path(args.shape_split)
    split = json.loads(original_path.read_text(encoding="utf-8"))
    for key in ("train", "val", "test"):
        if key not in split or not isinstance(split[key], list):
            raise ValueError(
                f"Invalid shape split {original_path}: missing list key {key!r}"
            )

    requested = _parse_shape_csv(args.extra_train_shapes)
    all_known = set(split["train"]) | set(split["val"]) | set(split["test"])
    unknown = [shape for shape in requested if shape not in all_known]
    if unknown:
        raise ValueError(
            "Requested Train/Test-overlap shapes are not present in the original "
            f"shape pool: {unknown}"
        )

    requested_set = set(requested)
    before = {
        shape: {
            "in_train": shape in split["train"],
            "in_val": shape in split["val"],
            "in_test": shape in split["test"],
        }
        for shape in requested
    }

    # Requested shapes are placed in BOTH Train and Test. By default they are
    # removed from Validation so model selection remains clean. The explicit
    # leakage-ablation flag instead sets Validation equal to Test.
    effective = {
        "train": list(split["train"]),
        "val": (
            list(split["test"])
            if args.validation_uses_test
            else [shape for shape in split["val"] if shape not in requested_set]
        ),
        "test": list(split["test"]),
    }
    for shape in requested:
        if shape not in effective["train"]:
            effective["train"].append(shape)
        if shape not in effective["test"]:
            effective["test"].append(shape)

    train_set = set(effective["train"])
    val_set = set(effective["val"])
    test_set = set(effective["test"])

    # Normal runs require a clean Validation split. The explicit
    # Test-as-Validation ablation permits exactly Val == Test and only the
    # requested Train/Test shapes may then overlap Train.
    if args.validation_uses_test:
        if val_set != test_set:
            raise RuntimeError("--validation-uses-test requires Validation == Test")
        if train_set & val_set != requested_set:
            raise RuntimeError(
                "Unexpected Train/Validation overlap in leakage ablation: "
                f"{sorted(train_set & val_set)}"
            )
    else:
        if train_set & val_set:
            raise RuntimeError(
                f"Train/Validation overlap is forbidden: {sorted(train_set & val_set)}"
            )
        if val_set & test_set:
            raise RuntimeError(
                f"Validation/Test overlap is forbidden: {sorted(val_set & test_set)}"
            )

    # Train/Test overlap is intentional, but only the explicitly requested shapes
    # are allowed to overlap.
    actual_train_test_overlap = train_set & test_set
    unexpected_overlap = actual_train_test_overlap - requested_set
    missing_requested_overlap = requested_set - actual_train_test_overlap
    if unexpected_overlap:
        raise RuntimeError(
            "Unexpected Train/Test overlap outside the requested shapes: "
            f"{sorted(unexpected_overlap)}"
        )
    if missing_requested_overlap:
        raise RuntimeError(
            "Failed to place requested shapes in both Train and Test: "
            f"{sorted(missing_requested_overlap)}"
        )

    effective_union = train_set | val_set | test_set
    if args.validation_uses_test:
        expected_union = set(effective["train"]) | set(effective["test"])
        if effective_union != expected_union:
            raise RuntimeError("Unexpected effective shape pool in Test-as-Validation ablation")
        excluded_original_validation_shapes = sorted(set(split["val"]) - effective_union)
    else:
        if effective_union != all_known:
            raise RuntimeError(
                "Effective split no longer covers the same original shape pool"
            )
        excluded_original_validation_shapes = []

    effective_path = args.output_dir / "effective_shape_split.json"
    write_json(effective_path, effective)

    summary = {
        "original_split": str(original_path),
        "effective_split": str(effective_path),
        "requested_train_test_overlap_shapes": requested,
        "location_before": before,
        "train_count_before": len(split["train"]),
        "train_count_after": len(effective["train"]),
        "val_count_before": len(split["val"]),
        "val_count_after": len(effective["val"]),
        "test_count_before": len(split["test"]),
        "test_count_after": len(effective["test"]),
        "train_test_overlap": sorted(actual_train_test_overlap),
        "train_test_overlap_is_intentional": True,
        "validation_uses_test": bool(args.validation_uses_test),
        "excluded_original_validation_shapes": excluded_original_validation_shapes,
        "evaluation_semantics": (
            "TEST-AS-VALIDATION LEAKAGE ABLATION: Test drives checkpoint selection; "
            "overlap shapes are seen in Train"
            if args.validation_uses_test
            else "overlapping Test shapes are seen-shape evaluation and must not be "
                 "reported as unseen/generalization Test performance"
        ),
        "train_shapes_after": effective["train"],
        "val_shapes_after": effective["val"],
        "test_shapes_after": effective["test"],
    }
    print(
        "SPLIT_OVERRIDE " + json.dumps(summary, ensure_ascii=False),
        flush=True,
    )

    args.original_shape_split = original_path
    args.shape_split = effective_path
    return summary


class SharedQualityPatchDataset(Dataset):
    """Frozen shared noisy/clean patches from Train or Validation only."""

    def __init__(self, manifest_path: Path, split_path: Path, split_name: str):
        if split_name not in {"train", "val"}:
            raise ValueError("Stage III forbids constructing or accessing the Test split")
        split = json.loads(split_path.read_text(encoding="utf-8"))
        allowed_shapes = set(split[split_name])
        rows = pf_base.read_jsonl(manifest_path)
        clean = {
            row["anchor_key"]: row
            for row in rows
            if row["family"] == "clean" and row["shape"] in allowed_shapes
        }
        noisy = [
            row
            for row in rows
            if row["family"] == "noisy" and row["shape"] in allowed_shapes
        ]
        noisy.sort(key=lambda row: (row["shape"], str(row["sigma"]), int(row["anchor_id"])))
        self.rows: list[dict[str, Any]] = []
        self.cache: list[dict[str, torch.Tensor]] = []
        for row in noisy:
            clean_row = clean.get(row["anchor_key"])
            if clean_row is None:
                raise KeyError(f"Missing clean anchor for {row['patch_id']}")
            item = {
                "sample_id": row["patch_id"],
                "shape": row["shape"],
                "sigma": str(row["sigma"]),
                "anchor_id": int(row["anchor_id"]),
                "noisy_path": row["patch_path"],
                "clean_path": clean_row["patch_path"],
            }
            self.rows.append(item)
            self.cache.append(self._load(item))

        expected = len(allowed_shapes) * 5 * 16
        if len(self.rows) != expected:
            raise ValueError(f"Expected {expected} {split_name} noisy patches, got {len(self.rows)}")
        if len({row["sample_id"] for row in self.rows}) != len(self.rows):
            raise ValueError(f"Duplicate {split_name} sample_id")

    @staticmethod
    def _load(row: dict[str, Any]) -> dict[str, torch.Tensor]:
        with np.load(row["noisy_path"]) as data:
            noisy_ref = np.asarray(data["points_ref"], dtype=np.float32).copy()
            frame = np.asarray(data["clean_pca_frame"], dtype=np.float32).copy()
            radius = np.asarray(data["clean_reference_radius"], dtype=np.float32).reshape(1)
            center = np.asarray(data["clean_center_xyz"], dtype=np.float32).copy()
        with np.load(row["clean_path"]) as data:
            clean_ref = np.asarray(data["points_ref"], dtype=np.float32).copy()
        for name, points in (("noisy", noisy_ref), ("clean", clean_ref)):
            if points.shape != (500, 3) or not np.isfinite(points).all():
                raise ValueError(f"Invalid {name} patch {row['sample_id']}: {points.shape}")
        if frame.shape != (3, 3) or radius.shape != (1,) or center.shape != (3,):
            raise ValueError(f"Invalid transform for {row['sample_id']}")
        return {
            "noisy_ref": torch.from_numpy(noisy_ref),
            "clean_ref": torch.from_numpy(clean_ref),
            "frame": torch.from_numpy(frame),
            "radius": torch.from_numpy(radius),
            "center": torch.from_numpy(center),
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        values = self.cache[index]
        return {
            **{key: value.clone() for key, value in values.items()},
            "sample_id": row["sample_id"],
            "shape": row["shape"],
            "sigma": row["sigma"],
            "anchor_id": row["anchor_id"],
        }


class DeterministicEvaluationPatchDataset(pf_base.PointcloudPatchDataset):
    """The official evaluation loader with an explicit, local sampling RNG."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.sampling_rng = np.random.RandomState(int(self.seed))

    def set_sampling_seed(self, seed: int) -> None:
        self.sampling_rng.seed(int(seed) % (2**32))

    def patch_sampling(self, patch_pts: np.ndarray) -> np.ndarray:
        replace = patch_pts.shape[0] <= self.points_per_patch
        return self.sampling_rng.choice(
            patch_pts.shape[0], self.points_per_patch, replace=replace
        )

    def __getitem__(self, index: int):
        try:
            item = super().__getitem__(index)
            if item is None:
                return None
            if any(
                torch.is_tensor(value) and not bool(torch.isfinite(value).all())
                for value in item
            ):
                return None
            return item
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return None


class OnlineQualityPatchDataset(Dataset):
    """Online QNet ROIs sampled from complete noisy parent clouds.

    For ``quality_guided_full`` this dataset uses *coverage-aware FPS* rather
    than a fixed number of anchors.  For every shape×sigma parent cloud, each
    epoch repeatedly selects a centre from the still-uncovered region, gathers
    its exact ``points_per_patch`` nearest neighbours, marks those source-point
    indices covered, and continues until the requested unique-point coverage
    ratio is reached.  The selected 500-point neighbourhoods are cached in the
    epoch schedule, so the coverage computed at schedule construction is the
    same neighbourhood later sent through PointFilter/QNet.

    The centre policy is deterministic for (seed, epoch, shape, sigma): the
    first centre is random, then the next centre is the uncovered point farthest
    from all previously selected centres.  This keeps the ROIs spatially spread
    while explicitly prioritising areas that have not yet participated in the
    quality loss.
    """

    def __init__(
        self,
        noisy_root: Path,
        clean_root: Path,
        split_path: Path,
        split_name: str,
        sigmas: tuple[str, ...],
        patch_radius: float,
        points_per_patch: int,
        seed: int,
        work_root: Path,
        samples_per_epoch: int,
        centers_per_cloud: int = 0,
        coverage_target: float = 0.0,
        max_centers_per_cloud: int = 0,
    ):
        if split_name not in {"train", "val"}:
            raise ValueError("Online Stage III forbids constructing or accessing the Test split")
        if points_per_patch < 3:
            raise ValueError("Online quality patches require at least 3 points")
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        if not 0.0 <= float(coverage_target) <= 1.0:
            raise ValueError("coverage_target must be in [0,1]")
        if max_centers_per_cloud < 0:
            raise ValueError("max_centers_per_cloud must be >= 0")

        split = json.loads(split_path.read_text(encoding="utf-8"))
        self.parent_shapes = tuple(str(shape) for shape in split[split_name])
        self.pairs = tuple((shape, sigma) for shape in self.parent_shapes for sigma in sigmas)
        if not self.pairs:
            raise ValueError(f"No online quality sources for split {split_name}")

        self.noisy_root = noisy_root
        self.clean_root = clean_root
        self.split_name = split_name
        self.patch_radius = float(patch_radius)
        self.points_per_patch = int(points_per_patch)
        self.seed = int(seed)
        self.work_root = work_root / split_name
        self.samples_per_epoch = int(samples_per_epoch)

        # For coverage-aware mode this is a *minimum* number of ROIs.  Sampling
        # continues beyond it until coverage_target is reached.
        self.centers_per_cloud = int(centers_per_cloud)
        self.coverage_target = float(coverage_target)
        self.max_centers_per_cloud = int(max_centers_per_cloud)
        self.coverage_aware = self.centers_per_cloud > 0 and self.coverage_target > 0.0

        self.epoch = 0
        self._sources: dict[tuple[str, str], dict[str, Any]] = {}
        # Entries are (pair_index, centre_index, exact 500 source indices or None).
        self._epoch_schedule: list[tuple[int, int, np.ndarray | None]] = []
        self._planned_coverage_by_parent: dict[tuple[str, str], float] = {}
        self._planned_centers_by_parent: dict[tuple[str, str], int] = {}

    @staticmethod
    def _seed32(*values: int) -> int:
        # Stable integer mixing; unlike hash(), this is invariant across processes.
        mixed = 2166136261
        for value in values:
            mixed = ((mixed ^ int(value)) * 16777619) % (2**32)
        return mixed

    def _coverage_schedule_for_parent(
        self,
        pair_index: int,
        shape: str,
        sigma: str,
    ) -> list[tuple[int, int, np.ndarray]]:
        """Build one shape×sigma schedule until unique source coverage >= target."""
        source = self._get_source(shape, sigma)
        cloud = source["cloud"]
        tree: sp.cKDTree = source["tree"]
        n = int(len(cloud))
        if self.points_per_patch > n:
            raise ValueError(
                f"quality patch size {self.points_per_patch} exceeds parent cloud size {n}"
            )

        target_count = int(math.ceil(self.coverage_target * n))
        min_centers = max(1, int(self.centers_per_cloud))
        max_centers = (
            int(self.max_centers_per_cloud)
            if self.max_centers_per_cloud > 0
            else n
        )
        if max_centers < min_centers:
            raise ValueError(
                f"max_centers_per_cloud={max_centers} < minimum centres={min_centers}"
            )

        rng = np.random.RandomState(
            self._seed32(self.seed, self.epoch, pair_index, 4049)
        )
        covered = np.zeros(n, dtype=np.bool_)
        min_squared_distance = np.full(n, np.inf, dtype=np.float32)
        selected_centers: set[int] = set()
        entries: list[tuple[int, int, np.ndarray]] = []

        # Epoch-dependent first point keeps the schedule dynamic while all later
        # choices are deterministic coverage-aware FPS.
        centre = int(rng.randint(0, n))
        while len(entries) < max_centers:
            if centre in selected_centers:
                # This should be impossible because the next centre is selected
                # from uncovered points, but keep a deterministic fallback.
                remaining = np.flatnonzero(~covered)
                if remaining.size == 0:
                    break
                centre = int(remaining[0])

            # The quality ROI itself is exactly the 500 nearest source points.
            # Store these indices now so schedule coverage and training coverage
            # refer to the same intended local region.
            _dist, nn_idx = tree.query(
                cloud[centre], k=self.points_per_patch
            )
            nn_idx = np.asarray(nn_idx, dtype=np.int64).reshape(-1)
            if nn_idx.size != self.points_per_patch:
                raise RuntimeError(
                    f"Expected {self.points_per_patch} kNN points, got {nn_idx.size} "
                    f"for {shape} sigma={sigma} centre={centre}"
                )
            if np.unique(nn_idx).size != nn_idx.size:
                raise RuntimeError(
                    f"Duplicate kNN indices for {shape} sigma={sigma} centre={centre}"
                )

            entries.append((pair_index, centre, nn_idx.astype(np.int32, copy=True)))
            selected_centers.add(centre)
            covered[nn_idx] = True

            # Update the FPS distance field with the newly selected centre.
            delta = cloud - cloud[centre]
            squared = np.einsum("ij,ij->i", delta, delta, optimize=True).astype(
                np.float32, copy=False
            )
            np.minimum(min_squared_distance, squared, out=min_squared_distance)

            covered_count = int(covered.sum())
            if len(entries) >= min_centers and covered_count >= target_count:
                break

            # Coverage-aware FPS: choose the farthest point among source points
            # that have never appeared in any selected QNet ROI.
            uncovered = np.flatnonzero(~covered)
            if uncovered.size == 0:
                break
            centre = int(uncovered[np.argmax(min_squared_distance[uncovered])])

        coverage = float(covered.mean())
        key = (shape, sigma)
        self._planned_coverage_by_parent[key] = coverage
        self._planned_centers_by_parent[key] = len(entries)

        if coverage + 1e-12 < self.coverage_target:
            raise RuntimeError(
                "Coverage-aware ROI construction failed to hit the requested target: "
                f"{shape} sigma={sigma} coverage={coverage:.6f} "
                f"target={self.coverage_target:.6f} centres={len(entries)} "
                f"max_centres={max_centers}. Increase --quality-max-centers-per-cloud."
            )
        return entries

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.centers_per_cloud <= 0:
            return

        schedule: list[tuple[int, int, np.ndarray | None]] = []
        self._planned_coverage_by_parent = {}
        self._planned_centers_by_parent = {}

        for pair_index, (shape, sigma) in enumerate(self.pairs):
            if self.coverage_aware:
                pair_entries = self._coverage_schedule_for_parent(
                    pair_index, shape, sigma
                )
                schedule.extend(pair_entries)
            else:
                # Legacy fixed-count FPS path retained for non-coverage experiments.
                cloud = self._get_source(shape, sigma)["cloud"]
                rng = np.random.RandomState(
                    self._seed32(self.seed, self.epoch, pair_index, 4049)
                )
                first = int(rng.randint(0, len(cloud)))
                selected = np.empty(self.centers_per_cloud, dtype=np.int64)
                minimum = np.full(len(cloud), np.inf, dtype=np.float64)
                farthest = first
                for center_i in range(self.centers_per_cloud):
                    selected[center_i] = farthest
                    squared = np.sum((cloud - cloud[farthest]) ** 2, axis=1)
                    minimum = np.minimum(minimum, squared)
                    farthest = int(np.argmax(minimum))
                schedule.extend(
                    (pair_index, int(center), None) for center in selected
                )

        if not schedule:
            raise RuntimeError("Online quality epoch schedule is empty")
        order = np.random.RandomState(
            self._seed32(self.seed, self.epoch, 65537)
        ).permutation(len(schedule))
        self._epoch_schedule = [schedule[int(i)] for i in order]

        if self.coverage_aware:
            planned_values = list(self._planned_coverage_by_parent.values())
            centre_values = list(self._planned_centers_by_parent.values())
            print(
                "COVERAGE_SCHEDULE " + json.dumps({
                    "epoch": self.epoch,
                    "target": self.coverage_target,
                    "parents": len(self.pairs),
                    "rois": len(self._epoch_schedule),
                    "centers_per_parent_min": int(min(centre_values)),
                    "centers_per_parent_mean": float(np.mean(centre_values)),
                    "centers_per_parent_max": int(max(centre_values)),
                    "planned_coverage_mean": float(np.mean(planned_values)),
                    "planned_coverage_min": float(np.min(planned_values)),
                }, ensure_ascii=False),
                flush=True,
            )

    def _get_source(self, shape: str, sigma: str) -> dict[str, Any]:
        key = (shape, sigma)
        cached = self._sources.get(key)
        if cached is not None:
            return cached

        xyz_path = self.noisy_root / f"sigma_{sigma}" / f"{shape}.xyz"
        points = np.loadtxt(xyz_path, dtype=np.float32)
        points = np.asarray(points[:, :3], dtype=np.float32)
        if points.shape != (50000, 3) or not np.isfinite(points).all():
            raise ValueError(f"Invalid online noisy cloud {xyz_path}: {points.shape}")

        source_dir = self.work_root / f"{shape}__sigma_{sigma}"
        source_dir.mkdir(parents=True, exist_ok=True)
        npy_path = source_dir / "cloud.npy"
        write_npy = True
        if npy_path.exists():
            existing = np.load(npy_path, mmap_mode="r")
            write_npy = existing.shape != points.shape or not np.array_equal(existing, points)
            del existing
        if write_npy:
            np.save(npy_path, points)

        dataset = DeterministicEvaluationPatchDataset(
            root=str(source_dir),
            shape_name="cloud",
            patch_radius=self.patch_radius,
            points_per_patch=self.points_per_patch,
            seed=self.seed,
            train_state="evaluation",
        )
        cloud = dataset.noise_shapes[0]["noise_pts"]
        source = {
            "dataset": dataset,
            "cloud": cloud,
            "tree": dataset.noise_shapes[0]["noise_kdtree"],
            "radius": float(dataset.patch_radius_absolute[0]),
            "npy_path": npy_path,
            "clean": np.loadtxt(self.clean_root / f"{shape}.xyz", dtype=np.float32)[:, :3],
        }
        if source["clean"].shape != cloud.shape or not np.isfinite(source["clean"]).all():
            raise ValueError(f"Invalid/misaligned clean cloud for {shape}: {source['clean'].shape}")
        self._sources[key] = source
        return source

    def __len__(self) -> int:
        if self.centers_per_cloud > 0:
            if self._epoch_schedule:
                return len(self._epoch_schedule)
            # Before the first set_epoch(), expose only a lower-bound length.
            # The real variable-length coverage schedule is built at epoch start.
            return len(self.pairs) * self.centers_per_cloud
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        planned_neighbors: np.ndarray | None = None
        if self.centers_per_cloud > 0:
            if not self._epoch_schedule:
                self.set_epoch(self.epoch)
            pair_index, center_index, planned_neighbors = self._epoch_schedule[int(index)]
        else:
            pair_index = (int(index) + self.epoch) % len(self.pairs)
            center_index = -1

        shape, sigma = self.pairs[pair_index]
        source = self._get_source(shape, sigma)
        cloud = source["cloud"]
        tree: sp.cKDTree = source["tree"]
        dataset: DeterministicEvaluationPatchDataset = source["dataset"]

        item_seed = self._seed32(
            self.seed,
            100 if self.split_name == "train" else 200,
            self.epoch,
            int(index),
        )
        rng = np.random.RandomState(item_seed)
        if center_index < 0:
            center_index = int(rng.randint(0, len(cloud)))

        if planned_neighbors is not None:
            primary_indices = np.asarray(planned_neighbors, dtype=np.int64).reshape(-1)
            # Extra nearest points are only fallbacks for the extremely rare
            # invalid local PCA patch.  The first 500 are exactly the planned ROI.
            extra_k = min(len(cloud), self.points_per_patch * 2)
            _d, fallback_indices = tree.query(cloud[center_index], k=extra_k)
            fallback_indices = np.asarray(fallback_indices, dtype=np.int64).reshape(-1)
            seen = set(int(x) for x in primary_indices.tolist())
            candidates = primary_indices.tolist() + [
                int(x) for x in fallback_indices.tolist() if int(x) not in seen
            ]
            sampling_strategy = "coverage_aware_fps_knn"
        else:
            neighbor_indices = np.asarray(
                tree.query_ball_point(cloud[center_index], source["radius"]), dtype=np.int64
            )
            if neighbor_indices.size < 3:
                raise RuntimeError(
                    f"Online centre has fewer than 3 neighbours: {shape} sigma={sigma} "
                    f"index={center_index}"
                )
            if neighbor_indices.size > self.points_per_patch:
                candidates = rng.permutation(neighbor_indices).tolist()
            else:
                candidates = rng.choice(
                    neighbor_indices, self.points_per_patch * 2, replace=True
                ).tolist()
            sampling_strategy = "epoch_fps" if self.centers_per_cloud > 0 else "random"

        sampling_seed = self._seed32(item_seed, 7919)
        dataset.set_sampling_seed(sampling_seed)
        patches: list[torch.Tensor] = []
        inverse_rotations: list[torch.Tensor] = []
        centers: list[torch.Tensor] = []
        accepted_indices: list[int] = []
        attempts = 0

        while len(patches) < self.points_per_patch:
            if attempts >= self.points_per_patch * 10:
                raise RuntimeError(
                    f"Could not collect {self.points_per_patch} valid online patches from "
                    f"{shape} sigma={sigma} centre={center_index}"
                )
            if not candidates:
                # Rare fallback only.  Draw from the local radius so the ROI
                # remains local even when PCA rejects several centres.
                local = np.asarray(
                    tree.query_ball_point(cloud[center_index], source["radius"]),
                    dtype=np.int64,
                )
                if local.size == 0:
                    local = np.arange(len(cloud), dtype=np.int64)
                candidates = rng.choice(
                    local, self.points_per_patch, replace=True
                ).tolist()
            neighbor_index = int(candidates.pop(0))
            attempts += 1
            value = dataset[neighbor_index]
            if value is None:
                continue
            patch, inverse_rotation, center = value
            patches.append(patch)
            inverse_rotations.append(inverse_rotation)
            centers.append(torch.as_tensor(center))
            accepted_indices.append(neighbor_index)

        accepted = np.asarray(accepted_indices, dtype=np.int64)
        return {
            "patches": torch.stack(patches),
            "inverse_rotation": torch.stack(inverse_rotations),
            "center": torch.stack(centers),
            "radius": torch.tensor(source["radius"], dtype=torch.float32),
            "shape": shape,
            "sigma": sigma,
            "online_center_index": center_index,
            "neighbor_indices": torch.from_numpy(accepted.copy()),
            "noisy_points": torch.from_numpy(cloud[accepted].copy()),
            "clean_points": torch.from_numpy(source["clean"][accepted].copy()),
            "sampling_seed": int(sampling_seed),
            "invalid_center_patches_skipped": int(attempts - self.points_per_patch),
            "source_npy": str(source["npy_path"]),
            "sampling_strategy": sampling_strategy,
            "centers_per_parent_cloud": self.centers_per_cloud,
            "coverage_target": self.coverage_target,
        }


class RobustPointcloudPatchDataset(pf_base.PointcloudPatchDataset):
    """Preserve the legacy loader while skipping its rare invalid PCA patch."""

    def __getitem__(self, index):
        try:
            item = super().__getitem__(index)
            if item is None:
                return None
            if any(torch.is_tensor(value) and not bool(torch.isfinite(value).all()) for value in item):
                return None
            return item
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return None


def build_pf_loader(args: argparse.Namespace) -> DataLoader:
    dataset = RobustPointcloudPatchDataset(
        root=str(args.train_root),
        shapes_list_file="train.txt",
        patch_radius=args.patch_radius,
        points_per_patch=args.pf_points_per_patch,
        seed=args.seed,
        train_state="train",
    )
    sampler = pf_base.RandomPointcloudPatchSampler(
        dataset,
        patches_per_shape=args.patches_per_shape,
        seed=args.seed,
        identical_epochs=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        sampler=sampler,
        shuffle=False,
        collate_fn=pf_base.my_collate,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=pf_base.seed_worker if args.workers > 0 else None,
        generator=generator,
    )

def build_quality_loader(
    args: argparse.Namespace,
    split_name: str,
    shuffle: bool,
) -> DataLoader:
    dataset = SharedQualityPatchDataset(args.shared_manifest, args.shape_split, split_name)
    generator = torch.Generator().manual_seed(args.seed + (100 if split_name == "train" else 200))
    return DataLoader(
        dataset,
        batch_size=args.quality_batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


class CleanAlignedLocalTargetDataset(Dataset):
    """Clean-aligned patches used as the target space for the quality-guided loss.

    Only the Train split is constructible here (same guard as
    ``SharedQualityPatchDataset``); the Validation split may never be a local
    target, and the Test split is locked.

    One patch supplies everything the loss needs, all in the clean PCA frame
    that ``clean_aligned_attribution_control.py`` uses:

        noisy_ref, clean_ref, frame, radius, center

    The *candidate* is this patch's ``noisy_ref`` denoised through PointFilter,
    and the denoising metric is then computed with ``to_raw`` + ``patch_metrics``
    exactly as the bank does.  That is the protocol whose effect reproduces
    exactly against the 480-patch bank, so it is the space the quality loss
    operates in.
    """

    def __init__(
        self,
        manifest_path: Path,
        split_path: Path,
        split_name: str,
        seed: int,
        samples_per_epoch: int,
    ):
        if split_name != "train":
            raise ValueError(
                "Local quality targets must come from the Train split; "
                f"refusing split_name={split_name!r}"
            )
        self.inner = SharedQualityPatchDataset(manifest_path, split_path, split_name)
        self.seed = int(seed)
        self.epoch = 0
        self.samples_per_epoch = int(samples_per_epoch)
        self.parent_shapes = sorted({row["shape"] for row in self.inner.rows})
        self.indices = np.arange(len(self.inner), dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        """Fresh permutation each epoch; the epoch stride keeps the draw even."""
        self.epoch = int(epoch)
        order = np.random.RandomState(self.seed + int(epoch)).permutation(len(self.inner))
        step = max(1, len(order) // max(self.samples_per_epoch, 1))
        self.indices = order[::step][: self.samples_per_epoch].astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.inner[int(self.indices[int(index) % self.indices.shape[0]])]


def build_clean_aligned_local_loader(
    args: argparse.Namespace, split_name: str, samples_per_epoch: int
) -> DataLoader:
    dataset = CleanAlignedLocalTargetDataset(
        args.shared_manifest, args.shape_split, split_name, args.seed,
        samples_per_epoch=samples_per_epoch,
    )
    dataset.set_epoch(1)
    generator = torch.Generator().manual_seed(args.seed + (700 if split_name == "train" else 800))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def build_online_quality_loader(
    args: argparse.Namespace,
    split_name: str,
    samples_per_epoch: int,
) -> DataLoader:
    sigmas = tuple(part.strip() for part in args.quality_sigmas.split(",") if part.strip())
    dataset = OnlineQualityPatchDataset(
        noisy_root=args.quality_noisy_root,
        clean_root=args.quality_clean_cloud_root,
        split_path=args.shape_split,
        split_name=split_name,
        sigmas=sigmas,
        patch_radius=args.patch_radius,
        points_per_patch=args.quality_points_per_patch,
        seed=args.seed,
        work_root=args.quality_work_root,
        samples_per_epoch=samples_per_epoch,
        centers_per_cloud=(
            args.quality_centers_per_cloud
            if args.variant == "quality_guided_full" else 0
        ),
        coverage_target=(
            args.quality_coverage_target
            if args.variant == "quality_guided_full" else 0.0
        ),
        max_centers_per_cloud=(
            args.quality_max_centers_per_cloud
            if args.variant == "quality_guided_full" else 0
        ),
    )
    generator = torch.Generator().manual_seed(
        args.seed + (300 if split_name == "train" else 400)
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": (args.quality_batch_size if args.variant == "quality_guided_full" else 1),
        "shuffle": False,
        "num_workers": int(args.quality_workers),
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
    }
    if args.quality_workers > 0:
        # set_epoch() is called before iter(loader) every epoch. Keep workers
        # non-persistent so each new iterator sees the freshly rebuilt coverage
        # schedule. Prefetch overlaps the expensive 500× local-PCA construction
        # with GPU compute.
        loader_kwargs["worker_init_fn"] = pf_base.seed_worker
        loader_kwargs["prefetch_factor"] = int(args.quality_prefetch_factor)
        loader_kwargs["persistent_workers"] = False
    return DataLoader(**loader_kwargs)


def _locked_weight_tensor(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [LOCKED_GUIDANCE_WEIGHTS[h] for h in DIMENSIONS],
        dtype=torch.float32,
        device=device,
    )


def _matched_quality_terms_locked(
    q_cand: torch.Tensor,
    q_clean: torch.Tensor,
    q_noisy: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Matched quality + pure hinge ranking on one aligned 500-point patch.

    ``structure_fidelity`` has zero optimization weight by design.  The rank
    term is a true hinge: once the requested margin is reached, its
    contribution is exactly zero.
    """
    if q_cand.shape != q_clean.shape or q_cand.shape != q_noisy.shape:
        raise ValueError(
            f"Mismatched QNet score shapes: cand={tuple(q_cand.shape)} "
            f"clean={tuple(q_clean.shape)} noisy={tuple(q_noisy.shape)}"
        )
    if q_cand.ndim != 2 or q_cand.shape[-1] != len(DIMENSIONS):
        raise ValueError(f"Unexpected QNet score shape {tuple(q_cand.shape)}")

    per_head_matched = torch.nn.functional.smooth_l1_loss(
        q_cand, q_clean, reduction="none"
    )
    per_head_rank = torch.relu(
        LOCKED_RANK_MARGIN - (q_cand - q_noisy)
    )
    weight_row = weights.view(1, -1)
    L_matched = (per_head_matched * weight_row).sum(dim=-1).mean()
    L_rank = (per_head_rank * weight_row).sum(dim=-1).mean()

    diag = {
        "q_cand": q_cand.detach().mean(dim=0).cpu().tolist(),
        "q_clean": q_clean.detach().mean(dim=0).cpu().tolist(),
        "q_noisy": q_noisy.detach().mean(dim=0).cpu().tolist(),
        "per_head_matched": per_head_matched.detach().mean(dim=0).cpu().tolist(),
        "per_head_rank_hinge": per_head_rank.detach().mean(dim=0).cpu().tolist(),
        "rank_margin": LOCKED_RANK_MARGIN,
    }
    return L_matched, L_rank, diag


def _sample_difficulty_locked(
    q_noisy: torch.Tensor,
    q_clean: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """How much quality headroom remains on this same aligned patch."""
    gap = torch.relu(q_clean - q_noisy)
    return (gap * weights.view(1, -1)).sum(dim=-1).mean()


def _grad_of_retained(loss: torch.Tensor, points: torch.Tensor) -> torch.Tensor | None:
    """Probe d(loss)/d(points) without consuming the graph needed by total_loss.backward()."""
    if not loss.requires_grad or not points.requires_grad:
        return None
    return torch.autograd.grad(
        loss,
        points,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]


def quality_guided_full_loss(
    pointfilter: torch.nn.Module,
    qnet: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    guidance_weights: dict[str, float],
    log: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The complete Quality-Guided objective plus clean-aligned geometry supervision.

    One candidate, one space.  The candidate is the clean-aligned patch's noisy
    points denoised through PointFilter, scored against that same anchor's clean
    patch in the shared clean PCA frame -- the protocol that reproduces the
    480-patch bank exactly (``baseline``/``quality`` match at relative error 0).

    The conflict gate needs ``grad L_D`` at P_hat.  It uses the differentiable
    patch Chamfer to the clean patch, taken at the same P_hat as ``grad L_Q``,
    so both gradients are measured on the same object in the same coordinates.
    That is the denoising direction this loss is being balanced against.
    """
    r = _locked_weight_tensor(device)

    noisy_ref = batch["noisy_ref"].float().to(device)
    clean_ref = batch["clean_ref"].float().to(device)
    frame = batch["frame"].float().to(device)
    radius = batch["radius"].float().to(device)
    center = batch["center"].float().to(device)

    # --- candidate P_hat (differentiable) ------------------------------------
    cand_ref, _ = pf_base.knn_pf_candidate(pointfilter, noisy_ref, args.quality_pf_k)
    cand_ref = cand_ref.clone().requires_grad_(True)

    cand_raw = to_raw(cand_ref, frame, radius, center)      # [1,500,3]
    clean_raw = to_raw(clean_ref, frame, radius, center)    # [1,500,3]
    noisy_raw = to_raw(noisy_ref, frame, radius, center)    # [1,500,3]

    # --- one QNet pass per cloud; the terms only combine the results --------
    q_cand = qnet_scores(qnet, cand_raw)                    # differentiable
    q_clean = qnet_scores(qnet, clean_raw).detach()
    q_noisy = qnet_scores(qnet, noisy_raw).detach()

    # Matched-quality + ranking.  L_global and L_local collapse into the single
    # matched term: the patch is the local unit, so there is one cloud to match.
    L_matched, L_rank, m_diag = _matched_quality_terms_locked(
        q_cand, q_clean, q_noisy, r,
    )

    L_Q = LOCKED_BETA_MATCHED * L_matched + LOCKED_GAMMA_RANK * L_rank

    # Direct clean-aligned geometry anchor.  Q/geometry conflict gates and
    # sample-difficulty weights are intentionally absent from this objective.
    D = torch.cdist(cand_raw[0], clean_raw[0])
    L_D = D.min(dim=1).values.mean() + D.min(dim=0).values.mean()
    # Probe the QNet coordinate gradient only once for the run sanity check.
    grad_quality = None if "quality_grad_norm_at_candidate" in log else _grad_of_retained(L_Q, cand_ref)
    quality_grad_norm = float(grad_quality.detach().norm().cpu()) if grad_quality is not None else float(log.get("quality_grad_norm_at_candidate", 0.0))
    candidate_has_grad_fn = cand_ref.grad_fn is not None
    log.update({
        "L_matched": float(L_matched.detach().cpu()),
        "L_rank": float(L_rank.detach().cpu()),
        "L_Q": float(L_Q.detach().cpu()),
        "L_geo": float(L_D.detach().cpu()),
        "L_denoise_probe": float(L_D.detach().cpu()),
        "quality_grad_norm_at_candidate": quality_grad_norm,
        "candidate_has_grad_fn": bool(candidate_has_grad_fn),
        "head_indices": {h: i for i, h in enumerate(DIMENSIONS)},
        "guidance_weights": {h: float(LOCKED_GUIDANCE_WEIGHTS[h]) for h in DIMENSIONS},
        "matched": m_diag,
    })
    return L_Q, L_D


def to_raw(
    points_ref: torch.Tensor,
    frame: torch.Tensor,
    radius: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.bmm(points_ref, frame.transpose(1, 2))
        * radius.view(-1, 1, 1)
        + center.view(-1, 1, 3)
    )


def qnet_scores(qnet: torch.nn.Module, candidate_raw: torch.Tensor) -> torch.Tensor:
    """Exact frozen-QNet preprocessing; intentionally not wrapped in no_grad."""
    qnet_input = candidate_raw - candidate_raw.mean(dim=1, keepdim=True)
    logits = qnet(qnet_input)
    if logits.shape[-1] != 5:
        raise ValueError(f"Unexpected QNet output shape {tuple(logits.shape)}")
    return torch.sigmoid(logits)


def quality_objective(scores: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "overall":
        return (1.0 - scores[:, 4]).mean()
    if variant == "overall_detail":
        return ((1.0 - scores[:, 4]) + 0.5 * (1.0 - scores[:, 1])).mean()
    if variant == "overall_only_multi_patch_online":
        # Overall-only aggregation over multiple PF-output local patches.
        return (1.0 - scores[:, 4]).mean()
    if variant == "baseline":
        return scores.new_zeros(())
    raise ValueError(variant)


def candidate_and_scores(
    pointfilter: torch.nn.Module,
    qnet: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    pf_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    noisy_ref = batch["noisy_ref"].float().to(device, non_blocking=True)
    frame = batch["frame"].float().to(device, non_blocking=True)
    radius = batch["radius"].float().to(device, non_blocking=True)
    center = batch["center"].float().to(device, non_blocking=True)
    candidate_ref, offset = pf_base.knn_pf_candidate(pointfilter, noisy_ref, pf_k)
    candidate_raw = to_raw(candidate_ref, frame, radius, center)
    scores = qnet_scores(qnet, candidate_raw)
    return candidate_ref, candidate_raw, scores, offset


def online_candidate_and_scores(
    pointfilter: torch.nn.Module,
    qnet: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denoise 500 online centres and score their world-coordinate candidate."""
    patches = batch["patches"].float().to(device, non_blocking=True)
    if patches.shape[0] != 1 or patches.ndim != 4:
        raise ValueError(f"Expected online patches [1,P,N,3], got {tuple(patches.shape)}")
    patches = patches.squeeze(0).transpose(1, 2).contiguous()
    offsets = pointfilter(patches)
    inverse_rotation = (
        batch["inverse_rotation"].squeeze(0).float().to(device, non_blocking=True)
    )
    center = batch["center"].squeeze(0).float().to(device, non_blocking=True)
    radius = batch["radius"].float().to(device, non_blocking=True).reshape(1, 1)
    candidate_raw = (
        torch.bmm(inverse_rotation, offsets.unsqueeze(-1)).squeeze(-1) * radius + center
    )
    candidate_batch = candidate_raw.unsqueeze(0)
    scores = qnet_scores(qnet, candidate_batch)
    return candidate_batch, scores


def dynamic_online_quality_geo_loss(
    pointfilter: torch.nn.Module,
    qnet: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    log: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quality-before/after and clean anchor on a dynamically sampled parent-cloud ROI."""
    patches = batch["patches"].float().to(device, non_blocking=True)
    if patches.ndim != 4:
        raise ValueError(f"Expected dynamic patches [B,P,N,3], got {tuple(patches.shape)}")
    batch_size, point_count, context_count, _ = patches.shape
    pf_input = patches.reshape(batch_size * point_count, context_count, 3).transpose(1, 2).contiguous()
    offsets = pointfilter(pf_input)
    inverse_rotation = batch["inverse_rotation"].float().to(device, non_blocking=True).reshape(-1, 3, 3)
    centers = batch["center"].float().to(device, non_blocking=True).reshape(-1, 3)
    radius = (
        batch["radius"].float().to(device, non_blocking=True)
        .reshape(batch_size, 1, 1).expand(batch_size, point_count, 1).reshape(-1, 1)
    )
    candidate = (
        torch.bmm(inverse_rotation, offsets.unsqueeze(-1)).squeeze(-1) * radius + centers
    ).reshape(batch_size, point_count, 3)
    noisy = batch["noisy_points"].float().to(device, non_blocking=True)
    clean = batch["clean_points"].float().to(device, non_blocking=True)
    if candidate.shape != noisy.shape or candidate.shape != clean.shape:
        raise ValueError(
            f"Dynamic triplet mismatch: candidate={tuple(candidate.shape)} "
            f"noisy={tuple(noisy.shape)} clean={tuple(clean.shape)}"
        )

    q_candidate = qnet_scores(qnet, candidate)
    q_noisy = qnet_scores(qnet, noisy).detach()
    q_clean = qnet_scores(qnet, clean).detach()
    weights = _locked_weight_tensor(device)
    matched, rank, diag = _matched_quality_terms_locked(
        q_candidate, q_clean, q_noisy, weights
    )
    quality = LOCKED_BETA_MATCHED * matched + LOCKED_GAMMA_RANK * rank

    # Frozen-QNet quality headroom of the same aligned noisy/clean ROIs.
    # This is detached and used ONLY to adapt the QNet-vs-geometry share.
    # Harder ROIs (larger clean-noisy gap) get relatively more QNet guidance;
    # easier ROIs get relatively more geometry stabilization.
    difficulty_per_item = (
        torch.relu(q_clean - q_noisy) * weights.view(1, -1)
    ).sum(dim=-1)
    difficulty = difficulty_per_item.mean()

    # Batched bidirectional unsquared Chamfer. quality_batch_size may be >1;
    # the objective is the mean over ROIs in the batch.
    distances = torch.cdist(candidate, clean)  # [B, P, P]
    geometry_per_item = (
        distances.min(dim=2).values.mean(dim=1)
        + distances.min(dim=1).values.mean(dim=1)
    )
    geometry = geometry_per_item.mean()

    grad_quality = None if "quality_grad_norm_at_candidate" in log else _grad_of_retained(quality, candidate)
    grad_norm = (
        float(grad_quality.detach().norm().cpu())
        if grad_quality is not None
        else float(log.get("quality_grad_norm_at_candidate", 0.0))
    )
    log.update({
        "L_matched": float(matched.detach().cpu()),
        "L_rank": float(rank.detach().cpu()),
        "L_Q": float(quality.detach().cpu()),
        "L_geo": float(geometry.detach().cpu()),
        "difficulty": float(difficulty.detach().cpu()),
        "difficulty_min_in_batch": float(difficulty_per_item.detach().min().cpu()),
        "difficulty_max_in_batch": float(difficulty_per_item.detach().max().cpu()),
        "L_denoise_probe": float(geometry.detach().cpu()),
        "quality_grad_norm_at_candidate": grad_norm,
        "candidate_has_grad_fn": candidate.grad_fn is not None,
        "dynamic_shapes": [str(x) for x in batch["shape"]],
        "dynamic_sigmas": [str(x) for x in batch["sigma"]],
        "dynamic_center_indices": [int(x) for x in batch["online_center_index"].reshape(-1)],
        "candidate_shape": list(candidate.shape),
        "parent_cloud_context": True,
        "matched": diag,
    })
    return quality, geometry


def farthest_point_sampling(points: torch.Tensor, m: int) -> torch.Tensor:
    """Standard FPS over a single cloud; returns world-centroid-frees of the M anchors."""
    device = points.device
    n = int(points.shape[0])
    m = int(m)
    if m > n:
        raise ValueError(f"FPS requests {m} anchors from {n} points")
    pts = points.detach()
    idx = torch.zeros(m, dtype=torch.long, device=device)
    distances = torch.full((n,), float("inf"), dtype=pts.dtype, device=device)
    farthest = torch.as_tensor(0, dtype=torch.long, device=device)
    for i in range(m):
        idx[i] = farthest
        centroid = pts[farthest]
        dist = ((pts - centroid) ** 2).sum(-1)
        distances = torch.minimum(distances, dist)
        farthest = int(torch.argmax(distances))
    return idx


def _quality_patch_indices_from_subcloud(
    subcloud_detached: torch.Tensor, n_anchors: int, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """FPS anchors then kNN gather indices inside the candidate subcloud only."""
    anchors = farthest_point_sampling(subcloud_detached, n_anchors)
    dist = torch.cdist(
        subcloud_detached,
        subcloud_detached[anchors],
    )  # [M, N]
    nn = dist.topk(min(patch_size, dist.shape[0]), largest=False, dim=0).indices  # [patch_size, N]
    return anchors, nn.transpose(0, 1)  # [N, patch_size]


class MultiPatchOnlineQualityDataset(OnlineQualityPatchDataset):
    """M independent online PF context patches whose aggregate quality feeds the loss.

    Reuses ``OnlineQualityPatchDataset``'s source plumbing (parent-cloud
    ``bbdiag`` radius, official ``DeterministicEvaluationPatchDataset``, stable
    per-epoch sampling).  Unlike the single-centre variant, each of ``num_centers``
    centres is denoised from *its own* official evaluation patch, so the produced
    candidate subcloud ``[M,3]`` is a set of genuine PF outputs for the current
    noisy cloud -- never clean-aligned.
    """

    def __init__(
        self,
        noisy_root: Path,
        split_path: Path,
        split_name: str,
        sigmas: tuple[str, ...],
        patch_radius: float,
        points_per_patch: int,
        seed: int,
        work_root: Path,
        samples_per_epoch: int,
        num_centers: int,
    ):
        super().__init__(
            noisy_root=noisy_root,
            split_path=split_path,
            split_name=split_name,
            sigmas=sigmas,
            patch_radius=patch_radius,
            points_per_patch=points_per_patch,
            seed=seed,
            work_root=work_root,
            samples_per_epoch=samples_per_epoch,
        )
        self.num_centers = int(num_centers)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair_index = (int(index) + self.epoch) % len(self.pairs)
        shape, sigma = self.pairs[pair_index]
        source = self._get_source(shape, sigma)
        cloud = source["cloud"]
        tree: sp.cKDTree = source["tree"]
        dataset: DeterministicEvaluationPatchDataset = source["dataset"]

        item_seed = self._seed32(
            self.seed,
            100 if self.split_name == "train" else 200,
            self.epoch,
            int(index),
        )
        rng = np.random.RandomState(item_seed)
        # Distinct centres sampled without replacement from the whole noisy cloud.
        center_order = rng.permutation(len(cloud)).astype(np.int64).tolist()

        patches: list[torch.Tensor] = []
        inverse_rotations: list[torch.Tensor] = []
        centers: list[torch.Tensor] = []
        center_indices: list[int] = []
        cursor = 0
        while len(patches) < self.num_centers and cursor < len(center_order):
            center_index = int(center_order[cursor])
            cursor += 1
            sampling_seed = self._seed32(item_seed, 7919, center_index)
            dataset.set_sampling_seed(sampling_seed)
            value = dataset[center_index]
            if value is None:
                continue
            patch, inverse_rotation, center = value
            patches.append(patch)
            inverse_rotations.append(inverse_rotation)
            centers.append(torch.as_tensor(center))
            center_indices.append(center_index)

        if len(patches) < self.num_centers:
            raise RuntimeError(
                f"Could not collect {self.num_centers} online centres for "
                f"{shape} sigma={sigma} index={index}: only {len(patches)}"
            )

        return {
            "patches": torch.stack(patches),            # [M, 500, 3]
            "inverse_rotation": torch.stack(inverse_rotations),  # [M, 3, 3]
            "center": torch.stack(centers),             # [M, 3]
            "radius": torch.tensor(source["radius"], dtype=torch.float32),
            "shape": shape,
            "sigma": sigma,
            "center_indices": torch.from_numpy(np.asarray(center_indices, dtype=np.int64)),
            "sampling_seeds": torch.from_numpy(
                np.asarray(
                    [self._seed32(item_seed, 7919, int(ci)) for ci in center_indices],
                    dtype=np.int64,
                )
            ),
        }


def build_multi_patch_online_quality_loader(
    args: argparse.Namespace,
    split_name: str,
    samples_per_epoch: int,
) -> DataLoader:
    sigmas = tuple(part.strip() for part in args.quality_sigmas.split(",") if part.strip())
    dataset = MultiPatchOnlineQualityDataset(
        noisy_root=args.quality_noisy_root,
        clean_root=args.quality_clean_cloud_root,
        split_path=args.shape_split,
        split_name=split_name,
        sigmas=sigmas,
        patch_radius=args.patch_radius,
        points_per_patch=args.quality_points_per_patch,
        seed=args.seed,
        work_root=args.quality_work_root,
        samples_per_epoch=samples_per_epoch,
        num_centers=args.quality_num_centers,
    )
    generator = torch.Generator().manual_seed(
        args.seed + (500 if split_name == "train" else 600)
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def multi_patch_candidate_and_scores(
    pointfilter: torch.nn.Module,
    qnet: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    n_anchors: int,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Denoise M online centres, then score N FPS-anchored local patches.

    Returns (candidate_subcloud [M,3], quality_patches [N,500,3],
    scores [N,5], gather_error float).  ``gather_error`` is
    ``max|candidate_subcloud[nn_idx] - quality_patches|`` and must be exactly 0:
    it documents that quality patches come from the PF output subcloud only.
    """
    patches = batch["patches"].float().to(device, non_blocking=True)  # [1, M, 500, 3]
    if patches.ndim != 4 or patches.shape[0] != 1:
        raise ValueError(f"Expected patches [1,M,500,3], got {tuple(patches.shape)}")
    m = patches.shape[1]
    inverse_rotation = batch["inverse_rotation"].float().to(device, non_blocking=True).squeeze(0)  # [M,3,3]
    center = batch["center"].float().to(device, non_blocking=True).squeeze(0)  # [M,3]
    radius = batch["radius"].float().to(device, non_blocking=True).reshape(1)  # [1]

    all_patches = patches.squeeze(0)  # [M, 500, 3]
    offsets_list = []
    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        chunk_patches = all_patches[start:stop].transpose(1, 2).contiguous()  # [c,3,500]
        offsets_list.append(pointfilter(chunk_patches))  # [c,3]
    offsets = torch.cat(offsets_list, dim=0)  # [M,3]

    candidate_subcloud = (
        torch.bmm(inverse_rotation, offsets.unsqueeze(-1)).squeeze(-1) * radius + center
    )  # [M,3]

    with torch.no_grad():
        _, nn_idx = _quality_patch_indices_from_subcloud(
            candidate_subcloud.detach(), n_anchors, patches.shape[2]
        )  # nn_idx [N,500]
    quality_patches = candidate_subcloud[nn_idx]  # [N,500,3]; gradients flow through
    gathered_check = candidate_subcloud[nn_idx]
    gather_error = float((gathered_check - quality_patches).abs().max().detach().cpu())
    scores = qnet_scores(qnet, quality_patches)  # [N,5]
    return candidate_subcloud, quality_patches, scores, gather_error


@torch.no_grad()
def patch_metrics(
    candidate_ref: torch.Tensor,
    clean_ref: torch.Tensor,
    candidate_raw: torch.Tensor,
    clean_raw: torch.Tensor,
    pca_k: int,
) -> dict[str, np.ndarray]:
    distances = torch.cdist(candidate_raw, clean_raw)
    chamfer = distances.min(dim=2).values.mean(dim=1) + distances.min(dim=1).values.mean(dim=1)
    candidate_normals, candidate_curvature = pf_base.local_pca(candidate_ref, pca_k)
    clean_normals, clean_curvature = pf_base.local_pca(clean_ref, pca_k)
    nearest = torch.cdist(clean_ref, candidate_ref).argmin(dim=-1)
    batch_index = torch.arange(candidate_ref.shape[0], device=candidate_ref.device)[:, None]
    matched_normals = candidate_normals[batch_index, nearest]
    matched_curvature = candidate_curvature[batch_index, nearest]
    normal = 1.0 - (clean_normals * matched_normals).sum(dim=-1).abs().clamp(0.0, 1.0)
    curvature = (clean_curvature - matched_curvature).abs()
    return {
        "chamfer_distance": chamfer.cpu().numpy(),
        "normal_error": normal.mean(dim=1).cpu().numpy(),
        "curvature_error": curvature.mean(dim=1).cpu().numpy(),
    }


@torch.no_grad()
def validate(
    pointfilter: torch.nn.Module,
    qnet: torch.nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    prediction_path: Path | None = None,
    record_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pointfilter.eval()
    qnet.eval()
    rows: list[dict[str, Any]] = []
    predicted: list[np.ndarray] = []
    noisy_all: list[np.ndarray] = []
    clean_all: list[np.ndarray] = []
    for step, batch in enumerate(loader, start=1):
        if args.max_val_batches > 0 and step > args.max_val_batches:
            break
        noisy_ref = batch["noisy_ref"].float().to(device, non_blocking=True)
        clean_ref = batch["clean_ref"].float().to(device, non_blocking=True)
        frame = batch["frame"].float().to(device, non_blocking=True)
        radius = batch["radius"].float().to(device, non_blocking=True)
        center = batch["center"].float().to(device, non_blocking=True)
        candidate_ref, candidate_raw, scores, _ = candidate_and_scores(
            pointfilter, qnet, batch, device, args.quality_pf_k
        )
        noisy_raw = to_raw(noisy_ref, frame, radius, center)
        clean_raw = to_raw(clean_ref, frame, radius, center)
        metrics = patch_metrics(
            candidate_ref, clean_ref, candidate_raw, clean_raw, args.validation_pca_k
        )
        score_np = scores.cpu().numpy()
        candidate_np = candidate_raw.cpu().numpy()
        noisy_np = noisy_raw.cpu().numpy()
        clean_np = clean_raw.cpu().numpy()
        for index in range(candidate_np.shape[0]):
            rows.append({
                "sample_id": batch["sample_id"][index],
                "shape": batch["shape"][index],
                "sigma": batch["sigma"][index],
                "anchor_id": int(batch["anchor_id"][index]),
                "chamfer_distance": float(metrics["chamfer_distance"][index]),
                "normal_error": float(metrics["normal_error"][index]),
                "curvature_error": float(metrics["curvature_error"][index]),
                "qnet_scores_01": {
                    name: float(score_np[index, dim]) for dim, name in enumerate(DIMENSIONS)
                },
                "qnet_scores_1to5": {
                    name: float(1.0 + 4.0 * score_np[index, dim])
                    for dim, name in enumerate(DIMENSIONS)
                },
            })
        if prediction_path is not None:
            predicted.append(candidate_np.astype(np.float32))
            noisy_all.append(noisy_np.astype(np.float32))
            clean_all.append(clean_np.astype(np.float32))
        if step == 1 or step % args.val_print_every == 0:
            print(f"[val {step}/{len(loader)}] samples={len(rows)}", flush=True)

    if not rows:
        raise RuntimeError("Validation produced no rows")
    aggregate = {
        "num_patches": len(rows),
        "chamfer_distance": float(np.mean([row["chamfer_distance"] for row in rows])),
        "normal_error": float(np.mean([row["normal_error"] for row in rows])),
        "curvature_error": float(np.mean([row["curvature_error"] for row in rows])),
        "qnet_scores_01": {
            name: float(np.mean([row["qnet_scores_01"][name] for row in rows]))
            for name in DIMENSIONS
        },
        "qnet_scores_1to5": {
            name: float(np.mean([row["qnet_scores_1to5"][name] for row in rows]))
            for name in DIMENSIONS
        },
    }
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            sample_ids=np.asarray([row["sample_id"] for row in rows]),
            predicted_points=np.concatenate(predicted, axis=0),
            noisy_points=np.concatenate(noisy_all, axis=0),
            clean_points=np.concatenate(clean_all, axis=0),
        )
    if record_path is not None:
        write_jsonl(record_path, rows)
    return aggregate, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("baseline", "overall", "overall_detail", "overall_only_multi_patch_online",
                 "quality_guided_full"),
        required=True,
    )
    parser.add_argument("--lambda-q", type=float, default=0.01, help="Legacy metadata only for quality_guided_full")
    parser.add_argument("--train-root", type=Path, default=DEFAULT_PUNET_PF_TRAIN)
    parser.add_argument("--beta-aux", type=float, default=0.1)
    parser.add_argument(
        "--quality-fraction", type=float, default=0.7,
        help=(
            "Base QNet share before dynamic difficulty adaptation. With "
            "--dynamic-guidance-ratio this is a prior, not a fixed ratio."
        ),
    )
    parser.add_argument(
        "--dynamic-guidance-ratio", action="store_true",
        help=(
            "Adapt the QNet/geometry share from the frozen-QNet clean-vs-noisy "
            "quality headroom of each quality batch while keeping the auxiliary "
            "budget approximately fixed."
        ),
    )
    parser.add_argument("--difficulty-weight-min", type=float, default=0.5)
    parser.add_argument("--difficulty-weight-max", type=float, default=2.0)
    parser.add_argument(
        "--difficulty-ema-decay", type=float, default=0.99,
        help="EMA decay used to normalize sample/batch quality headroom.",
    )
    parser.add_argument("--loss-ema-decay", type=float, default=0.99)
    parser.add_argument("--loss-ema-eps", type=float, default=1e-8)
    parser.add_argument("--shared-manifest", type=Path, default=pf_base.DEFAULT_SHARED)
    parser.add_argument(
        "--shape-split",
        type=Path,
        default=GAUSSIAN_ROOT / "manifests/shape_split_seed2026_train_test_overlap.json",
    )
    parser.add_argument(
        "--validation-uses-test", action="store_true",
        help=(
            "Intentional leakage ablation: replace Validation shapes with the full "
            "Test split and use them for checkpoint selection."
        ),
    )
    parser.add_argument(
        "--extra-train-shapes", default="dino,cup,gargoyle",
        help=(
            "Comma-separated shapes intentionally kept in BOTH Train and Test. "
            "They are removed from Validation if necessary. "
            "Default: dino,cup,gargoyle."
        ),
    )
    parser.add_argument("--pf-checkpoint", type=Path, default=pf_base.DEFAULT_PF_CKPT)
    parser.add_argument("--qnet-checkpoint", type=Path, default=pf_base.DEFAULT_QNET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--early-stop-patience", type=int, default=10,
        help=(
            "Stop when Validation Chamfer Distance has failed to set a new strict "
            "minimum for this many consecutive epochs; 0 disables early stopping"
        ),
    )
    parser.add_argument(
        "--early-stop-min-delta", type=float, default=0.0,
        help=(
            "Required absolute decrease in Validation Chamfer Distance to count as "
            "an improvement; default 0 means any strict decrease resets patience"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patch-radius", type=float, default=0.05)
    parser.add_argument("--pf-points-per-patch", type=int, default=500)
    parser.add_argument(
        "--patches-per-shape", type=int, default=7466,
        help=(
            "Default 7466 for the 30-shape overlap run so the PointFilter "
            "optimizer-step budget stays approximately equal to the old "
            "28-shape run that used 8000 patches per prepared shape."
        ),
    )
    parser.add_argument("--quality-batch-size", type=int, default=4)
    parser.add_argument(
        "--quality-workers", type=int, default=8,
        help="CPU workers for dynamic quality-ROI construction; 0 disables multiprocessing",
    )
    parser.add_argument(
        "--quality-prefetch-factor", type=int, default=2,
        help="Prefetched quality batches per worker when --quality-workers > 0",
    )
    parser.add_argument(
        "--quality-pf-k", type=int, default=128,
        help="PointFilter kNN size for clean-aligned candidate generation and legacy evaluation",
    )
    parser.add_argument("--quality-noisy-root", type=Path, default=GAUSSIAN_ROOT / "noisy")
    parser.add_argument(
        "--quality-clean-cloud-root", type=Path,
        default=GAUSSIAN_ROOT / "clean_normalized",
    )
    parser.add_argument("--quality-points-per-patch", type=int, default=500)
    parser.add_argument(
        "--quality-centers-per-cloud", type=int, default=100,
        help=(
            "Minimum number of coverage-aware ROI centres per Train shape×sigma parent cloud. "
            "Sampling continues beyond this value until --quality-coverage-target is reached."
        ),
    )
    parser.add_argument(
        "--quality-coverage-target", type=float, default=0.95,
        help=(
            "Required unique source-point coverage for every Train shape×sigma parent cloud. "
            "0 disables coverage-aware stopping and falls back to fixed-count FPS."
        ),
    )
    parser.add_argument(
        "--quality-max-centers-per-cloud", type=int, default=500,
        help=(
            "Safety cap for coverage-aware ROIs per parent cloud. Training aborts before the "
            "epoch if this cap cannot achieve --quality-coverage-target."
        ),
    )
    parser.add_argument(
        "--quality-work-root", type=Path,
        default=DEFAULT_OUTPUT_ROOT / "_online_quality_work",
    )
    parser.add_argument(
        "--quality-sigmas", default="0.005,0.010,0.015,0.020,0.025",
    )
    parser.add_argument(
        "--online-quality-samples-per-epoch", type=int, default=0,
        help="0 matches the number of PointFilter geometry steps",
    )
    parser.add_argument(
        "--quality-num-centers", type=int, default=2048,
        help="M: online centres denoised into the candidate subcloud (multi-patch variant)",
    )
    parser.add_argument(
        "--quality-n-anchors", type=int, default=16,
        help="N: FPS anchors whose local patches are scored by QNet (multi-patch variant)",
    )
    parser.add_argument(
        "--quality-pf-chunk", type=int, default=128,
        help="PointFilter forward chunk size for the M centres (multi-patch variant)",
    )
    parser.add_argument(
        "--quality-clean-dir", type=Path, default=PATCH_ROOT_CLEAN,
        help="Clean-aligned patch root; informational, paths come from --shared-manifest",
    )
    parser.add_argument(
        "--quality-qnet-summary", type=Path, default=DEFAULT_QNET_SUMMARY,
        help="QNet run summary.json (kept for experiment metadata; PLCC is not used as an optimization weight)",
    )
    parser.add_argument("--validation-pca-k", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--support-multiple", type=float, default=4.0)
    parser.add_argument("--support-angle", type=float, default=15.0)
    parser.add_argument("--repulsion-alpha", type=float, default=0.97)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--val-print-every", type=int, default=100)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.beta_aux < 0:
        raise ValueError("--beta-aux must be non-negative")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be >= 0")
    if args.quality_batch_size < 1:
        raise ValueError("--quality-batch-size must be positive")
    if args.quality_workers < 0:
        raise ValueError("--quality-workers must be >= 0")
    if args.quality_prefetch_factor < 1:
        raise ValueError("--quality-prefetch-factor must be >= 1")
    if not 0.0 <= args.quality_coverage_target <= 1.0:
        raise ValueError("--quality-coverage-target must be in [0,1]")
    if args.quality_centers_per_cloud < 1 and args.variant == "quality_guided_full":
        raise ValueError("quality_guided_full requires --quality-centers-per-cloud >= 1")
    if args.quality_max_centers_per_cloud < args.quality_centers_per_cloud:
        raise ValueError(
            "--quality-max-centers-per-cloud must be >= --quality-centers-per-cloud"
        )
    if not 0.0 <= args.quality_fraction <= 1.0:
        raise ValueError("--quality-fraction must be in [0,1]")
    if args.difficulty_weight_min <= 0:
        raise ValueError("--difficulty-weight-min must be > 0")
    if args.difficulty_weight_max < args.difficulty_weight_min:
        raise ValueError("--difficulty-weight-max must be >= --difficulty-weight-min")
    if not 0.0 <= args.loss_ema_decay < 1.0:
        raise ValueError("--loss-ema-decay must be in [0,1)")
    if args.variant == "quality_guided_full":
        allowed_quality_guided_lambdas = {0.01, 0.02, 0.05, 0.1, 0.2}
        if args.lambda_q not in allowed_quality_guided_lambdas:
            raise ValueError(
                "quality_guided_full requires --lambda-q in "
                "[0.01, 0.02, 0.05, 0.1, 0.2]"
            )
    elif args.variant == "baseline":
        if not math.isclose(args.lambda_q, 0.0):
            raise ValueError("Baseline requires --lambda-q 0")
    elif args.lambda_q not in {0.01, 0.02, 0.05, 0.1}:
        raise ValueError("Guided variants require lambda_q in [0.01, 0.02, 0.05, 0.1]")
    if args.output_dir is None:
        if args.variant == "baseline":
            suffix = "baseline"
        elif args.variant == "quality_guided_full":
            suffix = f"quality_guided_full_lambda_{args.lambda_q:g}"
        else:
            suffix = f"{args.variant}_online_lambda_{args.lambda_q:g}"
        args.output_dir = DEFAULT_OUTPUT_ROOT / suffix
    return args


def main() -> None:
    args = parse_args()
    pf_base.seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_override = build_effective_shape_split(args)

    pointfilter = pf_base.load_pf(args.pf_checkpoint, device)
    qnet = pf_base.load_frozen_qnet(args.qnet_checkpoint, device)
    qnet.eval()
    for parameter in qnet.parameters():
        parameter.requires_grad_(False)
    qnet_hash_before = pf_base.state_sha256(qnet)
    qnet_file_hash_before = pf_base.sha256(args.qnet_checkpoint)
    pf_initial_hash = pf_base.state_sha256(pointfilter)

    pf_metadata_path = args.train_root / "metadata.json"
    if not pf_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {pf_metadata_path}; prepare PointFilter data for effective split {args.shape_split} first"
        )
    pf_metadata = json.loads(pf_metadata_path.read_text(encoding="utf-8"))
    manifest_split = json.loads(args.shape_split.read_text(encoding="utf-8"))
    prepared_shapes = set(pf_metadata.get("shapes", []))
    required_shapes = set(manifest_split["train"])
    if prepared_shapes != required_shapes:
        missing = sorted(required_shapes - prepared_shapes)
        stale_extra = sorted(prepared_shapes - required_shapes)
        raise RuntimeError(
            "PointFilter train-root does not match the effective Train split. "
            f"missing_in_train_root={missing}, stale_extra_in_train_root={stale_extra}. "
            "Re-run prepare_punet_manifest_pf_train.py with the effective split and use a NEW --train-root; "
            "do not bypass this check, otherwise L_PF and QNet guidance would see different shape sets."
        )

    pf_loader = build_pf_loader(args)
    if Path(pf_metadata.get("source_punet_root", "")).resolve() != (ROOT / "PUNet/train/50000_poisson").resolve():
        raise RuntimeError("PointFilter training data were not prepared from PUNet/train")
    online_samples = (
        args.online_quality_samples_per_epoch
        if args.online_quality_samples_per_epoch > 0
        else len(pf_loader)
    )
    if args.variant == "overall_only_multi_patch_online":
        train_quality_loader = build_multi_patch_online_quality_loader(
            args, "train", samples_per_epoch=online_samples
        )
    elif args.variant in {"overall", "overall_detail", "quality_guided_full"}:
        train_quality_loader = build_online_quality_loader(
            args, "train", samples_per_epoch=online_samples
        )
    else:
        # baseline and quality_guided_full do not construct the online loader.
        train_quality_loader = None
    val_loader = build_quality_loader(args, "val", shuffle=False)
    train_shapes = (
        set(train_quality_loader.dataset.parent_shapes)
        if train_quality_loader is not None
        else set(json.loads(args.shape_split.read_text(encoding="utf-8"))["train"])
    )
    val_shapes = {row["shape"] for row in val_loader.dataset.rows}
    expected_train_val_overlap = (
        set(_parse_shape_csv(args.extra_train_shapes))
        if args.validation_uses_test else set()
    )
    if train_shapes & val_shapes != expected_train_val_overlap:
        raise RuntimeError(
            "Unexpected Train/Validation parent-cloud overlap: "
            f"actual={sorted(train_shapes & val_shapes)}, "
            f"expected={sorted(expected_train_val_overlap)}"
        )

    # --- quality_guided_full: frozen QNet over dynamic parent-cloud patches ---
    reliability = qgl_plcc = None
    train_clean_loader = None
    if args.variant == "quality_guided_full":
        # PLCC is deliberately NOT converted into optimization weights.
        # These weights are locked from the gradient-guidance design decision.
        reliability = dict(LOCKED_GUIDANCE_WEIGHTS)
        clean_shapes = set(train_quality_loader.dataset.parent_shapes)
        if clean_shapes & val_shapes != expected_train_val_overlap:
            raise RuntimeError(
                "Unexpected clean-target/Validation overlap: "
                f"actual={sorted(clean_shapes & val_shapes)}, "
                f"expected={sorted(expected_train_val_overlap)}"
            )
        print("GUIDANCE_WEIGHTS " + json.dumps({
            "val_plcc_used_for_weighting": False, "weights": reliability,
            "zeroed_heads": list(LOCKED_ZEROED_HEADS),
            "source": "locked design weights; PLCC is evaluation-only",
            "clean_train_shapes": sorted(clean_shapes),
            "dynamic_train_samples_per_epoch": "variable; built each epoch until coverage target",
            "coverage_target": float(args.quality_coverage_target),
            "minimum_centers_per_parent": int(args.quality_centers_per_cloud),
            "max_centers_per_parent": int(args.quality_max_centers_per_cloud),
        }, ensure_ascii=False), flush=True)

    optimizer = torch.optim.SGD(pointfilter.parameters(), lr=args.lr, momentum=args.momentum)
    quality_loss_ema = DetachedLossEMA(args.loss_ema_decay, args.loss_ema_eps)
    geometry_loss_ema = DetachedLossEMA(args.loss_ema_decay, args.loss_ema_eps)
    difficulty_ema = DetachedLossEMA(args.difficulty_ema_decay, args.loss_ema_eps)
    support_angle = args.support_angle / 360.0 * 2.0 * np.pi
    config = {
        "schema_version": "pointfilter_punet_quality_geo_ema_fullcover_v2",
        "variant": args.variant,
        "lambda_q": args.lambda_q,
        "loss": {
            "baseline": "L_PF",
            "overall": "L_PF + lambda_q*(1-q_overall)",
            "overall_detail": "L_PF + lambda_q*((1-q_overall)+0.5*(1-q_detail))",
            "overall_only_multi_patch_online": "L_PF + lambda_q*(1-mean_i q_overall_i) over N FPS-anchored PF-output patches",
            "quality_guided_full": (
                "L_PF + beta_aux*(s_Q*L_Q/EMA(L_Q) + s_G*L_geo/EMA(L_geo)); "
                "with dynamic s_Q/s_G from clean-vs-noisy QNet headroom when enabled; "
                "L_Q=L_matched+0.25*L_rank; EMA values are detached"
            ),
            "L_PF": "original PointFilter 100*(bilateral + repulsion) implementation",
        },
        "quality_guided_full": (
            {
                "beta_matched": LOCKED_BETA_MATCHED,
                "gamma_rank": LOCKED_GAMMA_RANK,
                "beta_aux": args.beta_aux,
                "quality_fraction": args.quality_fraction,
                "geometry_fraction": 1.0 - args.quality_fraction,
                "loss_ema_decay": args.loss_ema_decay,
                "loss_ema_eps": args.loss_ema_eps,
                "geometry_term": (
                    "L_geo = mean_{p in P_hat} min_{q in P_clean} ||p-q||_2 "
                    "+ mean_{q in P_clean} min_{p in P_hat} ||q-p||_2; "
                    "this term directly backpropagates into PointFilter"
                ),
                "loss_scale_normalization": "detached running EMA",
                "head_order": list(DIMENSIONS),
                "head_order_source": "checkpoint config: cleanliness, structure_fidelity, surface_fidelity, sampling_regularity, overall",
                "qnet_val_plcc_used_for_weighting": False,
                "guidance_weights": {h: float(reliability[h]) for h in DIMENSIONS},
                "guidance_weight_rule": "overall=0.5; cleanliness/surface_fidelity/sampling_regularity=1/6 each; structure_fidelity=0",
                "zeroed_heads": list(LOCKED_ZEROED_HEADS),
                "zeroed_heads_reason": (
                    "pre-training autograd audit: gradient alignment with the Chamfer "
                    "gradient at/below chance (median cosine -0.004, positive rate "
                    "0.458 at sigma=0.025). PLCC measures score accuracy, not gradient "
                    "usefulness, so the head keeps no weight."
                ),
                "gradient_alignment_audit": str(
                    ROOT / "PointFilter_frozen_quality_stage3/qnet_gradient_alignment_audit/alignment.json"
                ),
                "quality_unit": "a dynamically sampled 500-point ROI from a complete noisy Train parent cloud",
                "spatial_center_sampling": (
                    "epoch-dependent coverage-aware FPS: start from an epoch-dependent centre, "
                    "take the exact 500-NN source ROI, then repeatedly choose the farthest still-"
                    "uncovered source point until the requested unique-point coverage is reached"
                ),
                "coverage_target_per_parent_cloud": args.quality_coverage_target,
                "minimum_centers_per_parent_cloud": args.quality_centers_per_cloud,
                "maximum_centers_per_parent_cloud": args.quality_max_centers_per_cloud,
                "expected_dynamic_rois_per_epoch": "variable; determined by coverage target at epoch start",
                "matched_term": (
                    "sum_k r_k Huber(q_k(P_hat) - q_k(P_clean)); clean scores detached. "
                    "The design's separate global and local terms collapse into this one: "
                    "with a 500-point parent there is a single cloud to match against, so "
                    "8x500 nested neighbourhoods would each be the whole patch and the "
                    "anchor weighting would carry no localisation."
                ),
                "rank_term": "sum_k w_k ReLU(m - (q_k(P_hat) - q_k(P_noisy))), m=0.02; pure hinge, exactly zero after margin",
                "sample_difficulty_weighting": bool(args.dynamic_guidance_ratio),
                "dynamic_ratio_rule": (
                    "difficulty=weighted ReLU(q_clean-q_noisy); "
                    "w=clip(difficulty/EMA(difficulty), min, max); "
                    "q_share=(base_q*w)/(base_q*w+1-base_q)"
                    if args.dynamic_guidance_ratio else "disabled"
                ),
                "geometry_alignment_gate": False,
                "clean_target_shapes": (
                    sorted(train_quality_loader.dataset.parent_shapes)
                    if train_quality_loader is not None else None
                ),
                "clean_target_patches": "dynamic; one same-index clean ROI for every scheduled noisy ROI",
                "validation_never_used_as_quality_target": True,
            }
            if args.variant == "quality_guided_full"
            else "not used"
        ),
        "checkpoint_selection": "earliest strict minimum Validation Chamfer Distance; QNet scores excluded",
        "early_stopping": {
            "monitor": "validation.chamfer_distance",
            "mode": "min",
            "patience": args.early_stop_patience,
            "min_delta": args.early_stop_min_delta,
            "rule": (
                "stop after patience consecutive epochs whose Validation Chamfer "
                "Distance does not beat the historical minimum by min_delta"
            ),
        },
        "qnet_checkpoint": str(args.qnet_checkpoint),
        "qnet_checkpoint_sha256": qnet_file_hash_before,
        "qnet_frozen_parameters": True,
        "qnet_eval_mode": True,
        "qnet_forward_no_grad_during_guided_training": False,
        "qnet_input": "candidate_raw - candidate_raw.mean(dim=points); exactly 500 points",
        "quality_patch_source": (
            "dynamic Train parent-cloud ROI: noisy points, PF outputs using full-parent neighborhoods, and same-index clean references"
            if args.variant == "quality_guided_full"
            else (
                "online noisy 500-point neighborhood / multi-patch source"
                if args.variant in {"overall", "overall_detail", "overall_only_multi_patch_online"}
                else "not used"
            )
        ),
        "quality_sampling": (
            "dynamic deterministic coverage-aware shape-sigma sampling from complete Train parent clouds; exact 500-NN ROIs, target unique-point coverage enforced per parent"
            if args.variant == "quality_guided_full"
            else (
                "deterministic from seed, epoch, and sample index; shape-sigma pairs round-robin"
                if args.variant in {"overall", "overall_detail", "overall_only_multi_patch_online"}
                else "not used"
            )
        ),
        "quality_loader_optimization": {
            "quality_batch_size": args.quality_batch_size,
            "quality_workers": args.quality_workers,
            "quality_prefetch_factor": args.quality_prefetch_factor,
            "batch_semantics": "loss is mean over ROIs; full coverage schedule unchanged",
        },
        "qnet_score_activation": "sigmoid(logits), normalized [0,1]",
        "pf_architecture_modified": False,
        "pf_initial_checkpoint": str(args.pf_checkpoint),
        "pf_initial_checkpoint_sha256": pf_base.sha256(args.pf_checkpoint),
        "pf_initial_state_sha256": pf_initial_hash,
        "train_root": str(args.train_root),
        "train_data_protocol": f"PUNet/train 40-shape pool filtered to effective train={len(train_shapes)}; selected shapes may intentionally overlap effective Test; clean plus five Gaussian levels",
        "train_data_metadata": pf_metadata,
        "shared_manifest": str(args.shared_manifest),
        "shape_split": str(args.shape_split),
        "original_shape_split": str(args.original_shape_split),
        "split_override": split_override,
        "constructed_splits": ["train", "val"],
        "train_test_overlap_intentional": True,
        "train_test_overlap_shapes": _parse_shape_csv(args.extra_train_shapes),
        "test_accessed": bool(args.validation_uses_test),
        "test_access_reason": (
            "intentional Test-as-Validation checkpoint-selection ablation"
            if args.validation_uses_test else None
        ),
        "legacy_loader_invalid_patch_policy": "skip only rare PCA-SVD/nonfinite patch via existing my_collate",
        "train_parent_shapes": sorted(train_shapes),
        "val_parent_shapes": sorted(val_shapes),
        "train_quality_candidates_per_epoch": (
            "variable; coverage-aware schedule rebuilt each epoch"
            if train_quality_loader is not None else 0
        ),
        "train_quality_candidate_points": (
            args.quality_points_per_patch if train_quality_loader is not None else 0
        ),
        "multi_patch_online": (
            {
                "num_centers": args.quality_num_centers,
                "n_anchors": args.quality_n_anchors,
                "pf_chunk": args.quality_pf_chunk,
                "quality_loss": "L_quality = 1 - mean_i O_i (Overall-only)",
            }
            if args.variant == "overall_only_multi_patch_online"
            else "not used"
        ),
        "val_quality_patches": len(val_loader.dataset),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(args.output_dir / "config.json", config)

    history: list[dict[str, Any]] = []
    best_cd = float("inf")
    best_epoch = None
    sanity = None
    start_epoch = 1
    early_stop_bad_epochs = 0
    early_stopped = False
    stopped_epoch = None
    resume_path = args.output_dir / "last_state.pth"
    if args.resume:
        if not resume_path.exists():
            resume_path = args.output_dir / "best_validation.pth"
        if not resume_path.exists():
            raise FileNotFoundError(f"No resumable checkpoint in {args.output_dir}")
        resumed = torch.load(resume_path, map_location=device, weights_only=False)
        pointfilter.load_state_dict(resumed["state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        if "quality_loss_ema" in resumed:
            quality_loss_ema.load_state_dict(resumed["quality_loss_ema"])
        if "geometry_loss_ema" in resumed:
            geometry_loss_ema.load_state_dict(resumed["geometry_loss_ema"])
        if "difficulty_ema" in resumed:
            difficulty_ema.load_state_dict(resumed["difficulty_ema"])
        history = list(resumed["history"])
        sanity = resumed.get("sanity")
        start_epoch = int(resumed["epoch"]) + 1
        best_checkpoint = torch.load(
            args.output_dir / "best_validation.pth", map_location="cpu", weights_only=False
        )
        best_cd = float(best_checkpoint["validation"]["chamfer_distance"])
        best_epoch = int(best_checkpoint["epoch"])
        early_state = resumed.get("early_stop_state", {})
        early_stop_bad_epochs = int(
            early_state.get("bad_epochs", max(0, int(resumed["epoch"]) - best_epoch))
        )
        print(
            f"RESUME path={resume_path} start_epoch={start_epoch} "
            f"best_epoch={best_epoch} best_cd={best_cd:.8f} "
            f"early_stop_bad_epochs={early_stop_bad_epochs}/{args.early_stop_patience}",
            flush=True,
        )
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = pf_base.scheduled_lr(epoch - 1, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        pointfilter.train()
        qnet.eval()
        if train_quality_loader is not None:
            train_quality_loader.dataset.set_epoch(epoch)
            quality_iter = iter(train_quality_loader)
        else:
            quality_iter = None
        epoch_values = {"total": [], "pf": [], "quality": [], "overall": [], "detail": []}
        # quality_guided_full per-epoch accumulators
        qg_log: dict[str, Any] = {}
        quality_centers_seen: set[tuple[str, str, int]] = set()
        # Track the ACTUAL source points touched by the 500-point QNet ROIs.
        # This is stronger than merely counting FPS centers.
        quality_points_seen: dict[tuple[str, str], set[int]] = {}
        quality_batches_consumed = 0
        quality_rois_consumed = 0
        pf_steps_this_epoch = (
            min(len(pf_loader), args.max_steps_per_epoch)
            if args.max_steps_per_epoch > 0
            else len(pf_loader)
        )
        quality_batches_total = (
            len(train_quality_loader)
            if args.variant == "quality_guided_full" and train_quality_loader is not None
            else 0
        )
        qg_epoch: dict[str, list[float]] = {
            k: [] for k in ("L_matched", "L_rank", "L_Q", "L_geo",
                            "ema_q", "ema_geo", "normalized_q", "normalized_geo",
                            "difficulty", "ema_difficulty", "difficulty_weight",
                            "dynamic_quality_share", "dynamic_geometry_share",
                            "quality_contribution", "geometry_contribution", "aux_total")}
        for step, pf_batch in enumerate(pf_loader, start=1):
            if args.max_steps_per_epoch > 0 and step > args.max_steps_per_epoch:
                break
            pointfilter.train()
            qnet.eval()
            loss_pf = pf_base.original_pf_loss(
                pointfilter,
                pf_batch,
                device,
                args.support_multiple,
                support_angle,
                args.repulsion_alpha,
            )

            # -----------------------------------------------------------------
            # Full-cover Quality-Guided branch.
            #
            # The PF loader has far fewer optimizer steps than the complete
            # shape×sigma×FPS quality schedule.  Therefore we distribute the
            # ENTIRE quality schedule evenly over the available PF optimizer
            # steps. Each quality *batch* is processed independently and its
            # backward pass is done immediately, so memory stays bounded while
            # B ROIs are parallelized inside each GPU forward.
            # -----------------------------------------------------------------
            if args.variant == "quality_guided_full":
                if loss_pf is not None and not bool(torch.isfinite(loss_pf).detach().cpu()):
                    raise RuntimeError("Non-finite original PointFilter geometry loss")
                if quality_iter is None or train_quality_loader is None:
                    raise RuntimeError("Missing online quality loader for quality_guided_full")

                # Balanced integer partition of Q quality batches over P PF steps:
                # by the final PF step, exactly all Q batches have been consumed.
                target_consumed = (step * quality_batches_total) // pf_steps_this_epoch
                n_quality_this_step = target_consumed - quality_batches_consumed
                if n_quality_this_step < 0:
                    raise RuntimeError("Negative quality schedule allocation")

                # Exact ROI count represented by these quality batches. This
                # matters only for the final partial batch, but keeps gradient
                # averaging identical to a true per-ROI mean.
                roi_total = len(train_quality_loader.dataset)
                qbs = int(args.quality_batch_size)
                start_batch_index = quality_batches_consumed
                rois_this_step_expected = 0
                for batch_index in range(
                    start_batch_index, start_batch_index + n_quality_this_step
                ):
                    start_roi = batch_index * qbs
                    rois_this_step_expected += max(
                        0, min(qbs, roi_total - start_roi)
                    )

                optimizer.zero_grad(set_to_none=True)
                pf_value = 0.0
                if loss_pf is not None:
                    pf_value = float(loss_pf.detach().cpu())
                    loss_pf.backward()

                aux_values_this_step: list[float] = []
                q_values_this_step: list[float] = []
                geo_values_this_step: list[float] = []
                rois_this_step_actual = 0

                # Freeze BN statistics for the auxiliary quality stream, while
                # keeping gradients from QNet scores to PointFilter outputs.
                pointfilter.apply(pf_base.freeze_bn_stats)

                for _q_index in range(n_quality_this_step):
                    try:
                        quality_batch = next(quality_iter)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "Quality schedule ended early; full-cover invariant broken"
                        ) from exc

                    quality_loss, geometry_loss = dynamic_online_quality_geo_loss(
                        pointfilter, qnet, quality_batch, device, qg_log,
                    )
                    if not bool(torch.isfinite(quality_loss).detach().cpu()):
                        raise RuntimeError("Non-finite QNet quality loss")
                    if not bool(torch.isfinite(geometry_loss).detach().cpu()):
                        raise RuntimeError("Non-finite direct geometry loss")

                    # Record center coverage and actual 500-point ROI coverage.
                    shapes = [str(x) for x in quality_batch["shape"]]
                    sigmas = [str(x) for x in quality_batch["sigma"]]
                    centers_batch = [
                        int(x) for x in quality_batch["online_center_index"].reshape(-1)
                    ]
                    neighbor_batch = quality_batch["neighbor_indices"]
                    for bi, (shape, sigma, center_idx) in enumerate(
                        zip(shapes, sigmas, centers_batch)
                    ):
                        quality_centers_seen.add((shape, sigma, center_idx))
                        key = (shape, sigma)
                        point_set = quality_points_seen.setdefault(key, set())
                        point_set.update(
                            int(x) for x in neighbor_batch[bi].reshape(-1).tolist()
                        )

                    ema_q = quality_loss_ema.update(quality_loss)
                    ema_geo = geometry_loss_ema.update(geometry_loss)
                    normalized_q = quality_loss / ema_q
                    normalized_geo = geometry_loss / ema_geo

                    # Dynamic QNet-vs-geometry ratio. The total auxiliary budget
                    # remains approximately beta_aux; only the share changes.
                    difficulty_tensor = quality_loss.new_tensor(float(qg_log["difficulty"]))
                    ema_difficulty = difficulty_ema.update(difficulty_tensor)
                    if args.dynamic_guidance_ratio:
                        difficulty_weight = float(np.clip(
                            float(qg_log["difficulty"]) / float(ema_difficulty),
                            float(args.difficulty_weight_min),
                            float(args.difficulty_weight_max),
                        ))
                    else:
                        difficulty_weight = 1.0

                    base_q = float(args.quality_fraction)
                    base_g = 1.0 - base_q
                    denom = base_q * difficulty_weight + base_g
                    dynamic_q_share = (base_q * difficulty_weight) / max(denom, 1e-12)
                    dynamic_g_share = 1.0 - dynamic_q_share

                    q_contribution = (
                        float(args.beta_aux) * dynamic_q_share * normalized_q
                    )
                    g_contribution = (
                        float(args.beta_aux) * dynamic_g_share * normalized_geo
                    )
                    auxiliary_loss = q_contribution + g_contribution

                    qg_log.update({
                        "ema_q": float(ema_q),
                        "ema_geo": float(ema_geo),
                        "normalized_q": float(normalized_q.detach().cpu()),
                        "normalized_geo": float(normalized_geo.detach().cpu()),
                        "ema_difficulty": float(ema_difficulty),
                        "difficulty_weight": float(difficulty_weight),
                        "dynamic_quality_share": float(dynamic_q_share),
                        "dynamic_geometry_share": float(dynamic_g_share),
                        "quality_contribution": float(q_contribution.detach().cpu()),
                        "geometry_contribution": float(g_contribution.detach().cpu()),
                        "aux_total": float(auxiliary_loss.detach().cpu()),
                    })

                    # dynamic_online_quality_geo_loss() returns a mean over
                    # the B ROIs in this batch. Weight by actual B and normalize
                    # by the exact number of ROIs assigned to this optimizer step.
                    # Backward immediately to keep memory bounded.
                    batch_rois = len(shapes)
                    if rois_this_step_expected > 0:
                        (
                            auxiliary_loss
                            * (float(batch_rois) / float(rois_this_step_expected))
                        ).backward()

                    quality_batches_consumed += 1
                    quality_rois_consumed += batch_rois
                    rois_this_step_actual += batch_rois
                    aux_values_this_step.append(float(auxiliary_loss.detach().cpu()))
                    q_values_this_step.append(float(quality_loss.detach().cpu()))
                    geo_values_this_step.append(float(geometry_loss.detach().cpu()))
                    epoch_values["quality"].append(float(quality_loss.detach().cpu()))
                    for key in qg_epoch:
                        if key in qg_log:
                            qg_epoch[key].append(float(qg_log[key]))

                # With full-cover settings Q >> P, so every step normally has
                # quality ROIs.  This guard also makes debug settings safe.
                if loss_pf is None and n_quality_this_step == 0:
                    continue

                pf_grad_finite = all(
                    parameter.grad is None
                    or bool(torch.isfinite(parameter.grad).all().detach().cpu())
                    for parameter in pointfilter.parameters()
                )
                torch.nn.utils.clip_grad_norm_(pointfilter.parameters(), args.grad_clip)
                optimizer.step()

                mean_aux = float(np.mean(aux_values_this_step)) if aux_values_this_step else 0.0
                mean_q = float(np.mean(q_values_this_step)) if q_values_this_step else 0.0
                mean_geo = float(np.mean(geo_values_this_step)) if geo_values_this_step else 0.0
                total_value = pf_value + mean_aux
                epoch_values["total"].append(total_value)
                epoch_values["pf"].append(pf_value)

                if sanity is None and qg_log:
                    sanity = {
                        "variant": args.variant,
                        "qnet_training": qnet.training,
                        "qnet_all_requires_grad_false": all(
                            not p.requires_grad for p in qnet.parameters()
                        ),
                        "qnet_parameter_grads_none": all(
                            p.grad is None for p in qnet.parameters()
                        ),
                        "qnet_state_unchanged": (
                            pf_base.state_sha256(qnet) == qnet_hash_before
                        ),
                        "pf_gradient_finite": pf_grad_finite,
                        "head_order": list(DIMENSIONS),
                        "guidance_weights": {
                            h: float(reliability[h]) for h in DIMENSIONS
                        },
                        "zeroed_heads": list(LOCKED_ZEROED_HEADS),
                        "beta_aux": float(args.beta_aux),
                        "quality_fraction": float(args.quality_fraction),
                        "geometry_fraction": float(1.0 - args.quality_fraction),
                        "dynamic_guidance_ratio": bool(args.dynamic_guidance_ratio),
                        "difficulty_weight_range": [
                            float(args.difficulty_weight_min),
                            float(args.difficulty_weight_max),
                        ],
                        "first_dynamic_quality_share": float(qg_log.get("dynamic_quality_share", args.quality_fraction)),
                        "first_dynamic_geometry_share": float(qg_log.get("dynamic_geometry_share", 1.0 - args.quality_fraction)),
                        "direct_geometry_supervision": True,
                        "ema_normalization": True,
                        "quality_batch_size": int(args.quality_batch_size),
                        "quality_rois_this_optimizer_step": int(rois_this_step_actual),
                        "quality_gradient_to_predicted_coordinates_norm": float(
                            qg_log.get("quality_grad_norm_at_candidate", 0.0)
                        ),
                        "candidate_has_grad_fn": bool(
                            qg_log.get("candidate_has_grad_fn", False)
                        ),
                        "first_quality_roi": dict(qg_log),
                    }
                    sanity["quality_gradient_path_present"] = (
                        sanity["candidate_has_grad_fn"]
                        and sanity["quality_gradient_to_predicted_coordinates_norm"] > 0.0
                    )
                    finite_keys = (
                        "L_matched", "L_rank", "L_Q", "L_geo",
                        "ema_q", "ema_geo", "normalized_q", "normalized_geo",
                        "quality_contribution", "geometry_contribution", "aux_total",
                    )
                    sanity["quality_loss_finite"] = all(
                        math.isfinite(float(qg_log[k])) for k in finite_keys
                    )
                    checks = [
                        not sanity["qnet_training"],
                        sanity["qnet_all_requires_grad_false"],
                        sanity["qnet_parameter_grads_none"],
                        sanity["qnet_state_unchanged"],
                        sanity["pf_gradient_finite"],
                        sanity["quality_loss_finite"],
                        sanity["quality_gradient_path_present"],
                        sanity["quality_batch_size"] >= 1,
                        sanity["guidance_weights"]["structure_fidelity"] == 0.0,
                        abs(sanity["guidance_weights"]["overall"] - 0.5) < 1e-8,
                        abs(sum(sanity["guidance_weights"].values()) - 1.0) < 1e-6,
                    ]
                    sanity["passed"] = bool(all(checks))
                    write_json(args.output_dir / "sanity_check.json", sanity)
                    print("SANITY " + json.dumps(sanity, ensure_ascii=False), flush=True)
                    if not sanity["passed"]:
                        raise RuntimeError("First-batch Stage III sanity failed")

                if step == 1 or step % args.print_every == 0:
                    print(
                        f"[{args.variant} epoch={epoch}/{args.epochs} "
                        f"step={step}/{pf_steps_this_epoch}] "
                        f"total={total_value:.6f} pf={pf_value:.6f} "
                        f"q_rois={rois_this_step_actual} "
                        f"q_batches={quality_batches_consumed}/{quality_batches_total} "
                        f"q_consumed={quality_rois_consumed}/{len(train_quality_loader.dataset)} "
                        f"lQ={mean_q:.6f} geo={mean_geo:.6f} "
                        f"nq={qg_log.get('normalized_q', float('nan')):.3f} "
                        f"ng={qg_log.get('normalized_geo', float('nan')):.3f} "
                        f"dw={qg_log.get('difficulty_weight', float('nan')):.3f} "
                        f"qs={qg_log.get('dynamic_quality_share', float('nan')):.3f} "
                        f"gs={qg_log.get('dynamic_geometry_share', float('nan')):.3f} "
                        f"cq={qg_log.get('quality_contribution', float('nan')):.4f} "
                        f"cg={qg_log.get('geometry_contribution', float('nan')):.4f}",
                        flush=True,
                    )
                continue

            # Legacy/non-full-cover branches keep their original behavior.
            if loss_pf is None:
                continue
            if not bool(torch.isfinite(loss_pf).detach().cpu()):
                raise RuntimeError("Non-finite original PointFilter geometry loss")
            quality_loss = loss_pf.new_zeros(())
            geometry_loss = loss_pf.new_zeros(())
            scores = None
            candidate_for_grad = None
            gather_error = None
            quality_patches = None
            if args.variant == "quality_guided_full":
                try:
                    assert quality_iter is not None
                    quality_batch = next(quality_iter)
                except StopIteration:
                    assert train_quality_loader is not None
                    quality_iter = iter(train_quality_loader)
                    quality_batch = next(quality_iter)
                pointfilter.apply(pf_base.freeze_bn_stats)
                quality_loss, geometry_loss = dynamic_online_quality_geo_loss(
                    pointfilter, qnet, quality_batch, device, qg_log,
                )
                quality_centers_seen.update(
                    (shape, sigma, center)
                    for shape, sigma, center in zip(
                        qg_log["dynamic_shapes"],
                        qg_log["dynamic_sigmas"],
                        qg_log["dynamic_center_indices"],
                    )
                )
            elif args.variant != "baseline":
                try:
                    assert quality_iter is not None
                    quality_batch = next(quality_iter)
                except StopIteration:
                    assert train_quality_loader is not None
                    quality_iter = iter(train_quality_loader)
                    quality_batch = next(quality_iter)
                pointfilter.apply(pf_base.freeze_bn_stats)
                if args.variant == "overall_only_multi_patch_online":
                    candidate_for_grad, quality_patches, scores, gather_error = (
                        multi_patch_candidate_and_scores(
                            pointfilter, qnet, quality_batch, device,
                            args.quality_n_anchors, args.quality_pf_chunk,
                        )
                    )
                    gather_error = float(gather_error)
                else:
                    candidate_for_grad, scores = online_candidate_and_scores(
                        pointfilter, qnet, quality_batch, device
                    )
                    quality_patches = None
                    gather_error = None
                quality_loss = quality_objective(scores, args.variant)
            if args.variant == "quality_guided_full":
                ema_q = quality_loss_ema.update(quality_loss)
                ema_geo = geometry_loss_ema.update(geometry_loss)
                normalized_q = quality_loss / ema_q
                normalized_geo = geometry_loss / ema_geo
                q_contribution = float(args.beta_aux) * float(args.quality_fraction) * normalized_q
                g_contribution = float(args.beta_aux) * (1.0 - float(args.quality_fraction)) * normalized_geo
                auxiliary_loss = q_contribution + g_contribution
                total_loss = loss_pf + auxiliary_loss
                qg_log.update({
                    "ema_q": float(ema_q),
                    "ema_geo": float(ema_geo),
                    "normalized_q": float(normalized_q.detach().cpu()),
                    "normalized_geo": float(normalized_geo.detach().cpu()),
                    "quality_contribution": float(q_contribution.detach().cpu()),
                    "geometry_contribution": float(g_contribution.detach().cpu()),
                    "aux_total": float(auxiliary_loss.detach().cpu()),
                })
            else:
                total_loss = loss_pf + float(args.lambda_q) * quality_loss
            optimizer.zero_grad(set_to_none=True)
            if candidate_for_grad is not None:
                candidate_for_grad.retain_grad()
            total_loss.backward()
            pf_grad_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all().detach().cpu())
                for parameter in pointfilter.parameters()
            )
            torch.nn.utils.clip_grad_norm_(pointfilter.parameters(), args.grad_clip)
            optimizer.step()

            if sanity is None:
                coordinate_grad_norm = (
                    float(candidate_for_grad.grad.detach().norm().cpu())
                    if candidate_for_grad is not None and candidate_for_grad.grad is not None
                    else 0.0
                )
                sanity = {
                    "variant": args.variant,
                    "qnet_training": qnet.training,
                    "qnet_all_requires_grad_false": all(not p.requires_grad for p in qnet.parameters()),
                    "qnet_parameter_grads_none": all(p.grad is None for p in qnet.parameters()),
                    "qnet_state_unchanged": pf_base.state_sha256(qnet) == qnet_hash_before,
                    "pf_gradient_finite": pf_grad_finite,
                    "quality_scores_shape": list(scores.shape) if scores is not None else None,
                    "quality_scores_finite": bool(torch.isfinite(scores).all().cpu()) if scores is not None else None,
                    "quality_gradient_to_predicted_coordinates_norm": coordinate_grad_norm,
                    "quality_gradient_path_present": coordinate_grad_norm > 0.0 if scores is not None else None,
                }
                if args.variant == "quality_guided_full":
                    sanity.update({
                        "head_order": list(DIMENSIONS),
                        "head_order_source": "checkpoint config",
                        "guidance_weights": {
                            h: float(reliability[h]) for h in DIMENSIONS},
                        "zeroed_heads": list(LOCKED_ZEROED_HEADS),
                        "lambda_base": float(args.lambda_q),
                        "beta_aux": float(args.beta_aux),
                        "quality_fraction": float(args.quality_fraction),
                        "geometry_fraction": float(1.0 - args.quality_fraction),
                        "direct_geometry_supervision": True,
                        "ema_normalization": True,
                        "loss_mixture": {
                            "beta_matched": LOCKED_BETA_MATCHED,
                            "gamma_rank": LOCKED_GAMMA_RANK,
                        },
                        "rank_margin": LOCKED_RANK_MARGIN,
                        "quality_unit_points": 500,
                        "first_step": dict(qg_log),
                    })
                    # QNet must stay bit-identical and the Q gradient must actually reach P_hat.
                    sanity["quality_gradient_to_predicted_coordinates_norm"] = float(
                        qg_log.get("quality_grad_norm_at_candidate", 0.0)
                    )
                    sanity["candidate_has_grad_fn"] = bool(
                        qg_log.get("candidate_has_grad_fn", False)
                    )
                    sanity["quality_gradient_path_present"] = (
                        sanity["candidate_has_grad_fn"]
                        and sanity["quality_gradient_to_predicted_coordinates_norm"] > 0.0
                    )
                    sanity["quality_loss_finite"] = bool(
                        all(math.isfinite(float(qg_log[k])) for k in
                            ("L_matched", "L_rank", "L_Q", "L_geo",
                             "ema_q", "ema_geo", "normalized_q", "normalized_geo",
                             "quality_contribution", "geometry_contribution", "aux_total"))
                    )
                if args.variant == "overall_only_multi_patch_online":
                    sanity["candidate_subcloud_shape"] = (
                        list(candidate_for_grad.shape) if candidate_for_grad is not None else None
                    )
                    sanity["quality_patches_shape"] = (
                        list(quality_patches.shape) if quality_patches is not None else None
                    )
                    sanity["quality_patches_from_candidate_subcloud_max_error"] = gather_error
                    sanity["pf_chunk"] = args.quality_pf_chunk
                checks = [
                    not sanity["qnet_training"],
                    sanity["qnet_all_requires_grad_false"],
                    sanity["qnet_parameter_grads_none"],
                    sanity["qnet_state_unchanged"],
                    sanity["pf_gradient_finite"],
                ]
                if scores is not None:
                    checks.extend([
                        sanity["quality_scores_shape"][-1] == 5,
                        sanity["quality_scores_finite"],
                        sanity["quality_gradient_path_present"],
                    ])
                if args.variant == "overall_only_multi_patch_online":
                    checks.extend([
                        sanity["candidate_subcloud_shape"][0] == args.quality_num_centers,
                        sanity["quality_patches_shape"][0] == args.quality_n_anchors,
                        sanity["quality_patches_shape"][1] == args.quality_points_per_patch,
                        sanity["quality_patches_from_candidate_subcloud_max_error"] == 0.0,
                    ])
                if args.variant == "quality_guided_full":
                    checks.extend([
                        sanity["quality_loss_finite"],
                        sanity["quality_gradient_path_present"],
                        # The whole design rests on this: QNet must not train.
                        sanity["qnet_state_unchanged"],
                        sanity["zeroed_heads"] == ["structure_fidelity"],
                        sanity["guidance_weights"]["structure_fidelity"] == 0.0,
                        abs(sanity["guidance_weights"]["overall"] - 0.5) < 1e-8,
                        abs(sanity["guidance_weights"]["cleanliness"] - (1.0/6.0)) < 1e-8,
                        abs(sanity["guidance_weights"]["surface_fidelity"] - (1.0/6.0)) < 1e-8,
                        abs(sanity["guidance_weights"]["sampling_regularity"] - (1.0/6.0)) < 1e-8,
                        abs(sum(sanity["guidance_weights"].values()) - 1.0) < 1e-6,
                        # Scores must be five-dimensional, finite, and in [0,1].
                        len(sanity["first_step"]["matched"]["q_cand"]) == len(DIMENSIONS),
                        all(0.0 <= v <= 1.0 for v in sanity["first_step"]["matched"]["q_cand"]),
                        all(0.0 <= v <= 1.0 for v in sanity["first_step"]["matched"]["q_clean"]),
                        all(0.0 <= v <= 1.0 for v in sanity["first_step"]["matched"]["q_noisy"]),
                    ])
                sanity["passed"] = bool(all(checks))
                write_json(args.output_dir / "sanity_check.json", sanity)
                print("SANITY " + json.dumps(sanity, ensure_ascii=False), flush=True)
                if not sanity["passed"]:
                    raise RuntimeError("First-batch Stage III sanity failed")

            epoch_values["total"].append(float(total_loss.detach().cpu()))
            epoch_values["pf"].append(float(loss_pf.detach().cpu()))
            epoch_values["quality"].append(float(quality_loss.detach().cpu()))
            if args.variant == "quality_guided_full":
                for key in qg_epoch:
                    if key in qg_log:
                        qg_epoch[key].append(float(qg_log[key]))
            if scores is not None:
                epoch_values["overall"].append(float(scores[:, 4].mean().detach().cpu()))
                epoch_values["detail"].append(float(scores[:, 1].mean().detach().cpu()))
            if step == 1 or step % args.print_every == 0:
                quality_text = ""
                if scores is not None:
                    quality_text = (
                        f" qloss={epoch_values['quality'][-1]:.6f}"
                        f" qO={epoch_values['overall'][-1]:.4f}"
                        f" qD={epoch_values['detail'][-1]:.4f}"
                    )
                elif args.variant == "quality_guided_full":
                    quality_text = (
                        f" lQ={qg_log['L_Q']:.6f} m={qg_log['L_matched']:.6f}"
                        f" r={qg_log['L_rank']:.6f}"
                        f" geo={qg_log['L_geo']:.6f}"
                        f" nq={qg_log['normalized_q']:.3f} ng={qg_log['normalized_geo']:.3f}"
                        f" cq={qg_log['quality_contribution']:.4f}"
                        f" cg={qg_log['geometry_contribution']:.4f}"
                    )
                print(
                    f"[{args.variant} lambda={args.lambda_q:g} epoch={epoch}/{args.epochs} "
                    f"step={step}/{len(pf_loader)}] total={epoch_values['total'][-1]:.6f} "
                    f"pf={epoch_values['pf'][-1]:.6f}{quality_text}",
                    flush=True,
                )

        val_aggregate, _ = validate(pointfilter, qnet, val_loader, args, device)
        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train": {
                key: (float(np.mean(values)) if values else None)
                for key, values in epoch_values.items()
            },
            "validation": val_aggregate,
        }
        if args.variant == "quality_guided_full":
            record["quality_guided_full"] = {
                key: (float(np.mean(values)) if values else None)
                for key, values in qg_epoch.items()
            }
            record["quality_guided_full"]["guidance_weights"] = {
                h: float(reliability[h]) for h in DIMENSIONS}
            record["quality_guided_full"]["loss_ema"] = {
                "quality": quality_loss_ema.state_dict(),
                "geometry": geometry_loss_ema.state_dict(),
                "difficulty": difficulty_ema.state_dict(),
            }
            parent_pairs = list(train_quality_loader.dataset.pairs)
            coverage_by_parent = {
                f"{shape}|{sigma}": (
                    len(quality_points_seen.get((shape, sigma), set())) / 50000.0
                )
                for shape, sigma in parent_pairs
            }
            schedule_complete = (
                quality_batches_consumed == quality_batches_total
                and quality_rois_consumed == len(train_quality_loader.dataset)
                and len(quality_centers_seen) == len(train_quality_loader.dataset)
            )
            planned_coverage = {
                f"{shape}|{sigma}": float(value)
                for (shape, sigma), value in train_quality_loader.dataset._planned_coverage_by_parent.items()
            }
            planned_centers = {
                f"{shape}|{sigma}": int(value)
                for (shape, sigma), value in train_quality_loader.dataset._planned_centers_by_parent.items()
            }
            actual_values = list(coverage_by_parent.values())
            centre_values = list(planned_centers.values())
            actual_mean = float(np.mean(actual_values)) if actual_values else 0.0
            actual_min = float(np.min(actual_values)) if actual_values else 0.0
            actual_target_met = (
                actual_min + 1e-12 >= float(args.quality_coverage_target)
                if args.quality_coverage_target > 0 else True
            )
            record["quality_guided_full"]["spatial_coverage"] = {
                "strategy": "coverage_aware_fps_exact_500nn_full_schedule",
                "coverage_target": args.quality_coverage_target,
                "minimum_centers_per_parent_cloud": args.quality_centers_per_cloud,
                "maximum_centers_per_parent_cloud": args.quality_max_centers_per_cloud,
                "planned_centers_by_parent": planned_centers,
                "planned_centers_min": int(min(centre_values)) if centre_values else 0,
                "planned_centers_mean": float(np.mean(centre_values)) if centre_values else 0.0,
                "planned_centers_max": int(max(centre_values)) if centre_values else 0,
                "quality_batch_size": args.quality_batch_size,
                "expected_quality_batches": quality_batches_total,
                "consumed_quality_batches": quality_batches_consumed,
                "expected_quality_rois": len(train_quality_loader.dataset),
                "consumed_quality_rois": quality_rois_consumed,
                "expected_unique_centers": len(train_quality_loader.dataset),
                "observed_unique_centers": len(quality_centers_seen),
                "schedule_complete": schedule_complete,
                "planned_unique_point_coverage_by_parent": planned_coverage,
                "actual_unique_point_coverage_by_parent": coverage_by_parent,
                "actual_unique_point_coverage_mean": actual_mean,
                "actual_unique_point_coverage_min": actual_min,
                "actual_coverage_target_met_for_every_parent": actual_target_met,
            }
            if not schedule_complete:
                raise RuntimeError(
                    "Coverage-aware quality schedule was not completely consumed: "
                    f"{quality_batches_consumed}/{quality_batches_total} batches, "
                    f"{quality_rois_consumed}/{len(train_quality_loader.dataset)} ROIs, "
                    f"{len(quality_centers_seen)}/{len(train_quality_loader.dataset)} centers"
                )
            if not actual_target_met:
                worst_key = min(coverage_by_parent, key=coverage_by_parent.get)
                raise RuntimeError(
                    "Actual QNet ROI coverage fell below the requested target after invalid-PCA "
                    "replacement: "
                    f"worst={worst_key} coverage={coverage_by_parent[worst_key]:.6f} "
                    f"target={args.quality_coverage_target:.6f}. Increase "
                    "--quality-max-centers-per-cloud or inspect invalid patches."
                )
        # -------------------------------------------------------------
        # Early stopping monitors the SAME quantity used for checkpoint
        # selection: Validation Chamfer Distance.  Training losses are
        # intentionally not used because their dynamic ROI/EMA scales
        # change during training and are not directly comparable across
        # epochs.
        # -------------------------------------------------------------
        current_cd = float(val_aggregate["chamfer_distance"])
        previous_best_cd = float(best_cd)
        improved = current_cd < (previous_best_cd - float(args.early_stop_min_delta))
        if improved:
            best_cd = current_cd
            best_epoch = epoch
            early_stop_bad_epochs = 0
        else:
            early_stop_bad_epochs += 1

        record["early_stopping"] = {
            "monitor": "validation.chamfer_distance",
            "current": current_cd,
            "historical_best": float(best_cd),
            "improved": bool(improved),
            "bad_epochs": int(early_stop_bad_epochs),
            "patience": int(args.early_stop_patience),
            "min_delta": float(args.early_stop_min_delta),
        }
        history.append(record)
        write_json(args.output_dir / "history.json", history)
        print("EPOCH " + json.dumps(record, ensure_ascii=False), flush=True)

        early_stop_state = {
            "monitor": "validation.chamfer_distance",
            "best_cd": float(best_cd),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "bad_epochs": int(early_stop_bad_epochs),
            "patience": int(args.early_stop_patience),
            "min_delta": float(args.early_stop_min_delta),
        }
        last_checkpoint = {
            "epoch": epoch,
            "state_dict": pointfilter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "history": history,
            "sanity": sanity,
            "validation": val_aggregate,
            "qnet_state_sha256": qnet_hash_before,
            "quality_loss_ema": quality_loss_ema.state_dict(),
            "geometry_loss_ema": geometry_loss_ema.state_dict(),
            "difficulty_ema": difficulty_ema.state_dict(),
            "early_stop_state": early_stop_state,
        }
        torch.save(last_checkpoint, args.output_dir / "last_state.pth")

        if improved:
            checkpoint = {
                "epoch": epoch,
                "state_dict": pointfilter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "history": history,
                "sanity": sanity,
                "validation": val_aggregate,
                "qnet_state_sha256": qnet_hash_before,
                "quality_loss_ema": quality_loss_ema.state_dict(),
                "geometry_loss_ema": geometry_loss_ema.state_dict(),
                "difficulty_ema": difficulty_ema.state_dict(),
                "early_stop_state": early_stop_state,
            }
            torch.save(checkpoint, args.output_dir / "best_validation.pth")
            validate(
                pointfilter,
                qnet,
                val_loader,
                args,
                device,
                prediction_path=args.output_dir / "best_validation_predictions.npz",
                record_path=args.output_dir / "best_validation_records.jsonl",
            )
            print(
                f"BEST epoch={epoch} val_cd={best_cd:.8f} "
                f"patience_reset=0/{args.early_stop_patience}",
                flush=True,
            )
        else:
            print(
                f"NO_IMPROVEMENT epoch={epoch} val_cd={current_cd:.8f} "
                f"best_cd={best_cd:.8f}@epoch{best_epoch} "
                f"patience={early_stop_bad_epochs}/{args.early_stop_patience}",
                flush=True,
            )

        if (
            args.early_stop_patience > 0
            and early_stop_bad_epochs >= args.early_stop_patience
        ):
            early_stopped = True
            stopped_epoch = epoch
            print(
                f"EARLY_STOP epoch={epoch}: Validation Chamfer Distance has not "
                f"beaten best_cd={best_cd:.8f} (epoch {best_epoch}) for "
                f"{early_stop_bad_epochs} consecutive epochs.",
                flush=True,
            )
            break

    checkpoint = torch.load(args.output_dir / "best_validation.pth", map_location=device, weights_only=False)
    pointfilter.load_state_dict(checkpoint["state_dict"], strict=True)
    final_validation, _ = validate(
        pointfilter,
        qnet,
        val_loader,
        args,
        device,
        prediction_path=args.output_dir / "best_validation_predictions.npz",
        record_path=args.output_dir / "best_validation_records.jsonl",
    )
    qnet_hash_after = pf_base.state_sha256(qnet)
    qnet_file_hash_after = pf_base.sha256(args.qnet_checkpoint)
    if qnet_hash_after != qnet_hash_before or qnet_file_hash_after != qnet_file_hash_before:
        raise RuntimeError("Frozen QNet changed during Stage III")
    summary = {
        "status": (
            "train_test_as_validation_leakage_complete"
            if args.validation_uses_test
            else "train_validation_complete_effective_test_not_accessed"
        ),
        "variant": args.variant,
        "lambda_q": args.lambda_q,
        "best_epoch": best_epoch,
        "best_validation": final_validation,
        "best_checkpoint": str(args.output_dir / "best_validation.pth"),
        "validation_predictions": str(args.output_dir / "best_validation_predictions.npz"),
        "validation_records": str(args.output_dir / "best_validation_records.jsonl"),
        "qnet_state_sha256_before": qnet_hash_before,
        "qnet_state_sha256_after": qnet_hash_after,
        "qnet_checkpoint_sha256_before": qnet_file_hash_before,
        "qnet_checkpoint_sha256_after": qnet_file_hash_after,
        "qnet_unchanged": qnet_hash_after == qnet_hash_before and qnet_file_hash_after == qnet_file_hash_before,
        "test_accessed": bool(args.validation_uses_test),
        "test_access_reason": (
            "intentional Test-as-Validation checkpoint-selection ablation"
            if args.validation_uses_test else None
        ),
        "split_override": split_override,
        "elapsed_seconds": time.time() - started,
        "early_stopping": {
            "monitor": "validation.chamfer_distance",
            "patience": int(args.early_stop_patience),
            "min_delta": float(args.early_stop_min_delta),
            "early_stopped": bool(early_stopped),
            "stopped_epoch": int(stopped_epoch) if stopped_epoch is not None else None,
            "bad_epochs_at_end": int(early_stop_bad_epochs),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_cd": float(best_cd),
        },
        "sanity": sanity,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
