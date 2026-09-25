#!/usr/bin/env python3
"""Retrain Patch QNet with intentional cup/dino/gargoyle Train-Test overlap.

Supervision is intentionally limited to:

  L = L_real + 0.2 * L_cf_cleanliness + 0.2 * L_cf_structure

The network input is candidate-only ``points_raw``. Every patch is translated
by its own centroid and receives no PCA alignment, scale normalization, clean
reference, method, sigma, structure metadata, image, or Teacher reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

from edgeconv_quality_network import EdgeConvQualityQueryNet5D


ROOT = Path("/data/zhangzy/zzyy/PUNet/Gaussion")
EXPERIMENT_ROOT = ROOT / "patch_quality_network/test_as_val_qnet_retrain_seed2026"
DEFAULT_DREAL = EXPERIMENT_ROOT / "labels/d_real_mvp1032_test_as_val_seed2026.jsonl"
DEFAULT_CF_CLEAN = ROOT / "manifests/d_cf_cleanliness_ranking_train_v1.jsonl"
DEFAULT_CF_STRUCTURE = ROOT / "manifests/d_cf_structure_ranking_train_v1.jsonl"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "run"
DEFAULT_INIT = ROOT / "patch_quality_network/runs/stage1_dcf_specialization_v1_seed2026/best_stage1_dcf_specialization_v1.pth"
OVERLAP_SHAPES = ("cup", "dino", "gargoyle")

DIMENSIONS = (
    "cleanliness",
    "structure_fidelity",
    "surface_fidelity",
    "sampling_regularity",
    "overall",
)
CLEANLINESS_IDX = 0
STRUCTURE_IDX = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_candidate_only(path: str) -> torch.Tensor:
    """Load the only allowed Student input and apply translation only."""
    with np.load(path) as data:
        if "points_raw" not in data:
            raise KeyError(f"points_raw missing: {path}")
        points = np.asarray(data["points_raw"], dtype=np.float32).copy()
    if points.shape != (500, 3):
        raise ValueError(f"Expected points_raw [500,3], got {points.shape}: {path}")
    if not np.isfinite(points).all():
        raise ValueError(f"NaN/Inf in points_raw: {path}")
    points -= points.mean(axis=0, keepdims=True)
    return torch.from_numpy(points)


class DRealDataset(Dataset):
    def __init__(self, manifest: Path, split: str, preload: bool = True,
                 shapes: set[str] | None = None):
        all_rows = read_jsonl(manifest)
        self.rows = [row for row in all_rows if row["split"] == split
                     and (shapes is None or row["shape"] in shapes)]
        if not self.rows:
            raise ValueError(f"No D_real rows for split={split}")
        self.preloaded = (
            [load_candidate_only(row["candidate_patch_path"]) for row in self.rows]
            if preload
            else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        points = (
            self.preloaded[index].clone()
            if self.preloaded is not None
            else load_candidate_only(row["candidate_patch_path"])
        )
        scores = [row[name] for name in DIMENSIONS]
        target = torch.tensor(
            [0.0 if value is None else (float(value) - 1.0) / 4.0 for value in scores],
            dtype=torch.float32,
        )
        mask = torch.ones(len(DIMENSIONS), dtype=torch.bool)
        mask[STRUCTURE_IDX] = bool(row["structure_mask"])
        return {
            "points": points,
            "target": target,
            "mask": mask,
            "sample_id": row["sample_id"],
            "teacher_id": row["teacher_id"],
            "shape": row["shape"],
        }


class RankingDataset(Dataset):
    def __init__(self, manifest: Path, preload: bool = True):
        self.rows = read_jsonl(manifest)
        if not self.rows:
            raise ValueError(f"No ranking rows in {manifest}")
        self.preloaded = None
        if preload:
            cache: dict[str, torch.Tensor] = {}
            for row in self.rows:
                for field in ("best_patch", "middle_patch", "worst_patch"):
                    path = row[field]
                    if path not in cache:
                        cache[path] = load_candidate_only(path)
            self.preloaded = cache

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        paths = [row["best_patch"], row["middle_patch"], row["worst_patch"]]
        if self.preloaded is None:
            points = torch.stack([load_candidate_only(path) for path in paths])
        else:
            points = torch.stack([self.preloaded[path].clone() for path in paths])
        return {
            "points": points,
            "family_id": row["family_id"],
            "shape": row["shape"],
        }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def predict_01(model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(model(points))


def real_absolute_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parts: dict[str, torch.Tensor] = {}
    total = prediction.new_zeros(())
    for dim, name in enumerate(DIMENSIONS):
        valid = mask[:, dim]
        # A shuffled mini-batch may contain only smooth/random samples. In that
        # case Structure contributes an exact differentiable zero for this step.
        value = (
            F.smooth_l1_loss(prediction[valid, dim], target[valid, dim])
            if valid.any()
            else prediction[:, dim].sum() * 0.0
        )
        parts[name] = value
        total = total + value
    return total, parts


def chain_rank_loss(scores: torch.Tensor, margin: float) -> torch.Tensor:
    """scores are ordered [best, middle, worst]."""
    return (
        F.relu(float(margin) - scores[:, 0] + scores[:, 1])
        + F.relu(float(margin) - scores[:, 1] + scores[:, 2])
    ).mean()


def safe_corr(function, target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2 or np.std(target) == 0 or np.std(prediction) == 0:
        return 0.0
    result = function(target, prediction)
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return 0.0 if not np.isfinite(value) else float(value)


def regression_metrics(target_01: np.ndarray, pred_01: np.ndarray) -> dict[str, float | int]:
    target = target_01 * 4.0 + 1.0
    prediction = pred_01 * 4.0 + 1.0
    error = prediction - target
    return {
        "count": int(len(target)),
        "plcc": safe_corr(pearsonr, target, prediction),
        "srcc": safe_corr(spearmanr, target, prediction),
        "krcc": safe_corr(kendalltau, target, prediction),
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
    }


@torch.no_grad()
def evaluate_real(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    loss_values = []
    for batch in loader:
        points = batch["points"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        prediction = predict_01(model, points)
        loss, _ = real_absolute_loss(prediction, target, mask)
        loss_values.append(float(loss.cpu()))
        pred_np = prediction.cpu().numpy()
        target_np = target.cpu().numpy()
        mask_np = mask.cpu().numpy()
        predictions.append(pred_np)
        targets.append(target_np)
        masks.append(mask_np)
        for i in range(len(batch["sample_id"])):
            records.append(
                {
                    "sample_id": batch["sample_id"][i],
                    "teacher_id": batch["teacher_id"][i],
                    "shape": batch["shape"][i],
                    "target_1_5": {
                        name: (float(target_np[i, j] * 4.0 + 1.0) if mask_np[i, j] else None)
                        for j, name in enumerate(DIMENSIONS)
                    },
                    "prediction_1_5": {
                        name: float(pred_np[i, j] * 4.0 + 1.0)
                        for j, name in enumerate(DIMENSIONS)
                    },
                    "structure_mask": int(mask_np[i, STRUCTURE_IDX]),
                }
            )

    pred = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    mask = np.concatenate(masks, axis=0)
    per_dimension = {}
    for dim, name in enumerate(DIMENSIONS):
        valid = mask[:, dim]
        per_dimension[name] = regression_metrics(target[valid, dim], pred[valid, dim])
    return (
        {
            "real_loss_normalized": float(np.mean(loss_values)),
            "overall": per_dimension["overall"],
            "per_dimension": per_dimension,
            "samples": int(len(pred)),
        },
        records,
    )


@torch.no_grad()
def evaluate_ranking(
    model: torch.nn.Module,
    loader: DataLoader,
    query_index: int,
    device: torch.device,
    margin: float,
) -> dict[str, float | int]:
    model.eval()
    families = 0
    chain_correct = 0
    adjacent_correct = 0
    margin_correct = 0
    losses = []
    for batch in loader:
        points = batch["points"].to(device, non_blocking=True)
        b, levels, n, c = points.shape
        scores = predict_01(model, points.reshape(b * levels, n, c)).reshape(b, levels, -1)
        scores = scores[:, :, query_index]
        losses.append(float(chain_rank_loss(scores, margin).cpu()))
        first = scores[:, 0] > scores[:, 1]
        second = scores[:, 1] > scores[:, 2]
        first_margin = scores[:, 0] - scores[:, 1] >= margin
        second_margin = scores[:, 1] - scores[:, 2] >= margin
        families += b
        chain_correct += int((first & second).sum().cpu())
        adjacent_correct += int(first.sum().cpu() + second.sum().cpu())
        margin_correct += int((first_margin & second_margin).sum().cpu())
    return {
        "families": int(families),
        "chain_accuracy": float(chain_correct / families),
        "adjacent_pair_accuracy": float(adjacent_correct / (2 * families)),
        "margin_chain_accuracy": float(margin_correct / families),
        "rank_loss": float(np.mean(losses)),
    }


def write_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class TrainConfig:
    seed: int
    epochs: int
    batch_size: int
    rank_batch_size: int
    workers: int
    learning_rate: float
    weight_decay: float
    rank_margin: float
    cf_clean_weight: float
    cf_structure_weight: float
    patience: int
    k: int
    query_heads: int
    graph_mode: str
    query_interaction: bool
    amp: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dreal-manifest", type=Path, default=DEFAULT_DREAL)
    parser.add_argument("--cf-clean-manifest", type=Path, default=DEFAULT_CF_CLEAN)
    parser.add_argument("--cf-structure-manifest", type=Path, default=DEFAULT_CF_STRUCTURE)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=DEFAULT_INIT,
        help="Stage-I checkpoint used to initialize the overlap3 QNet.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rank-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--cf-clean-weight", type=float, default=0.2)
    parser.add_argument("--cf-structure-weight", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument(
        "--graph-mode",
        choices=("fixed", "fixed_fixed_dynamic", "dynamic"),
        default="dynamic",
    )
    parser.add_argument("--no-query-interaction", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overlap-shapes", default=",".join(OVERLAP_SHAPES))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        seed=args.seed,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
        rank_batch_size=args.rank_batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rank_margin=args.rank_margin,
        cf_clean_weight=args.cf_clean_weight,
        cf_structure_weight=args.cf_structure_weight,
        patience=args.patience,
        k=args.k,
        query_heads=args.query_heads,
        graph_mode=args.graph_mode,
        query_interaction=not args.no_query_interaction,
        amp=(not args.no_amp and device.type == "cuda"),
    )
    provenance = {
        "schema_version": "patch_qnet_test_as_validation_leakage_ablation_v1",
        "evaluation_semantics": (
            "Original 7-shape Test is intentionally used as Validation for checkpoint selection; "
            "original 6-shape Validation is the untouched holdout Test"
        ),
        "quality_dimensions": list(DIMENSIONS),
        "score_training_scale": "[0,1] via (score_1_5 - 1) / 4",
        "score_reporting_scale": "[1,5]",
        "student_input": "candidate-only points_raw",
        "preprocessing": "points_raw - mean(points_raw, axis=0)",
        "forbidden_preprocessing": ["candidate PCA", "unit-sphere normalization", "clean-reference alignment"],
        "loss": (
            f"L_real + {config.cf_clean_weight:g} L_cf_cleanliness + "
            f"{config.cf_structure_weight:g} L_cf_structure"
        ),
        "real_loss": "sum of five SmoothL1 factor means; Structure uses structure_mask",
        "cf_clean_order": "clean > under_denoise > noisy",
        "cf_structure_order": "PGD > smooth_mild > smooth_strong",
        "manifests": {
            "d_real": {"path": str(args.dreal_manifest), "sha256": sha256(args.dreal_manifest)},
            "d_cf_cleanliness": {"path": str(args.cf_clean_manifest), "sha256": sha256(args.cf_clean_manifest)},
            "d_cf_structure": {"path": str(args.cf_structure_manifest), "sha256": sha256(args.cf_structure_manifest)},
        },
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    if args.init_checkpoint is not None:
        provenance["initialization"] = {
            "checkpoint": str(args.init_checkpoint),
            "sha256": sha256(args.init_checkpoint),
            "role": "Stage-I D_cf factor-specialization initialization",
        }
    write_json(args.output_dir / "config.json", provenance)

    overlap_shapes = {part.strip() for part in args.overlap_shapes.split(",") if part.strip()}
    train_set = DRealDataset(args.dreal_manifest, "train")
    val_set = DRealDataset(args.dreal_manifest, "val")
    test_set = DRealDataset(args.dreal_manifest, "test")
    # Leakage ablation: the current val split is the original 7-shape Test.
    # Decompose that selection set into Train-overlap and unseen-shape subsets.
    val_seen_set = DRealDataset(args.dreal_manifest, "val", shapes=overlap_shapes)
    val_unseen_shapes = {row["shape"] for row in val_set.rows} - overlap_shapes
    val_unseen_set = DRealDataset(args.dreal_manifest, "val", shapes=val_unseen_shapes)
    clean_rank_set = RankingDataset(args.cf_clean_manifest)
    structure_rank_set = RankingDataset(args.cf_structure_manifest)
    train_loader = make_loader(train_set, config.batch_size, True, config.workers, config.seed)
    val_loader = make_loader(val_set, config.batch_size * 2, False, config.workers, config.seed + 1)
    test_loader = make_loader(test_set, config.batch_size * 2, False, config.workers, config.seed + 2)
    val_seen_loader = make_loader(val_seen_set, config.batch_size * 2, False, config.workers, config.seed + 7)
    val_unseen_loader = make_loader(val_unseen_set, config.batch_size * 2, False, config.workers, config.seed + 8)
    clean_rank_loader = make_loader(clean_rank_set, config.rank_batch_size, True, config.workers, config.seed + 3)
    structure_rank_loader = make_loader(structure_rank_set, config.rank_batch_size, True, config.workers, config.seed + 4)
    clean_rank_eval_loader = make_loader(clean_rank_set, config.rank_batch_size * 2, False, config.workers, config.seed + 5)
    structure_rank_eval_loader = make_loader(structure_rank_set, config.rank_batch_size * 2, False, config.workers, config.seed + 6)

    model = EdgeConvQualityQueryNet5D(
        k=config.k,
        output_dim=len(DIMENSIONS),
        query_heads=config.query_heads,
        graph_mode=config.graph_mode,
        query_interaction=config.query_interaction,
        encoder_type="edgeconv",
        quality_mode="direct",
    ).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state = initial["model"] if isinstance(initial, dict) and "model" in initial else initial
        model.load_state_dict(state, strict=True)
        print(f"Initialized from Stage-I checkpoint: {args.init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)

    history = []
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = args.output_dir / "best_patch_qnet_test_as_val_v1.pth"
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        clean_iter = iter(clean_rank_loader)
        structure_iter = iter(structure_rank_loader)
        totals = {"total": 0.0, "real": 0.0, "cf_clean": 0.0, "cf_structure": 0.0}
        steps = 0
        for real_batch in train_loader:
            clean_batch, clean_iter = next_or_restart(clean_iter, clean_rank_loader)
            structure_batch, structure_iter = next_or_restart(structure_iter, structure_rank_loader)
            real_points = real_batch["points"].to(device, non_blocking=True)
            real_target = real_batch["target"].to(device, non_blocking=True)
            real_mask = real_batch["mask"].to(device, non_blocking=True)
            clean_points = clean_batch["points"].to(device, non_blocking=True)
            structure_points = structure_batch["points"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=config.amp):
                real_pred = predict_01(model, real_points)
                real_loss, _ = real_absolute_loss(real_pred, real_target, real_mask)

                cb, cl, cn, cc = clean_points.shape
                clean_scores = predict_01(model, clean_points.reshape(cb * cl, cn, cc))
                clean_scores = clean_scores.reshape(cb, cl, -1)[:, :, CLEANLINESS_IDX]
                cf_clean_loss = chain_rank_loss(clean_scores, config.rank_margin)

                sb, sl, sn, sc = structure_points.shape
                structure_scores = predict_01(model, structure_points.reshape(sb * sl, sn, sc))
                structure_scores = structure_scores.reshape(sb, sl, -1)[:, :, STRUCTURE_IDX]
                cf_structure_loss = chain_rank_loss(structure_scores, config.rank_margin)

                total_loss = (
                    real_loss
                    + config.cf_clean_weight * cf_clean_loss
                    + config.cf_structure_weight * cf_structure_loss
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            totals["total"] += float(total_loss.detach().cpu())
            totals["real"] += float(real_loss.detach().cpu())
            totals["cf_clean"] += float(cf_clean_loss.detach().cpu())
            totals["cf_structure"] += float(cf_structure_loss.detach().cpu())
            steps += 1
            if args.smoke_test and steps >= 2:
                break

        val_metrics, _ = evaluate_real(model, val_loader, device)
        val_loss = float(val_metrics["real_loss_normalized"])
        scheduler.step(val_loss)
        improved = val_loss < best_val - 1e-7
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "provenance": provenance,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_total": totals["total"] / max(steps, 1),
            "train_real": totals["real"] / max(steps, 1),
            "train_cf_clean": totals["cf_clean"] / max(steps, 1),
            "train_cf_structure": totals["cf_structure"] / max(steps, 1),
            "val_real_loss": val_loss,
            "val_overall_plcc": val_metrics["overall"]["plcc"],
            "val_overall_srcc": val_metrics["overall"]["srcc"],
        }
        history.append(row)
        write_json(args.output_dir / "history.json", history)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if not args.smoke_test and epochs_without_improvement >= config.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    val_metrics, val_predictions = evaluate_real(model, val_loader, device)
    test_metrics, test_predictions = evaluate_real(model, test_loader, device)
    val_seen_metrics, val_seen_predictions = evaluate_real(model, val_seen_loader, device)
    val_unseen_metrics, val_unseen_predictions = evaluate_real(model, val_unseen_loader, device)
    clean_rank_metrics = evaluate_ranking(
        model, clean_rank_eval_loader, CLEANLINESS_IDX, device, config.rank_margin
    )
    structure_rank_metrics = evaluate_ranking(
        model, structure_rank_eval_loader, STRUCTURE_IDX, device, config.rank_margin
    )
    write_predictions(args.output_dir / "val_predictions.jsonl", val_predictions)
    write_predictions(args.output_dir / "test_predictions.jsonl", test_predictions)
    write_predictions(args.output_dir / "selection_val_seen_overlap_predictions.jsonl", val_seen_predictions)
    write_predictions(args.output_dir / "selection_val_unseen_predictions.jsonl", val_unseen_predictions)

    summary = {
        "status": "smoke_test_complete" if args.smoke_test else "training_complete",
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "elapsed_seconds": float(time.time() - started),
        "checkpoint": str(best_path),
        "d_real_selection_val_original_test": val_metrics,
        "d_real_holdout_test_original_val": test_metrics,
        "d_real_selection_val_seen_overlap": val_seen_metrics,
        "d_real_selection_val_unseen": val_unseen_metrics,
        "d_cf_cleanliness_train": clean_rank_metrics,
        "d_cf_structure_train": structure_rank_metrics,
        "data_counts": {
            "d_real_train": len(train_set),
            "d_real_selection_val_original_test": len(val_set),
            "d_real_holdout_test_original_val": len(test_set),
            "d_real_selection_val_seen_overlap": len(val_seen_set),
            "d_real_selection_val_unseen": len(val_unseen_set),
            "d_cf_cleanliness_train_families": len(clean_rank_set),
            "d_cf_structure_train_families": len(structure_rank_set),
        },
        "split_semantics": {
            "selection_val_seen_overlap_shapes": sorted(overlap_shapes),
            "selection_val_unseen_shapes": sorted(val_unseen_shapes),
            "holdout_test_shapes": sorted({row["shape"] for row in test_set.rows}),
            "warning": "Original Test was used for checkpoint selection; its metrics are test-tuned and biased.",
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
