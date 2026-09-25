#!/usr/bin/env python3
"""PointFilter + frozen QNet Structure local-weight guidance v1.

The only A/B difference is the per-point geometry weight:

  uniform: w_i = 1
  guided:  w_i = 1 + percentile_rank(Structure cross-attention_i)

QNet never receives gradients. Geometry loss coefficients and total normalized
strength are identical across variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path("/data/zhangzy/zzyy")
PF_ROOT = ROOT / "Pointfilter-master"
sys.path.insert(0, str(PF_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Pointfilter_DataLoader import (  # noqa: E402
    PointcloudPatchDataset,
    RandomPointcloudPatchSampler,
    my_collate,
)
from Pointfilter_Network_Architecture import pointfilternet  # noqa: E402
from Pointfilter_Utils import compute_bilateral_loss_with_repulsion  # noqa: E402
from edgeconv_quality_network import EdgeConvQualityQueryNet5D  # noqa: E402


GAUSSIAN_ROOT = ROOT / "PUNet/Gaussion"
DEFAULT_QNET = (
    GAUSSIAN_ROOT
    / "patch_quality_network/runs/stage2_real_calibration_v1_seed2026"
    / "best_patch_qnet_mvp_v1.pth"
)
DEFAULT_PF_CKPT = PF_ROOT / "Summary/pre_train_model/model_full_ae.pth"
DEFAULT_PF_TRAIN = ROOT / "pointcleannet-master/data/pointCleanNetDataset_pf_standard"
DEFAULT_SHARED = GAUSSIAN_ROOT / "manifests/shared_patch_manifest.jsonl"
DEFAULT_SPLIT = GAUSSIAN_ROOT / "manifests/shape_split_seed2026.json"
DEFAULT_OUTPUT_ROOT = ROOT / "PointFilter_structure_guidance_v1"
STRUCTURE_QUERY_INDEX = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
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
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def freeze_bn_stats(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def scheduled_lr(epoch_zero_based: int, base_lr: float) -> float:
    lr = float(base_lr)
    if epoch_zero_based > 36:
        lr *= 0.5e-3
    elif epoch_zero_based > 32:
        lr *= 1e-3
    elif epoch_zero_based > 24:
        lr *= 1e-2
    elif epoch_zero_based > 16:
        lr *= 1e-1
    return lr


class SharedNoisyCleanGeometryDataset(Dataset):
    """Train-only shared noisy/clean pairs in clean-reference display space."""

    def __init__(self, shared_manifest: Path, split_path: Path, preload: bool = True):
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_shapes = set(split["train"])
        rows = read_jsonl(shared_manifest)
        clean = {
            row["anchor_key"]: row
            for row in rows
            if row["family"] == "clean" and row["shape"] in train_shapes
        }
        noisy = [
            row
            for row in rows
            if row["family"] == "noisy" and row["shape"] in train_shapes
        ]
        noisy.sort(key=lambda row: (row["shape"], str(row["sigma"]), row["anchor_name"]))
        self.rows = []
        for row in noisy:
            clean_row = clean.get(row["anchor_key"])
            if clean_row is None:
                raise KeyError(f"Missing clean anchor: {row['anchor_key']}")
            self.rows.append(
                {
                    "pair_id": row["patch_id"],
                    "shape": row["shape"],
                    "sigma": row["sigma"],
                    "anchor_key": row["anchor_key"],
                    "noisy_path": row["patch_path"],
                    "clean_path": clean_row["patch_path"],
                }
            )
        if len(self.rows) != 28 * 16 * 5:
            raise ValueError(f"Expected 2240 train noisy/clean pairs, got {len(self.rows)}")
        self.cache = [self._load(row) for row in self.rows] if preload else None

    @staticmethod
    def _load(row: dict[str, Any]) -> dict[str, torch.Tensor]:
        with np.load(row["noisy_path"]) as noisy_data:
            noisy_ref = np.asarray(noisy_data["points_ref"], dtype=np.float32).copy()
            frame = np.asarray(noisy_data["clean_pca_frame"], dtype=np.float32).copy()
            radius = np.asarray(noisy_data["clean_reference_radius"], dtype=np.float32).reshape(1)
        with np.load(row["clean_path"]) as clean_data:
            clean_ref = np.asarray(clean_data["points_ref"], dtype=np.float32).copy()
        for name, points in (("noisy", noisy_ref), ("clean", clean_ref)):
            if points.shape != (500, 3) or not np.isfinite(points).all():
                raise ValueError(f"Invalid {name} patch: {row['pair_id']}")
        return {
            "noisy_ref": torch.from_numpy(noisy_ref),
            "clean_ref": torch.from_numpy(clean_ref),
            "clean_pca_frame": torch.from_numpy(frame),
            "clean_reference_radius": torch.from_numpy(radius),
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        values = self.cache[index] if self.cache is not None else self._load(row)
        return {
            **{key: value.clone() for key, value in values.items()},
            "pair_id": row["pair_id"],
            "shape": row["shape"],
        }


def build_pf_loader(args: argparse.Namespace) -> DataLoader:
    dataset = PointcloudPatchDataset(
        root=str(args.train_root),
        shapes_list_file="train.txt",
        patch_radius=args.patch_radius,
        points_per_patch=args.pf_points_per_patch,
        seed=args.seed,
        train_state="train",
    )
    sampler = RandomPointcloudPatchSampler(
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
        collate_fn=my_collate,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if args.workers > 0 else None,
        generator=generator,
    )


def build_geometry_loader(args: argparse.Namespace) -> DataLoader:
    dataset = SharedNoisyCleanGeometryDataset(args.shared_manifest, args.shape_split)
    generator = torch.Generator().manual_seed(args.seed + 100)
    return DataLoader(
        dataset,
        batch_size=args.geometry_batch_size,
        shuffle=True,
        num_workers=args.geometry_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if args.geometry_workers > 0 else None,
        generator=generator,
    )


def load_pf(path: Path, device: torch.device) -> torch.nn.Module:
    model = pointfilternet().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    return model


def load_frozen_qnet(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["provenance"]["config"]
    model = EdgeConvQualityQueryNet5D(
        k=int(config["k"]),
        output_dim=5,
        query_heads=int(config["query_heads"]),
        graph_mode=str(config["graph_mode"]),
        query_interaction=bool(config["query_interaction"]),
        encoder_type="edgeconv",
        quality_mode="direct",
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def knn_pf_candidate(pointfilter: torch.nn.Module, noisy: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    b, n, _ = noisy.shape
    k_eff = min(k, n)
    with torch.no_grad():
        idx = torch.topk(torch.cdist(noisy, noisy), k=k_eff, largest=False, dim=-1).indices
    batch = torch.arange(b, device=noisy.device)[:, None, None]
    neighbors = noisy[batch, idx]
    local = neighbors - noisy[:, :, None, :]
    pf_input = local.reshape(b * n, k_eff, 3).transpose(2, 1).contiguous()
    offset = pointfilter(pf_input).reshape(b, n, 3)
    return noisy + offset, offset


def local_pca(points: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    b, n, _ = points.shape
    k_eff = min(max(3, int(k)), n)
    with torch.no_grad():
        idx = torch.topk(torch.cdist(points.detach(), points.detach()), k=k_eff, largest=False, dim=-1).indices
    batch = torch.arange(b, device=points.device)[:, None, None]
    neighbors = points[batch, idx]
    centered = neighbors - neighbors.mean(dim=2, keepdim=True)
    covariance = centered.transpose(-1, -2) @ centered / float(k_eff)
    eye = torch.eye(3, dtype=points.dtype, device=points.device).view(1, 1, 3, 3)
    eigvals, eigvecs = torch.linalg.eigh(covariance + 1e-6 * eye)
    eigvals = eigvals.clamp_min(0.0)
    normals = torch.nan_to_num(eigvecs[..., 0])
    curvature = eigvals[..., 0] / eigvals.sum(dim=-1).clamp_min(1e-8)
    return normals, torch.nan_to_num(curvature, nan=0.0, posinf=1.0, neginf=0.0)


def percentile_rank_weights(response: torch.Tensor) -> torch.Tensor:
    """Exact per-row ranks in [0,1], then w=1+ranks."""
    b, n = response.shape
    order = torch.argsort(response, dim=1, stable=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(n, device=response.device).view(1, n).expand(b, n)
    ranks.scatter_(1, order, rank_values)
    return 1.0 + ranks.to(response.dtype) / float(max(n - 1, 1))


@torch.no_grad()
def structure_weights(
    qnet: torch.nn.Module,
    candidate_ref: torch.Tensor,
    frame: torch.Tensor,
    radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Reconstruct the candidate in full-cloud raw scale/orientation. Translation
    # cancels because the frozen Student protocol centers the candidate itself.
    candidate_raw = torch.bmm(candidate_ref, frame.transpose(1, 2)) * radius.view(-1, 1, 1)
    qnet_input = candidate_raw - candidate_raw.mean(dim=1, keepdim=True)
    _, attention = qnet(qnet_input, return_attention=True)
    response = attention[:, :, STRUCTURE_QUERY_INDEX, :].mean(dim=1)
    weights = percentile_rank_weights(response)
    return response.detach(), weights.detach()


def geometry_loss(
    pointfilter: torch.nn.Module,
    qnet: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    variant: str,
    pf_k: int,
    pca_k: int,
    normal_weight: float,
    curvature_weight: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    noisy = batch["noisy_ref"].float().to(device, non_blocking=True)
    clean = batch["clean_ref"].float().to(device, non_blocking=True)
    frame = batch["clean_pca_frame"].float().to(device, non_blocking=True)
    radius = batch["clean_reference_radius"].float().to(device, non_blocking=True)
    candidate, offset = knn_pf_candidate(pointfilter, noisy, pf_k)
    if variant == "guided":
        response, weights = structure_weights(qnet, candidate, frame, radius)
    elif variant == "uniform":
        response = torch.zeros(candidate.shape[:2], device=device)
        weights = torch.ones(candidate.shape[:2], device=device)
    else:
        raise ValueError(variant)

    _, pred_curvature = local_pca(candidate, pca_k)
    with torch.no_grad():
        clean_normals, clean_curvature = local_pca(clean, pca_k)
        nearest = torch.cdist(candidate.detach(), clean).argmin(dim=-1)
        batch_idx = torch.arange(clean.shape[0], device=device)[:, None]
        ref_points = clean[batch_idx, nearest]
        ref_normals = clean_normals[batch_idx, nearest]
        ref_curvature = clean_curvature[batch_idx, nearest]
    normal_error = ((candidate - ref_points) * ref_normals).sum(dim=-1).abs()
    curvature_error = (pred_curvature - ref_curvature).abs()
    normal_error = torch.nan_to_num(normal_error, nan=0.0, posinf=1.0, neginf=1.0)
    curvature_error = torch.nan_to_num(curvature_error, nan=0.0, posinf=1.0, neginf=1.0)
    denominator = weights.sum(dim=1).clamp_min(1e-6)
    normal_loss = ((weights * normal_error).sum(dim=1) / denominator).mean()
    curvature_loss = ((weights * curvature_error).sum(dim=1) / denominator).mean()
    loss = float(normal_weight) * normal_loss + float(curvature_weight) * curvature_loss
    stats = {
        "normal": float(normal_loss.detach().cpu()),
        "curvature": float(curvature_loss.detach().cpu()),
        "weight_min": float(weights.min(dim=1).values.mean().cpu()),
        "weight_mean": float(weights.mean(dim=1).mean().cpu()),
        "weight_max": float(weights.max(dim=1).values.mean().cpu()),
        "response_finite": float(torch.isfinite(response).all().cpu()),
        "offset_norm": float(offset.detach().norm(dim=-1).mean().cpu()),
    }
    return loss, stats, candidate, weights


def original_pf_loss(
    pointfilter: torch.nn.Module,
    batch: Any,
    device: torch.device,
    support_multiple: float,
    support_angle: float,
    repulsion_alpha: float,
) -> torch.Tensor | None:
    if batch is None or len(batch) == 0:
        return None
    noise, gt, normals, support_radius = batch
    noise = noise.float().to(device, non_blocking=True).transpose(2, 1).contiguous()
    gt = gt.float().to(device, non_blocking=True)
    normals = normals.float().to(device, non_blocking=True)
    support_radius = support_multiple * support_radius.float().to(device, non_blocking=True)
    prediction = pointfilter(noise)
    return 100.0 * compute_bilateral_loss_with_repulsion(
        prediction, gt, normals, support_radius, support_angle, repulsion_alpha
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("uniform", "guided"), required=True)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_PF_TRAIN)
    parser.add_argument("--shared-manifest", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--shape-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--pf-checkpoint", type=Path, default=DEFAULT_PF_CKPT)
    parser.add_argument("--qnet-checkpoint", type=Path, default=DEFAULT_QNET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patch-radius", type=float, default=0.05)
    parser.add_argument("--pf-points-per-patch", type=int, default=500)
    parser.add_argument("--patches-per-shape", type=int, default=8000)
    parser.add_argument("--geometry-batch-size", type=int, default=1)
    parser.add_argument("--geometry-workers", type=int, default=0)
    parser.add_argument("--geometry-pf-k", type=int, default=128)
    parser.add_argument("--geometry-pca-k", type=int, default=24)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--normal-weight", type=float, default=1.0)
    parser.add_argument("--curvature-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--support-multiple", type=float, default=4.0)
    parser.add_argument("--support-angle", type=float, default=15.0)
    parser.add_argument("--repulsion-alpha", type=float, default=0.97)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"{args.variant}_seed{args.seed}"
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pointfilter = load_pf(args.pf_checkpoint, device)
    qnet = load_frozen_qnet(args.qnet_checkpoint, device)
    initial_pf_hash = state_sha256(pointfilter)
    qnet_hash_before = state_sha256(qnet)
    pf_loader = build_pf_loader(args)
    geometry_loader = build_geometry_loader(args)
    optimizer = torch.optim.SGD(pointfilter.parameters(), lr=args.lr, momentum=args.momentum)
    support_angle = args.support_angle / 360.0 * 2.0 * np.pi

    config = {
        "schema_version": "pointfilter_structure_guidance_v1",
        "variant": args.variant,
        "only_ab_difference": "per-point geometry weights",
        "uniform_weights": "w_i=1",
        "guided_weights": "w_i=1+percentile_rank(mean-head Structure cross-attention_i)",
        "qnet_gradient_to_coordinates": False,
        "qnet_checkpoint": str(args.qnet_checkpoint),
        "qnet_checkpoint_sha256": sha256(args.qnet_checkpoint),
        "pf_initial_checkpoint": str(args.pf_checkpoint),
        "pf_initial_checkpoint_sha256": sha256(args.pf_checkpoint),
        "pf_initial_state_sha256": initial_pf_hash,
        "train_root": str(args.train_root),
        "shared_manifest": str(args.shared_manifest),
        "shared_manifest_sha256": sha256(args.shared_manifest),
        "shape_split": str(args.shape_split),
        "shape_split_sha256": sha256(args.shape_split),
        "qnet_input": "candidate raw-scale coordinates - candidate mean",
        "geometry_loss": "0.1 * (1.0 normalized weighted normal + 1.0 normalized weighted curvature)",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pf_batches_per_epoch": len(pf_loader),
        "geometry_train_pairs": len(geometry_loader.dataset),
    }
    write_json(args.output_dir / "config.json", config)

    history = []
    sanity = None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        current_lr = scheduled_lr(epoch - 1, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        pointfilter.train()
        geometry_iter = iter(geometry_loader)
        totals = {key: [] for key in ("total", "pf", "geometry", "normal", "curvature", "wmin", "wmean", "wmax")}
        for step, pf_batch in enumerate(pf_loader, start=1):
            if args.max_steps_per_epoch > 0 and step > args.max_steps_per_epoch:
                break
            geometry_batch, geometry_iter = next_or_restart(geometry_iter, geometry_loader)
            pointfilter.train()
            loss_pf = original_pf_loss(
                pointfilter, pf_batch, device, args.support_multiple,
                support_angle, args.repulsion_alpha,
            )
            if loss_pf is None:
                continue
            pointfilter.apply(freeze_bn_stats)
            loss_geometry, stats, candidate, weights = geometry_loss(
                pointfilter, qnet, geometry_batch, device, args.variant,
                args.geometry_pf_k, args.geometry_pca_k,
                args.normal_weight, args.curvature_weight,
            )
            total_loss = loss_pf + args.geometry_weight * loss_geometry
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            pf_has_grad = any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all().detach().cpu().item())
                for parameter in pointfilter.parameters()
            )
            torch.nn.utils.clip_grad_norm_(pointfilter.parameters(), args.grad_clip)
            optimizer.step()

            if sanity is None:
                qnet_hash_after = state_sha256(qnet)
                qnet_grad_none = all(parameter.grad is None for parameter in qnet.parameters())
                sanity = {
                    "variant": args.variant,
                    "qnet_parameters_grad_none": qnet_grad_none,
                    "qnet_parameters_unchanged": qnet_hash_after == qnet_hash_before,
                    "response_shape": [int(weights.shape[0]), int(weights.shape[1])],
                    "weights_finite": bool(torch.isfinite(weights).all().cpu()),
                    "weight_min": float(weights.min().cpu()),
                    "weight_mean": float(weights.mean().cpu()),
                    "weight_max": float(weights.max().cpu()),
                    "geometry_loss_finite": bool(torch.isfinite(loss_geometry).cpu()),
                    "pf_parameters_receive_gradients": bool(pf_has_grad),
                    "qnet_backward_path_present": False,
                }
                expected_max = 2.0 if args.variant == "guided" else 1.0
                checks = [
                    sanity["qnet_parameters_grad_none"],
                    sanity["qnet_parameters_unchanged"],
                    sanity["response_shape"][1] == 500,
                    sanity["weights_finite"],
                    abs(sanity["weight_min"] - 1.0) < 1e-5,
                    abs(sanity["weight_max"] - expected_max) < 1e-5,
                    sanity["geometry_loss_finite"],
                    sanity["pf_parameters_receive_gradients"],
                ]
                sanity["passed"] = all(checks)
                write_json(args.output_dir / "sanity_check.json", sanity)
                print("SANITY " + json.dumps(sanity, ensure_ascii=False), flush=True)
                if not sanity["passed"]:
                    raise RuntimeError("First-batch sanity check failed")

            totals["total"].append(float(total_loss.detach().cpu()))
            totals["pf"].append(float(loss_pf.detach().cpu()))
            totals["geometry"].append(float((args.geometry_weight * loss_geometry).detach().cpu()))
            totals["normal"].append(stats["normal"])
            totals["curvature"].append(stats["curvature"])
            totals["wmin"].append(stats["weight_min"])
            totals["wmean"].append(stats["weight_mean"])
            totals["wmax"].append(stats["weight_max"])
            if step == 1 or step % args.print_every == 0:
                print(
                    f"[{args.variant} {epoch}/{args.epochs} {step}/{len(pf_loader)}] "
                    f"total={totals['total'][-1]:.6f} pf={totals['pf'][-1]:.6f} "
                    f"geo={totals['geometry'][-1]:.6f} n={stats['normal']:.6f} "
                    f"c={stats['curvature']:.6f} w=[{stats['weight_min']:.3f},"
                    f"{stats['weight_mean']:.3f},{stats['weight_max']:.3f}]",
                    flush=True,
                )

        record = {
            "epoch": epoch,
            "lr": current_lr,
            "valid_batches": len(totals["total"]),
            **{key: float(np.mean(values)) for key, values in totals.items()},
        }
        history.append(record)
        write_json(args.output_dir / "history.json", history)
        print("EPOCH " + json.dumps(record), flush=True)
        checkpoint = {
            "epoch": epoch,
            "state_dict": pointfilter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "history": history,
            "sanity": sanity,
            "qnet_state_sha256": qnet_hash_before,
        }
        if epoch % args.save_every == 0:
            torch.save(checkpoint, args.output_dir / f"model_epoch_{epoch}.pth")

    qnet_hash_final = state_sha256(qnet)
    if qnet_hash_final != qnet_hash_before:
        raise RuntimeError("Frozen QNet parameters changed during training")
    final_path = args.output_dir / "model_final.pth"
    torch.save(checkpoint, final_path)
    summary = {
        "status": "training_complete",
        "variant": args.variant,
        "epochs": len(history),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(final_path),
        "pf_initial_state_sha256": initial_pf_hash,
        "qnet_state_sha256_before": qnet_hash_before,
        "qnet_state_sha256_after": qnet_hash_final,
        "qnet_parameters_unchanged": qnet_hash_final == qnet_hash_before,
        "sanity": sanity,
        "last_epoch": history[-1],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
