#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整点云 + 局部放大框（Mitsuba 湖水蓝小球）。

patch 只用于选择 anchor/ROI；图中的整云和放大区域都来自真实 full-cloud
推理输出。四种方法使用同一个世界坐标球形 ROI，绝不直接渲染 500 点 patch。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("DRJIT_LIBLLVM_PATH", "/home/zhangzy/anaconda3/envs/cvpr/lib/libLLVM-14.so")

import mitsuba as mi
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

mi.set_variant("scalar_rgb")

ROOT = Path("/data/zhangzy/zzyy")
STAGE3 = ROOT / "PointFilter_frozen_quality_stage3"
CLEAN_DIR = ROOT / "PUNet/Gaussion/clean_normalized"
PATCH_ROOT = ROOT / "PUNet/Gaussion/shared_patches/clean"
RECORDS = STAGE3 / "paired_patch_distribution_lambda_0.01_epoch17/records.json"
FULL_ROOT = STAGE3 / "fullcloud_lambda0.01_ep17_roi_cases/work"
OUT_DIR = STAGE3 / "visualizations_fullcloud_roi_inset_lake"

METHODS = (
    ("Noisy", "iter_0.npy"),
    ("Baseline PointFilter", "iter_1_base.npy"),
    ("Baseline + QNet", "iter_1_qual.npy"),
    ("Clean GT", None),
)

LAKE_RGB = (0.25, 0.75, 0.70)
PEACH = (255, 185, 145)
GLOBAL_SIZE = 900
INSET_SIZE = 430
GLOBAL_RADIUS = 0.0075
GLOBAL_FOV = 25.0
GLOBAL_ORIGIN = np.asarray([4.0, 0.0, 2.0], dtype=np.float64)
GLOBAL_TARGET = np.zeros(3, dtype=np.float64)
GLOBAL_UP = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
ROTATION_XYZ = (90.0, 270.0, 0.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", default="pulley")
    p.add_argument("--sigma", default="0.020")
    p.add_argument("--anchor", default="anchor_02")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--full-root", type=Path, default=FULL_ROOT,
                   help="Root containing <shape>/sigma_xxx/iter_{0,1_base,1_qual}.npy")
    p.add_argument("--global-spp", type=int, default=16)
    p.add_argument("--roi-spp", type=int, default=256)
    p.add_argument("--roi-radius-factor", type=float, default=0.82)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_case(full_root: Path, shape: str, sigma: str, anchor: str):
    work = full_root / shape / f"sigma_{float(sigma):.3f}"
    patch_file = PATCH_ROOT / shape / f"{anchor}.npz"
    if not work.exists():
        raise FileNotFoundError(f"full-cloud outputs do not exist: {work}")
    if not patch_file.exists():
        raise FileNotFoundError(patch_file)
    gt = np.loadtxt(CLEAN_DIR / f"{shape}.xyz", dtype=np.float64)[:, :3]
    clouds = {
        "Noisy": np.load(work / "iter_0.npy").astype(np.float64),
        "Baseline PointFilter": np.load(work / "iter_1_base.npy").astype(np.float64),
        "Baseline + QNet": np.load(work / "iter_1_qual.npy").astype(np.float64),
        "Clean GT": gt,
    }
    with np.load(patch_file) as data:
        center = np.asarray(data["clean_center_xyz"], dtype=np.float64)
        radius = float(data["clean_reference_radius"])
        frame = np.asarray(data["clean_pca_frame"], dtype=np.float64)
    return work, patch_file, clouds, center, radius, frame


def patch_record(shape: str, sigma: str, anchor: str) -> dict:
    sid = f"noisy/sigma_{float(sigma):.3f}/{shape}/{anchor}"
    for record in json.loads(RECORDS.read_text(encoding="utf-8")):
        if record["sample_id"] == sid:
            base = float(record["baseline"]["patch_metrics"]["chamfer_distance"])
            qnet = float(record["quality"]["patch_metrics"]["chamfer_distance"])
            return {
                "sample_id": sid,
                "patch_cd_baseline": base,
                "patch_cd_qnet": qnet,
                "patch_cd_improvement_percent": 100.0 * (base - qnet) / base,
                "structure_type": record.get("structure_type"),
            }
    return {"sample_id": sid, "patch_cd_baseline": None, "patch_cd_qnet": None,
            "patch_cd_improvement_percent": None, "structure_type": None}


def shared_global_transform(gt: np.ndarray):
    """Only GT defines the display transform; every method reuses it."""
    lo, hi = gt.min(0), gt.max(0)
    center1 = (lo + hi) * 0.5
    scale1 = float(np.max(hi - lo))
    rotation = Rotation.from_euler("xyz", ROTATION_XYZ, degrees=True)

    def first(points):
        normalized = (points - center1) / scale1
        return rotation.apply(normalized[:, [2, 0, 1]])

    gt_first = first(gt)
    lo2, hi2 = gt_first.min(0), gt_first.max(0)
    center2 = (lo2 + hi2) * 0.5
    scale2 = float(np.max(hi2 - lo2))

    def transform(points):
        return ((first(points) - center2) / scale2).astype(np.float32)

    return transform, {
        "raw_center": center1.tolist(), "raw_scale": scale1,
        "display_center": center2.tolist(), "display_scale": scale2,
        "rotation_xyz_degree": list(ROTATION_XYZ),
    }


def crop_same_world_roi(clouds, center, radius, frame):
    """Crop one physical ball from all full clouds and map to the clean patch frame."""
    rois, counts = {}, {}
    for name, cloud in clouds.items():
        keep = np.linalg.norm(cloud - center, axis=1) <= radius
        raw = cloud[keep]
        rois[name] = (((raw - center) @ frame) / radius).astype(np.float32)
        counts[name] = int(keep.sum())
    return rois, counts


def scene_xml(points, sphere_radius, size, spp, origin, up, fov,
              colors=None, clean_background=False):
    if colors is None:
        colors = np.tile(np.asarray(LAKE_RGB), (len(points), 1))
    spheres = []
    for (x, y, z), (r, g, b) in zip(points, colors):
        spheres.append(f"""
        <shape type="sphere">
          <float name="radius" value="{sphere_radius:.8f}"/>
          <transform name="to_world"><translate x="{x:.7f}" y="{y:.7f}" z="{z:.7f}"/></transform>
          <bsdf type="diffuse"><rgb name="reflectance" value="{r},{g},{b}"/></bsdf>
        </shape>""")
    if clean_background:
        lighting = """
        <emitter type="constant"><rgb name="radiance" value="1,1,1"/></emitter>
        """
    else:
        lighting = """
        <shape type="rectangle">
          <transform name="to_world"><scale x="10" y="10" z="1"/><translate x="0" y="0" z="-0.5"/></transform>
          <bsdf type="roughplastic"><string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/><float name="int_ior" value="1.46"/>
            <rgb name="diffuse_reflectance" value="1,1,1"/></bsdf>
        </shape>
        <shape type="rectangle"><transform name="to_world"><scale x="10" y="10" z="1"/>
          <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/></transform>
          <emitter type="area"><rgb name="radiance" value="6,6,6"/></emitter></shape>
        """
    return f"""
    <scene version="2.1.0">
      <integrator type="path"><integer name="max_depth" value="-1"/></integrator>
      <sensor type="perspective">
        <float name="far_clip" value="100"/><float name="near_clip" value="0.01"/>
        <transform name="to_world"><lookat
          origin="{origin[0]},{origin[1]},{origin[2]}" target="0,0,0"
          up="{up[0]},{up[1]},{up[2]}"/></transform>
        <float name="fov" value="{fov}"/>
        <sampler type="independent"><integer name="sample_count" value="{spp}"/></sampler>
        <film type="hdrfilm"><integer name="width" value="{size}"/>
          <integer name="height" value="{size}"/><rfilter type="gaussian"/>
          <boolean name="banner" value="false"/></film>
      </sensor>
      {''.join(spheres)}
      {lighting}
    </scene>"""


def render(points, sphere_radius, size, spp, origin, up, fov,
           colors=None, clean_background=False):
    xml = scene_xml(points, sphere_radius, size, spp, origin, up, fov,
                    colors=colors, clean_background=clean_background)
    image = np.clip(np.asarray(mi.render(mi.load_string(xml))), 0.0, 1.0)
    return Image.fromarray((image * 255.0).astype(np.uint8))


def project(points, origin, target, up, fov, size):
    forward = target - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rel = points - origin
    depth = rel @ forward
    denom = depth * np.tan(np.deg2rad(fov) / 2.0)
    x = (rel @ right) / denom
    y = (rel @ true_up) / denom
    return np.column_stack(((x + 1.0) * 0.5 * size, (1.0 - y) * 0.5 * size))


def font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)
    except OSError:
        return ImageFont.load_default()


def make_card(global_image, inset, global_roi_points, title):
    """Whole cloud + orange source box + overlapping enlarged full-cloud ROI."""
    label_h = 82
    card = Image.new("RGB", (GLOBAL_SIZE, GLOBAL_SIZE + label_h), "white")
    card.paste(global_image, (0, 0))
    draw = ImageDraw.Draw(card)
    uv = project(global_roi_points, GLOBAL_ORIGIN, GLOBAL_TARGET, GLOBAL_UP, GLOBAL_FOV, GLOBAL_SIZE)
    uv = uv[np.isfinite(uv).all(axis=1)]
    x0, y0 = np.maximum(uv.min(0) - 13, 5).astype(int)
    x1, y1 = np.minimum(uv.max(0) + 13, GLOBAL_SIZE - 5).astype(int)
    draw.rectangle((x0, y0, x1, y1), outline=PEACH, width=9)

    ix, iy = GLOBAL_SIZE - INSET_SIZE - 18, GLOBAL_SIZE - INSET_SIZE - 18
    border = 9
    card.paste(Image.new("RGB", (INSET_SIZE + 2 * border, INSET_SIZE + 2 * border), PEACH),
               (ix - border, iy - border))
    card.paste(inset, (ix, iy))
    draw.line((x1, (y0 + y1) // 2, ix - border, iy + INSET_SIZE // 2), fill=PEACH, width=6)
    bbox = draw.textbbox((0, 0), title, font=font(34, bold=True))
    draw.text(((GLOBAL_SIZE - (bbox[2] - bbox[0])) / 2, GLOBAL_SIZE + 20), title,
              fill=(10, 15, 18), font=font(34, bold=True))
    return card


def mark_global(global_image, global_roi_points):
    """Draw only the source ROI box on a full-cloud panel."""
    image = global_image.copy()
    draw = ImageDraw.Draw(image)
    uv = project(global_roi_points, GLOBAL_ORIGIN, GLOBAL_TARGET, GLOBAL_UP,
                 GLOBAL_FOV, GLOBAL_SIZE)
    uv = uv[np.isfinite(uv).all(axis=1)]
    x0, y0 = np.maximum(uv.min(0) - 13, 5).astype(int)
    x1, y1 = np.minimum(uv.max(0) + 13, GLOBAL_SIZE - 5).astype(int)
    draw.rectangle((x0, y0, x1, y1), outline=PEACH, width=10)
    return image


def compose_top_bottom(global_images, roi_images, titles, heading, output):
    """Paper layout: full-cloud localization on top, clean ROI directly below."""
    panel = 650
    gap, left, top, label_h, row_gap = 16, 16, 86, 52, 18
    width = left * 2 + len(titles) * panel + (len(titles) - 1) * gap
    height = top + label_h + panel + row_gap + label_h + panel + 20
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 18), heading, fill=(20, 25, 28), font=font(27, bold=True))
    for col, title in enumerate(titles):
        x = left + col * (panel + gap)
        bbox = draw.textbbox((0, 0), title, font=font(25, bold=True))
        draw.text((x + (panel - (bbox[2] - bbox[0])) / 2, top + 8), title,
                  fill=(15, 20, 22), font=font(25, bold=True))
        canvas.paste(global_images[col].resize((panel, panel), Image.Resampling.LANCZOS),
                     (x, top + label_h))
        y2 = top + label_h + panel + row_gap
        bbox2 = draw.textbbox((0, 0), "Full-cloud ROI", font=font(22, bold=True))
        draw.text((x + (panel - (bbox2[2] - bbox2[0])) / 2, y2 + 10),
                  "Full-cloud ROI", fill=(45, 50, 52), font=font(22, bold=True))
        roi = roi_images[col].resize((panel, panel), Image.Resampling.LANCZOS)
        canvas.paste(roi, (x, y2 + label_h))
        ImageDraw.Draw(canvas).rectangle(
            (x, y2 + label_h, x + panel - 1, y2 + label_h + panel - 1),
            outline=PEACH, width=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    canvas.save(output.with_suffix(".pdf"), resolution=300.0)


def error_colors(values, vmax):
    """Blue -> cyan -> yellow -> red, shared across Baseline and Ours."""
    t = np.clip(np.asarray(values) / max(vmax, 1e-12), 0.0, 1.0)
    stops = np.asarray([[0.08, 0.25, 0.80], [0.00, 0.78, 0.92],
                        [1.00, 0.88, 0.12], [0.88, 0.08, 0.05]])
    q = t * 3.0
    i = np.minimum(q.astype(int), 2)
    f = (q - i)[:, None]
    return stops[i] * (1.0 - f) + stops[i + 1] * f


def difference_colors(values, limit):
    """Blue=worse, white=unchanged, red=better (Baseline error - QNet error)."""
    t = np.clip(np.asarray(values) / max(limit, 1e-12), -1.0, 1.0)
    blue = np.asarray([0.10, 0.35, 0.90])
    white = np.asarray([0.94, 0.94, 0.94])
    red = np.asarray([0.92, 0.12, 0.08])
    out = np.empty((len(t), 3), dtype=np.float64)
    neg = t < 0
    out[neg] = blue * (-t[neg, None]) + white * (1.0 + t[neg, None])
    out[~neg] = white * (1.0 - t[~neg, None]) + red * t[~neg, None]
    return out


def compose_error_maps(images, titles, vmax, output):
    panel, gap, top, label_h = 650, 20, 90, 58
    width = 2 * panel + 3 * gap
    height = top + label_h + panel + 74
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 18), "Full-cloud ROI point-to-GT error (shared scale)",
              fill=(20, 25, 28), font=font(28, bold=True))
    for col, (image, title) in enumerate(zip(images, titles)):
        x = gap + col * (panel + gap)
        bbox = draw.textbbox((0, 0), title, font=font(27, bold=True))
        draw.text((x + (panel - (bbox[2] - bbox[0])) / 2, top + 9), title,
                  fill=(15, 20, 22), font=font(27, bold=True))
        canvas.paste(image.resize((panel, panel), Image.Resampling.LANCZOS),
                     (x, top + label_h))
    # Horizontal shared colorbar.
    bx0, bx1 = 120, width - 120
    by0, by1 = height - 48, height - 27
    for x in range(bx0, bx1):
        value = (x - bx0) / max(bx1 - bx0 - 1, 1)
        color = tuple((error_colors([value], 1.0)[0] * 255).astype(np.uint8))
        draw.line((x, by0, x, by1), fill=color)
    draw.text((bx0, by1 + 4), "0", fill=(20, 20, 20), font=font(16))
    label = f"{vmax:.5f} (95th percentile)"
    box = draw.textbbox((0, 0), label, font=font(16))
    draw.text((bx1 - (box[2] - box[0]), by1 + 4), label,
              fill=(20, 20, 20), font=font(16))
    canvas.save(output)
    canvas.save(output.with_suffix(".pdf"), resolution=300.0)


def compose_difference(image, limit, improved_fraction, mean_delta, output):
    panel, margin, top = 720, 24, 132
    width, height = panel + 2 * margin, top + panel + 92
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 16), "Full-cloud ROI error difference",
              fill=(20, 25, 28), font=font(29, bold=True))
    draw.text((margin, 58), "red: QNet better | blue: QNet worse",
              fill=(45, 50, 53), font=font(18))
    draw.text((margin, 84),
              f"improved points {100.0 * improved_fraction:.1f}% | mean delta {mean_delta:+.2e}",
              fill=(45, 50, 53), font=font(18))
    canvas.paste(image.resize((panel, panel), Image.Resampling.LANCZOS), (margin, top))
    bx0, bx1, by0, by1 = 90, width - 90, height - 49, height - 28
    values = np.linspace(-limit, limit, bx1 - bx0)
    colors = (difference_colors(values, limit) * 255).astype(np.uint8)
    for i, x in enumerate(range(bx0, bx1)):
        draw.line((x, by0, x, by1), fill=tuple(colors[i]))
    draw.text((bx0, by1 + 4), f"{-limit:.1e}", fill=(20, 20, 20), font=font(15))
    zero = "0"
    zw = draw.textbbox((0, 0), zero, font=font(15))[2]
    draw.text(((bx0 + bx1 - zw) / 2, by1 + 4), zero, fill=(20, 20, 20), font=font(15))
    right = f"+{limit:.1e}"
    rw = draw.textbbox((0, 0), right, font=font(15))[2]
    draw.text((bx1 - rw, by1 + 4), right, fill=(20, 20, 20), font=font(15))
    canvas.save(output)
    canvas.save(output.with_suffix(".pdf"), resolution=300.0)


def compose(cards, heading, output):
    gap, top = 16, 82
    width = len(cards) * GLOBAL_SIZE + (len(cards) + 1) * gap
    height = top + cards[0].height + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 18), heading, fill=(20, 25, 28), font=font(28, bold=True))
    x = gap
    for card in cards:
        canvas.paste(card, (x, top))
        x += GLOBAL_SIZE + gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    canvas.save(output.with_suffix(".pdf"), resolution=300.0)


def main():
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    work, patch_file, clouds, center, roi_radius, frame = load_case(
        a.full_root, a.shape, a.sigma, a.anchor)
    record = patch_record(a.shape, a.sigma, a.anchor)
    transform, transform_meta = shared_global_transform(clouds["Clean GT"])
    global_clouds = {name: transform(cloud) for name, cloud in clouds.items()}
    rois, counts = crop_same_world_roi(clouds, center, roi_radius, frame)

    # Clean ROI defines one shared inset framing. This centers/fills the inset
    # without allowing each method to choose its own crop or zoom.
    clean_local = rois["Clean GT"]
    lo_xy, hi_xy = clean_local[:, :2].min(0), clean_local[:, :2].max(0)
    roi_view_center = np.asarray([
        *((lo_xy + hi_xy) * 0.5),
        float((clean_local[:, 2].min() + clean_local[:, 2].max()) * 0.5),
    ])
    roi_view_half = float(np.max(hi_xy - lo_xy) * 0.5 * 1.04)
    rois = {name: ((points - roi_view_center) / roi_view_half).astype(np.float32)
            for name, points in rois.items()}
    clean_xy = rois["Clean GT"][:, :2]
    spacing = float(np.median(cKDTree(clean_xy).query(clean_xy, k=2)[0][:, 1]))
    roi_sphere_radius = float(np.clip(spacing * a.roi_radius_factor, 0.008, 0.050))
    roi_distance = 1.12 / np.tan(np.deg2rad(GLOBAL_FOV) / 2.0)
    roi_origin = np.asarray([0.0, 0.0, roi_distance])
    roi_up = np.asarray([0.0, 1.0, 0.0])
    gt_keep = np.linalg.norm(clouds["Clean GT"] - center, axis=1) <= roi_radius
    global_roi_gt = transform(clouds["Clean GT"][gt_keep])

    # Honest full-cloud local Chamfer. Nearest-neighbour searches use complete
    # counterpart clouds, while averaging is restricted to the fixed ROI.
    gt_tree = cKDTree(clouds["Clean GT"])
    full_trees = {name: cKDTree(cloud) for name, cloud in clouds.items()}
    raw_masks = {name: np.linalg.norm(cloud - center, axis=1) <= roi_radius
                 for name, cloud in clouds.items()}
    local_cd, point_errors = {}, {}
    gt_roi = clouds["Clean GT"][raw_masks["Clean GT"]]
    for name in ("Noisy", "Baseline PointFilter", "Baseline + QNet"):
        pred_roi = clouds[name][raw_masks[name]]
        point_errors[name] = gt_tree.query(pred_roi, k=1, workers=-1)[0]
        local_cd[name] = float(
            point_errors[name].mean()
            + full_trees[name].query(gt_roi, k=1, workers=-1)[0].mean()
        )
    local_gain = local_cd["Baseline PointFilter"] - local_cd["Baseline + QNet"]
    local_gain_percent = 100.0 * local_gain / local_cd["Baseline PointFilter"]

    cards, method_files, global_images, roi_images = [], {}, [], []
    for name, _ in METHODS:
        print(f"[render global] {name}", flush=True)
        whole = render(global_clouds[name], GLOBAL_RADIUS, GLOBAL_SIZE, a.global_spp,
                       GLOBAL_ORIGIN, GLOBAL_UP, GLOBAL_FOV)
        print(f"[render ROI] {name} n={counts[name]}", flush=True)
        inset = render(rois[name], roi_sphere_radius, INSET_SIZE, a.roi_spp,
                       roi_origin, roi_up, GLOBAL_FOV, clean_background=True)
        safe = name.lower().replace(" ", "_").replace("+", "plus")
        whole_path = a.out_dir / f"{safe}_full.png"
        roi_path = a.out_dir / f"{safe}_fullcloud_roi.png"
        card_path = a.out_dir / f"{safe}_full_with_inset.png"
        whole.save(whole_path)
        inset.save(roi_path)
        card = make_card(whole, inset, global_roi_gt, name)
        card.save(card_path)
        cards.append(card)
        global_images.append(mark_global(whole, global_roi_gt))
        roi_images.append(inset)
        method_files[name] = {"full": str(whole_path), "roi": str(roi_path), "card": str(card_path)}

    heading = (f"{a.shape}  sigma={float(a.sigma):.3f}  {a.anchor}  |  "
               f"full-cloud ROI CD {local_cd['Baseline PointFilter']:.6f} -> "
               f"{local_cd['Baseline + QNet']:.6f}  ({local_gain_percent:+.3f}% better)")
    all_output = a.out_dir / "fullcloud_roi_inset_all_methods.png"
    pair_output = a.out_dir / "fullcloud_roi_inset_baseline_vs_qnet.png"
    compose(cards, heading, all_output)
    compose(cards[1:3], heading, pair_output)

    grid_output = a.out_dir / "fullcloud_top_roi_bottom.png"
    titles = [name for name, _ in METHODS]
    compose_top_bottom(global_images, roi_images, titles, heading, grid_output)

    error_vmax = float(np.quantile(np.concatenate([
        point_errors["Baseline PointFilter"], point_errors["Baseline + QNet"]
    ]), 0.95))
    error_images = []
    for name in ("Baseline PointFilter", "Baseline + QNet"):
        error_images.append(render(
            rois[name], roi_sphere_radius, INSET_SIZE, a.roi_spp,
            roi_origin, roi_up, GLOBAL_FOV,
            colors=error_colors(point_errors[name], error_vmax),
            clean_background=True,
        ))
    error_output = a.out_dir / "fullcloud_roi_error_maps.png"
    compose_error_maps(error_images, ["Baseline PointFilter", "Baseline + QNet"],
                       error_vmax, error_output)

    # Per-index displacement outputs preserve input ordering, so corresponding
    # points can support a signed error-difference map on their ROI intersection.
    common = raw_masks["Baseline PointFilter"] & raw_masks["Baseline + QNet"]
    base_common = clouds["Baseline PointFilter"][common]
    qnet_common = clouds["Baseline + QNet"][common]
    base_error_common = gt_tree.query(base_common, k=1, workers=-1)[0]
    qnet_error_common = gt_tree.query(qnet_common, k=1, workers=-1)[0]
    difference = base_error_common - qnet_error_common
    difference_limit = float(np.quantile(np.abs(difference), 0.95))
    qnet_common_local = (((qnet_common - center) @ frame) / roi_radius
                         - roi_view_center) / roi_view_half
    difference_image = render(
        qnet_common_local.astype(np.float32), roi_sphere_radius, INSET_SIZE,
        a.roi_spp, roi_origin, roi_up, GLOBAL_FOV,
        colors=difference_colors(difference, difference_limit),
        clean_background=True,
    )
    difference_output = a.out_dir / "fullcloud_roi_error_difference.png"
    compose_difference(difference_image, difference_limit, float(np.mean(difference > 0)),
                       float(difference.mean()), difference_output)

    local_distance = {}
    for name, cloud in clouds.items():
        if name == "Clean GT":
            continue
        keep = np.linalg.norm(cloud - center, axis=1) <= roi_radius
        local_distance[name] = float(gt_tree.query(cloud[keep], k=1, workers=-1)[0].mean())
    manifest = {
        "scope": "patch selects location; real full-cloud ROI produces the figure",
        "selection": record,
        "world_roi": {"center_xyz": center.tolist(), "radius": roi_radius,
                      "point_counts": counts, "clean_pca_frame": frame.tolist(),
                      "shared_inset_center_in_patch_frame": roi_view_center.tolist(),
                      "shared_inset_half_extent": roi_view_half},
        "render": {"lake_rgb": list(LAKE_RGB), "global_sphere_radius": GLOBAL_RADIUS,
                   "roi_sphere_radius": roi_sphere_radius,
                   "roi_radius_factor": a.roi_radius_factor,
                   "clean_roi_median_projected_nn_spacing": spacing,
                   "global_fov_degree": GLOBAL_FOV,
                   "global_camera_origin": GLOBAL_ORIGIN.tolist(),
                   "global_spp": a.global_spp, "roi_spp": a.roi_spp,
                   "shared_global_transform": transform_meta},
        "fullcloud_local_mean_distance_to_gt": local_distance,
        "fullcloud_local_symmetric_chamfer": local_cd,
        "fullcloud_local_gain_base_minus_qnet": local_gain,
        "fullcloud_local_improvement_percent": local_gain_percent,
        "error_map": {"metric": "prediction point to full GT nearest-neighbour distance",
                      "shared_vmax_95th_percentile": error_vmax,
                      "path": str(error_output)},
        "error_difference_map": {
            "metric": "baseline point-to-GT error minus QNet point-to-GT error",
            "positive_means": "QNet improves the corresponding point",
            "common_roi_points": int(common.sum()),
            "improved_fraction": float(np.mean(difference > 0)),
            "mean_difference": float(difference.mean()),
            "symmetric_color_limit_95th_abs_percentile": difference_limit,
            "path": str(difference_output),
        },
        "sources": {"fullcloud_work_dir": str(work),
                    "clean_patch_metadata_only": str(patch_file),
                    "paired_patch_records": str(RECORDS), "method_images": method_files},
        "outputs": {"all_methods": str(all_output), "baseline_vs_qnet": str(pair_output),
                    "top_full_bottom_roi": str(grid_output),
                    "roi_error_maps": str(error_output),
                    "roi_error_difference": str(difference_output)},
    }
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"[out] {all_output}", flush=True)
    print(f"[out] {pair_output}", flush=True)
    print(f"[out] {grid_output}", flush=True)
    print(f"[out] {error_output}", flush=True)
    print(f"[out] {difference_output}", flush=True)
    print(f"[out] {a.out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
