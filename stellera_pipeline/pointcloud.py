import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import depth_io as readbin


def load_labelme_mask(json_file, width=None, height=None, labels=None):
    json_path = Path(json_file)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_width = int(width or data.get("imageWidth") or 1280)
    image_height = int(height or data.get("imageHeight") or 720)
    allowed_labels = set(labels) if labels else None

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if allowed_labels is not None and label not in allowed_labels:
            continue

        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if len(points) < 3:
            continue

        shape_type = shape.get("shape_type", "polygon")
        if shape_type == "polygon":
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
        elif shape_type == "rectangle" and len(points) >= 2:
            p1, p2 = np.round(points[:2]).astype(np.int32)
            cv2.rectangle(mask, tuple(p1), tuple(p2), 255, thickness=-1)

    return mask


def depth_to_xyz(depth, camera_params):
    height, width = depth.shape
    y_grid, x_grid = np.mgrid[0:height, 0:width]

    z = depth.astype(np.float64)
    x = (x_grid - camera_params["cx"]) * z / camera_params["fx"]
    y = (y_grid - camera_params["cy"]) * z / camera_params["fy"]
    return x, y, z


def fit_ground_plane_ransac(x, y, z, candidate_mask, max_points=50000, iterations=300, inlier_mm=20.0, seed=42):
    """Fit z = ax + by + c on background points using a small RANSAC loop."""
    valid_idx = np.flatnonzero(candidate_mask.ravel())
    if valid_idx.size < 100:
        raise ValueError(f"Not enough background points for ground fitting: {valid_idx.size}")

    rng = np.random.default_rng(seed)
    if valid_idx.size > max_points:
        valid_idx = rng.choice(valid_idx, size=max_points, replace=False)

    xs = x.ravel()[valid_idx]
    ys = y.ravel()[valid_idx]
    zs = z.ravel()[valid_idx]

    sample_count = xs.size
    best_inliers = None
    best_count = -1

    for _ in range(iterations):
        pick = rng.choice(sample_count, size=3, replace=False)
        matrix = np.column_stack((xs[pick], ys[pick], np.ones(3)))
        try:
            a, b, c = np.linalg.solve(matrix, zs[pick])
        except np.linalg.LinAlgError:
            continue

        residual = np.abs((a * xs + b * ys + c) - zs)
        inliers = residual < inlier_mm
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 100:
        raise ValueError("RANSAC failed to find a stable ground plane.")

    matrix = np.column_stack((xs[best_inliers], ys[best_inliers], np.ones(best_count)))
    a, b, c = np.linalg.lstsq(matrix, zs[best_inliers], rcond=None)[0]
    residual = np.abs((a * xs + b * ys + c) - zs)

    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "candidate_points": int(sample_count),
        "inlier_points": int(np.count_nonzero(residual < inlier_mm)),
        "median_abs_residual_mm": float(np.median(residual)),
    }


def build_point_cloud(x, y, z, color_bgr, keep_mask):
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    rgb = color_rgb[keep_mask]
    point_cloud = np.column_stack((x[keep_mask], y[keep_mask], z[keep_mask], rgb)).astype(np.float64)
    return point_cloud


def save_masked_rgb(color_bgr, mask, output_file):
    preview = np.full_like(color_bgr, 255)
    preview[mask] = color_bgr[mask]
    readbin.imwrite_unicode(output_file, preview)


def save_height_map(height_above_ground, mask, output_file, max_height_mm=300.0):
    image = np.zeros(height_above_ground.shape, dtype=np.uint8)
    scaled = np.clip(height_above_ground / max_height_mm, 0.0, 1.0)
    image[mask] = (scaled[mask] * 255).astype(np.uint8)
    colored = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    colored[~mask] = 255
    readbin.imwrite_unicode(output_file, colored)


def crop_ground_removed_pointcloud(
    bin_file,
    image_file,
    json_file,
    output_dir,
    width=1280,
    height=720,
    camera="D435i",
    thresholds=(20, 30, 50),
    label=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_params = readbin.CAMERA_PRESETS[camera]
    depth = readbin.read_depth_bin(bin_file, width=width, height=height)
    color_bgr = readbin.read_color_aligned(image_file, depth.shape)
    label_mask = load_labelme_mask(json_file, width=width, height=height, labels=[label] if label else None) > 0
    valid_depth = np.isfinite(depth) & (depth > 0)

    # Keep a gap around the plant mask so plant edge pixels do not influence ground fitting.
    margin_kernel = np.ones((31, 31), dtype=np.uint8)
    expanded_label_mask = cv2.dilate(label_mask.astype(np.uint8), margin_kernel) > 0
    ground_candidate_mask = valid_depth & (~expanded_label_mask)

    x, y, z = depth_to_xyz(depth, camera_params)
    plane = fit_ground_plane_ransac(x, y, z, ground_candidate_mask)
    ground_z = plane["a"] * x + plane["b"] * y + plane["c"]
    height_above_ground = ground_z - z

    base_name = Path(json_file).stem
    readbin.imwrite_unicode(output_dir / f"{base_name}_label_mask.png", (label_mask.astype(np.uint8) * 255))
    save_masked_rgb(color_bgr, label_mask, output_dir / f"{base_name}_label_rgb_preview.png")
    save_height_map(height_above_ground, label_mask & valid_depth, output_dir / f"{base_name}_height_above_ground.png")

    print("Ground plane z = a*x + b*y + c")
    print(f"a={plane['a']:.8f}, b={plane['b']:.8f}, c={plane['c']:.3f}")
    print(
        "ground fit: "
        f"{plane['inlier_points']}/{plane['candidate_points']} inliers, "
        f"median residual={plane['median_abs_residual_mm']:.2f} mm"
    )
    print(f"label pixels: {int(np.count_nonzero(label_mask))}")
    print(f"valid depth pixels in label: {int(np.count_nonzero(label_mask & valid_depth))}")

    outputs = []
    for threshold in thresholds:
        keep_mask = label_mask & valid_depth & (height_above_ground > float(threshold))
        point_count = int(np.count_nonzero(keep_mask))
        if point_count == 0:
            print(f"threshold {threshold} mm: no points kept")
            continue

        point_cloud = build_point_cloud(x, y, z, color_bgr, keep_mask)
        prefix = f"{base_name}_wolf_region_ground_removed_h{int(threshold)}mm"
        ply_path = output_dir / f"{prefix}.ply"
        preview_path = output_dir / f"{prefix}_rgb_preview.png"
        topdown_path = output_dir / f"{prefix}_topdown.png"

        readbin.save_pointcloud_as_ply(point_cloud, ply_path)
        save_masked_rgb(color_bgr, keep_mask, preview_path)
        readbin.save_pointcloud_topdown_image(
            point_cloud,
            topdown_path,
            image_width=1200,
            image_height=900,
            percentile_clip=0.5,
            point_size=2,
        )

        kept_heights = height_above_ground[keep_mask]
        print(
            f"threshold {threshold} mm: {point_count} points, "
            f"height range=[{np.nanmin(kept_heights):.1f}, {np.nanmax(kept_heights):.1f}] mm, "
            f"H95={np.nanpercentile(kept_heights, 95):.1f} mm"
        )
        outputs.append(ply_path)

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Crop LabelMe plant point cloud after removing fitted ground points.")
    parser.add_argument("--bin", dest="bin_file", type=Path, default=Path("test_work/010.1.bin"))
    parser.add_argument("--image", dest="image_file", type=Path, default=Path("test_work/010.1.jpg"))
    parser.add_argument("--json", dest="json_file", type=Path, default=Path("test_work/010.1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_010_1_ground_removed"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera", choices=sorted(readbin.CAMERA_PRESETS), default="D435i")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[20, 30, 50])
    parser.add_argument("--label", default=None, help="Optional LabelMe label to keep, e.g. wolf_region.")
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = crop_ground_removed_pointcloud(
        bin_file=args.bin_file,
        image_file=args.image_file,
        json_file=args.json_file,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        camera=args.camera,
        thresholds=args.thresholds,
        label=args.label,
    )
    print("Generated PLY files:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
