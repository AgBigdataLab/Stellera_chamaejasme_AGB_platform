import argparse
import csv
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull, cKDTree, distance
from sklearn.cluster import DBSCAN

from . import depth_io as readbin
from . import pointcloud as crop


def ground_coordinates(points_xyz, plane):
    """Project XYZ points onto orthonormal ground axes and return signed height."""
    a, b, c = plane["a"], plane["b"], plane["c"]
    normal = np.array([a, b, -1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    axis_u = np.array([1.0, 0.0, a], dtype=np.float64)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)

    uv = np.column_stack((points_xyz @ axis_u, points_xyz @ axis_v))
    height = (a * points_xyz[:, 0] + b * points_xyz[:, 1] + c - points_xyz[:, 2]) / np.sqrt(
        a * a + b * b + 1.0
    )
    return uv, height


def statistical_outlier_mask(points_xyz, neighbors=20, std_ratio=2.0):
    """Apply the standard mean-neighbor-distance statistical outlier filter."""
    if points_xyz.shape[0] <= neighbors:
        return np.ones(points_xyz.shape[0], dtype=bool), np.nan

    tree = cKDTree(points_xyz)
    distances, _ = tree.query(points_xyz, k=neighbors + 1, workers=-1)
    mean_distance = np.mean(distances[:, 1:], axis=1)
    cutoff = float(np.mean(mean_distance) + std_ratio * np.std(mean_distance))
    return mean_distance <= cutoff, cutoff


def cluster_points(points_xyz, eps_mm=25.0, min_samples=10, min_cluster_points=200):
    labels = DBSCAN(eps=eps_mm, min_samples=min_samples, algorithm="kd_tree", n_jobs=-1).fit_predict(points_xyz)
    cluster_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    cluster_sizes = {int(cluster_id): int(count) for cluster_id, count in zip(cluster_ids, counts)}
    eligible = [(cluster_id, count) for cluster_id, count in cluster_sizes.items() if count >= min_cluster_points]
    if not eligible:
        raise ValueError(
            "No eligible plant cluster was found. Increase --cluster-eps, reduce "
            "--cluster-min-points, or provide --json to constrain the plant region."
        )
    main_cluster = max(eligible, key=lambda item: item[1])[0]
    return labels, main_cluster, cluster_sizes


def pca_extents(uv, lower=1.0, upper=99.0):
    centered = uv - np.mean(uv, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh.T
    limits = np.percentile(projected, [lower, upper], axis=0)
    extents = limits[1] - limits[0]
    return float(np.max(extents)), float(np.min(extents))


def feret_extents(uv):
    """Return maximum Feret diameter and its perpendicular crown width."""
    points = np.asarray(uv, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError("Feret extents require at least three 2D points.")
    hull = points[ConvexHull(points).vertices]
    pairwise = distance.squareform(distance.pdist(hull))
    first, second = np.unravel_index(np.argmax(pairwise), pairwise.shape)
    axis = hull[second] - hull[first]
    length = float(np.linalg.norm(axis))
    if length <= 0:
        raise ValueError("Feret diameter is zero.")
    perpendicular = np.array([-axis[1], axis[0]]) / length
    projected = hull @ perpendicular
    width = float(np.max(projected) - np.min(projected))
    return length, width


def robust_feret_extents(uv, lower=0.5, upper=99.5):
    """Trim marginal projection outliers, then calculate Feret length and width."""
    points = np.asarray(uv, dtype=np.float64)
    centered = points - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh.T
    low, high = np.percentile(projected, [lower, upper], axis=0)
    keep = np.all((projected >= low) & (projected <= high), axis=1)
    if np.count_nonzero(keep) < 3:
        raise ValueError("Too few points remain after robust Feret trimming.")
    return feret_extents(points[keep])


def _cross_2d(first, second):
    return float(first[0] * second[1] - first[1] * second[0])


def _line_hull_chord(center, direction, hull, tolerance=1e-9):
    """Return the two intersections of an infinite line with a convex polygon."""
    parameters = []
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        edge = end - start
        denominator = _cross_2d(direction, edge)
        if abs(denominator) <= tolerance:
            continue
        offset = start - center
        line_parameter = _cross_2d(offset, edge) / denominator
        edge_parameter = _cross_2d(offset, direction) / denominator
        if -tolerance <= edge_parameter <= 1.0 + tolerance:
            parameters.append(line_parameter)

    if len(parameters) < 2:
        raise ValueError("Centroid axis does not intersect the crown hull twice.")
    parameters = np.unique(np.round(parameters, decimals=10))
    lower = float(np.min(parameters))
    upper = float(np.max(parameters))
    if upper - lower <= tolerance:
        raise ValueError("Centroid chord has zero length.")
    endpoints = np.vstack((center + lower * direction, center + upper * direction))
    return endpoints, upper - lower


def centroid_chord_geometry(uv):
    """Measure orthogonal crown chords that both pass through the point centroid.

    The axes are the two PCA directions of the cleaned 2D canopy projection. Each
    reported extent is the chord bounded by the convex hull along a line through
    the centroid, rather than a support-line span away from the centroid.
    """
    points = np.asarray(uv, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError("Centroid-chord extents require at least three 2D points.")
    center = np.mean(points, axis=0)
    centered = points - center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size < 2 or singular_values[1] <= np.finfo(float).eps:
        raise ValueError("Crown projection is degenerate and has no 2D extent.")

    first_axis = vh[0].astype(np.float64)
    first_axis /= np.linalg.norm(first_axis)
    if first_axis[0] < 0 or (abs(first_axis[0]) <= 1e-12 and first_axis[1] < 0):
        first_axis *= -1.0
    second_axis = np.array([-first_axis[1], first_axis[0]], dtype=np.float64)
    hull = points[ConvexHull(points).vertices]
    first_endpoints, first_length = _line_hull_chord(
        center, first_axis, hull
    )
    second_endpoints, second_length = _line_hull_chord(
        center, second_axis, hull
    )

    if second_length > first_length:
        first_axis, second_axis = second_axis, -first_axis
        first_endpoints, second_endpoints = second_endpoints, first_endpoints
        first_length, second_length = second_length, first_length
    return {
        "center": center,
        "long_axis": first_axis,
        "short_axis": second_axis,
        "length_endpoints": first_endpoints,
        "width_endpoints": second_endpoints,
        "length": float(first_length),
        "width": float(second_length),
        "hull": hull,
        "method": "PCA_centroid_hull_chords",
    }


def centroid_chord_extents(uv):
    geometry = centroid_chord_geometry(uv)
    return geometry["length"], geometry["width"]


def occupied_grid_area(uv, cell_mm=5.0):
    cells = np.floor((uv - np.min(uv, axis=0)) / cell_mm).astype(np.int64)
    occupied = np.unique(cells, axis=0).shape[0]
    return float(occupied * cell_mm * cell_mm)


def projected_grid_areas(uv, cell_mm=5.0):
    """Return occupied and convex-envelope areas on the same cell grid."""
    if cell_mm <= 0:
        raise ValueError("Grid cell size must be positive.")
    points = np.asarray(uv, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
        raise ValueError("Projected grid areas require nonempty 2D points.")

    cells = np.floor((points - np.min(points, axis=0)) / cell_mm).astype(np.int32)
    occupied_cells = np.unique(cells, axis=0)
    cell_area = cell_mm * cell_mm
    occupied_area = float(len(occupied_cells) * cell_area)
    if len(occupied_cells) < 3:
        return occupied_area, np.nan

    hull = cv2.convexHull(occupied_cells)
    if cv2.contourArea(hull) <= 0:
        return occupied_area, np.nan
    grid = np.zeros((int(cells[:, 1].max()) + 1, int(cells[:, 0].max()) + 1), dtype=np.uint8)
    cv2.fillConvexPoly(grid, hull, 1)
    hull_area = float(np.count_nonzero(grid) * cell_area)
    return occupied_area, hull_area


def convex_hull_area(uv):
    if uv.shape[0] < 3:
        return 0.0
    hull = cv2.convexHull(uv.astype(np.float32))
    return float(cv2.contourArea(hull))


def compute_apparent_traits(uv, heights, grid_cell_mm=5.0):
    crown_length, crown_width = pca_extents(uv)
    crown_length_max, crown_width_max = pca_extents(uv, lower=0.0, upper=100.0)
    hull_area = convex_hull_area(uv)
    grid_area, grid_hull_area = projected_grid_areas(uv, grid_cell_mm)
    percentiles = np.percentile(heights, [10, 25, 50, 75, 90, 95, 99, 99.5, 99.9])
    h10, h25, h50, h75, h90, h95, h99, h99_5, h99_9 = [float(value) for value in percentiles]
    hmean = float(np.mean(heights))
    hstd = float(np.std(heights))

    layer_edges = np.array([0.0, 0.25, 0.50, 0.75, 1.0]) * h95
    layer_counts, _ = np.histogram(np.clip(heights, 0.0, h95), bins=layer_edges)
    layer_ratios = layer_counts / max(int(np.sum(layer_counts)), 1)

    return {
        "point_count": int(heights.size),
        "crown_length_pca_p98_mm": crown_length,
        "crown_width_pca_p98_mm": crown_width,
        "crown_length_pca_max_mm": crown_length_max,
        "crown_width_pca_max_mm": crown_width_max,
        "crown_length_width_ratio": crown_length / crown_width if crown_width > 0 else np.nan,
        "projected_convex_hull_area_mm2": hull_area,
        "projected_occupied_grid_area_mm2": grid_area,
        "projected_convex_hull_grid_area_mm2": grid_hull_area,
        "projected_fill_ratio": grid_area / grid_hull_area if grid_hull_area > 0 else np.nan,
        "hmax_mm": float(np.max(heights)),
        "h99_mm": h99,
        "h99_5_mm": h99_5,
        "h99_9_mm": h99_9,
        "h95_mm": h95,
        "h90_mm": h90,
        "h75_mm": h75,
        "h50_mm": h50,
        "h25_mm": h25,
        "h10_mm": h10,
        "hmean_mm": hmean,
        "hstd_mm": hstd,
        "height_cv": hstd / hmean if hmean > 0 else np.nan,
        "height_iqr_mm": h75 - h25,
        "layer_0_25_ratio": float(layer_ratios[0]),
        "layer_25_50_ratio": float(layer_ratios[1]),
        "layer_50_75_ratio": float(layer_ratios[2]),
        "layer_75_100_ratio": float(layer_ratios[3]),
        "apparent_volume_hull_h95_mm3": hull_area * h95,
        "apparent_volume_grid_h95_mm3": grid_area * h95,
    }


def select_local_ring_ground_candidates(
    depth,
    valid_depth,
    roi_mask,
    inner_radius_px=20,
    outer_radius_px=80,
    grid_size_px=24,
    deep_quantile=90.0,
):
    """Select locally deepest pixels around a plant as candidate ground returns."""
    if inner_radius_px < 0 or outer_radius_px <= inner_radius_px:
        raise ValueError("Ground-ring radii must satisfy 0 <= inner < outer.")

    inner_size = inner_radius_px * 2 + 1
    outer_size = outer_radius_px * 2 + 1
    # A rectangular image-space ring samples all four directions around the crown,
    # including the diagonal ground patches visible in oblique camera views.
    inner_kernel = np.ones((inner_size, inner_size), dtype=np.uint8)
    outer_kernel = np.ones((outer_size, outer_size), dtype=np.uint8)
    inner = cv2.dilate(roi_mask.astype(np.uint8), inner_kernel) > 0
    outer = cv2.dilate(roi_mask.astype(np.uint8), outer_kernel) > 0
    ring = valid_depth & outer & (~inner)
    candidates = np.zeros(depth.shape, dtype=bool)

    ys, xs = np.nonzero(ring)
    if xs.size == 0:
        return candidates, ring
    x_start = max((int(xs.min()) // grid_size_px) * grid_size_px, 0)
    y_start = max((int(ys.min()) // grid_size_px) * grid_size_px, 0)
    x_stop = min(int(xs.max()) + grid_size_px + 1, depth.shape[1])
    y_stop = min(int(ys.max()) + grid_size_px + 1, depth.shape[0])

    for y0 in range(y_start, y_stop, grid_size_px):
        for x0 in range(x_start, x_stop, grid_size_px):
            y1 = min(y0 + grid_size_px, depth.shape[0])
            x1 = min(x0 + grid_size_px, depth.shape[1])
            cell_ring = ring[y0:y1, x0:x1]
            cell_depth = depth[y0:y1, x0:x1]
            values = cell_depth[cell_ring]
            if values.size < 8:
                continue
            cutoff = float(np.nanpercentile(values, deep_quantile))
            candidates[y0:y1, x0:x1] |= cell_ring & (cell_depth >= cutoff)
    return candidates, ring


def save_ground_candidate_preview(color_bgr, ring_mask, candidate_mask, output_file):
    preview = (color_bgr.astype(np.float32) * 0.45).astype(np.uint8)
    preview[ring_mask] = (0.55 * preview[ring_mask] + 0.45 * np.array([255, 120, 0])).astype(np.uint8)
    preview[candidate_mask] = (0, 255, 255)
    readbin.imwrite_unicode(output_file, preview)


def save_pixel_preview(color_bgr, pixel_indices, output_file):
    mask = np.zeros(color_bgr.shape[:2], dtype=bool)
    mask.ravel()[pixel_indices] = True
    crop.save_masked_rgb(color_bgr, mask, output_file)


def save_cluster_preview(color_bgr, pixel_indices, labels, main_cluster, output_file):
    preview = np.full_like(color_bgr, 255)
    palette = np.array(
        [[0, 0, 255], [0, 170, 0], [255, 0, 0], [0, 165, 255], [180, 0, 180], [255, 180, 0]],
        dtype=np.uint8,
    )
    flat = preview.reshape(-1, 3)
    for cluster_id in np.unique(labels):
        if cluster_id < 0:
            continue
        color = np.array([0, 0, 0], dtype=np.uint8) if cluster_id == main_cluster else palette[cluster_id % len(palette)]
        flat[pixel_indices[labels == cluster_id]] = color
    readbin.imwrite_unicode(output_file, preview)


def compute_depth_gradient(depth, valid_depth):
    """Estimate local depth change in mm/pixel while ignoring invalid neighbors."""
    gradient = np.zeros(depth.shape, dtype=np.float32)
    horizontal_valid = valid_depth[:, 2:] & valid_depth[:, :-2]
    horizontal = np.zeros(depth.shape, dtype=np.float32)
    horizontal[:, 1:-1][horizontal_valid] = (
        np.abs(depth[:, 2:][horizontal_valid] - depth[:, :-2][horizontal_valid]) / 2.0
    )
    vertical_valid = valid_depth[2:, :] & valid_depth[:-2, :]
    vertical = np.zeros(depth.shape, dtype=np.float32)
    vertical[1:-1, :][vertical_valid] = (
        np.abs(depth[2:, :][vertical_valid] - depth[:-2, :][vertical_valid]) / 2.0
    )
    np.maximum(horizontal, vertical, out=gradient)
    return gradient


def estimate_rgb_depth_alignment(
    depth,
    color_bgr,
    valid_depth,
    roi_mask,
    output_dir,
    sample_id,
    depth_edge_threshold_mm=12.0,
    max_shift_px=8,
):
    """Create a diagnostic edge overlay and estimate residual integer pixel shift.

    This cannot replace intrinsic/extrinsic calibration. It only tests whether the
    saved depth frame appears to have already been aligned to the RGB frame.
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    rgb_edges = cv2.Canny(gray, 70, 150) > 0
    depth_gradient = compute_depth_gradient(depth, valid_depth)
    depth_edges = valid_depth & (depth_gradient >= depth_edge_threshold_mm)

    region = cv2.dilate(roi_mask.astype(np.uint8), np.ones((31, 31), dtype=np.uint8)) > 0
    rgb_edges &= region
    depth_edges &= region
    edge_y, edge_x = np.nonzero(depth_edges)
    if edge_x.size == 0 or not np.any(rgb_edges):
        raise ValueError("Not enough RGB/depth edges for alignment diagnostics.")

    # Distance from every pixel to its nearest RGB edge.
    distance_to_rgb = cv2.distanceTransform((~rgb_edges).astype(np.uint8), cv2.DIST_L2, 3)
    if edge_x.size > 50000:
        select = np.linspace(0, edge_x.size - 1, 50000, dtype=np.int64)
        edge_x = edge_x[select]
        edge_y = edge_y[select]

    scores = []
    image_height, image_width = depth.shape
    for dy in range(-max_shift_px, max_shift_px + 1):
        for dx in range(-max_shift_px, max_shift_px + 1):
            shifted_x = edge_x + dx
            shifted_y = edge_y + dy
            inside = (
                (shifted_x >= 0)
                & (shifted_x < image_width)
                & (shifted_y >= 0)
                & (shifted_y < image_height)
            )
            distances = distance_to_rgb[shifted_y[inside], shifted_x[inside]]
            score = float(np.mean(np.minimum(distances, 10.0)))
            scores.append((score, dx, dy))

    best_score, best_dx, best_dy = min(scores)
    zero_distances = distance_to_rgb[edge_y, edge_x]
    alignment = {
        "diagnostic_only": True,
        "mapping_mode": "direct_pixel_mapping_assuming_depth_already_aligned_to_rgb",
        "calibration_available": False,
        "depth_edge_threshold_mm_per_pixel": depth_edge_threshold_mm,
        "rgb_edge_count_in_roi": int(np.count_nonzero(rgb_edges)),
        "depth_edge_count_in_roi": int(np.count_nonzero(depth_edges)),
        "zero_shift_mean_edge_distance_px": float(np.mean(zero_distances)),
        "zero_shift_median_edge_distance_px": float(np.median(zero_distances)),
        "zero_shift_p90_edge_distance_px": float(np.percentile(zero_distances, 90)),
        "zero_shift_within_2px_ratio": float(np.mean(zero_distances <= 2.0)),
        "best_residual_shift_dx_px": int(best_dx),
        "best_residual_shift_dy_px": int(best_dy),
        "best_clipped_mean_edge_distance_px": best_score,
        "interpretation": (
            "A small best shift supports, but does not prove, prior RGB-depth alignment. "
            "Factory intrinsics, distortion, and depth-to-color extrinsics are still required "
            "for metric validation."
        ),
    }

    overlay = (color_bgr.astype(np.float32) * 0.55).astype(np.uint8)
    rgb_dilated = cv2.dilate(rgb_edges.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
    overlap = depth_edges & rgb_dilated
    overlay[rgb_edges] = (255, 255, 0)
    overlay[depth_edges] = (0, 0, 255)
    overlay[overlap] = (0, 255, 255)
    readbin.imwrite_unicode(output_dir / f"{sample_id}_rgb_depth_edge_alignment.png", overlay)

    mask_core = cv2.erode(roi_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) > 0
    mask_boundary = roi_mask & (~mask_core)
    confidence = (color_bgr.astype(np.float32) * 0.45).astype(np.uint8)
    confidence[mask_core] = (0.35 * confidence[mask_core] + 0.65 * np.array([0, 190, 0])).astype(np.uint8)
    confidence[mask_boundary] = (0, 215, 255)
    readbin.imwrite_unicode(output_dir / f"{sample_id}_mask_confidence_zones.png", confidence)
    return alignment, depth_gradient


def process_rgbd_plant(
    bin_file,
    image_file,
    output_dir,
    json_file=None,
    label=None,
    width=1280,
    height=720,
    camera="D435i",
    min_depth_mm=300.0,
    max_depth_mm=3000.0,
    min_height_mm=20.0,
    max_height_mm=1000.0,
    sor_neighbors=20,
    sor_std_ratio=2.0,
    cluster_eps_mm=25.0,
    cluster_min_samples=10,
    cluster_min_points=200,
    grid_cell_mm=5.0,
    ground_method="global_ransac",
    ground_ring_inner_px=20,
    ground_ring_outer_px=80,
    ground_grid_size_px=24,
    ground_deep_quantile=90.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_id = Path(bin_file).stem

    depth = readbin.read_depth_bin(bin_file, width=width, height=height)
    color_bgr = readbin.read_color_aligned(image_file, depth.shape)
    x, y, z = crop.depth_to_xyz(depth, readbin.CAMERA_PRESETS[camera])

    valid = np.isfinite(depth) & (depth >= min_depth_mm) & (depth <= max_depth_mm)
    roi_mask = np.ones(depth.shape, dtype=bool)
    if json_file is not None:
        roi_mask = crop.load_labelme_mask(json_file, width=width, height=height, labels=[label] if label else None) > 0
        expanded = cv2.dilate(roi_mask.astype(np.uint8), np.ones((31, 31), dtype=np.uint8)) > 0
        global_ground_candidates = valid & (~expanded)
    else:
        warnings.warn(
            "No RGB/LabelMe plant ROI was provided. Largest-cluster selection is only reliable "
            "for a controlled scene with one isolated plant.",
            stacklevel=2,
        )
        global_ground_candidates = valid

    alignment_qc, depth_gradient = estimate_rgb_depth_alignment(
        depth,
        color_bgr,
        valid,
        roi_mask,
        output_dir,
        sample_id,
    )

    ground_ring = np.zeros(depth.shape, dtype=bool)
    if ground_method == "local_ring_grid":
        if json_file is None:
            raise ValueError("--ground-method local_ring_grid requires a plant ROI from --json.")
        ground_candidates, ground_ring = select_local_ring_ground_candidates(
            depth,
            valid,
            roi_mask,
            inner_radius_px=ground_ring_inner_px,
            outer_radius_px=ground_ring_outer_px,
            grid_size_px=ground_grid_size_px,
            deep_quantile=ground_deep_quantile,
        )
        if np.count_nonzero(ground_candidates) < 100:
            raise ValueError("Not enough local ring ground candidates.")
        plane = crop.fit_ground_plane_ransac(x, y, z, ground_candidates, inlier_mm=15.0)
        save_ground_candidate_preview(
            color_bgr,
            ground_ring,
            ground_candidates,
            output_dir / f"{sample_id}_local_ground_candidates.png",
        )
    else:
        ground_candidates = global_ground_candidates
        plane = crop.fit_ground_plane_ransac(x, y, z, ground_candidates)
    all_xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    _, all_heights = ground_coordinates(all_xyz, plane)
    height_map = all_heights.reshape(depth.shape)

    height_keep = (
        valid
        & roi_mask
        & (height_map >= min_height_mm)
        & (height_map <= max_height_mm)
    )
    pixel_indices = np.flatnonzero(height_keep.ravel())
    if pixel_indices.size < cluster_min_points:
        raise ValueError(f"Only {pixel_indices.size} points remain after depth/height filtering.")

    points_xyz = all_xyz[pixel_indices]
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)[pixel_indices]
    sor_keep, sor_cutoff = statistical_outlier_mask(points_xyz, sor_neighbors, sor_std_ratio)
    points_xyz = points_xyz[sor_keep]
    color_rgb = color_rgb[sor_keep]
    pixel_indices = pixel_indices[sor_keep]

    labels, main_cluster, cluster_sizes = cluster_points(
        points_xyz,
        eps_mm=cluster_eps_mm,
        min_samples=cluster_min_samples,
        min_cluster_points=cluster_min_points,
    )
    main_keep = labels == main_cluster
    plant_xyz = points_xyz[main_keep]
    plant_rgb = color_rgb[main_keep]
    plant_pixels = pixel_indices[main_keep]
    plant_uv, plant_heights = ground_coordinates(plant_xyz, plane)

    traits = compute_apparent_traits(plant_uv, plant_heights, grid_cell_mm=grid_cell_mm)
    traits = {
        "sample_id": sample_id,
        **traits,
        "depth_range_point_count": int(np.count_nonzero(valid)),
        "height_filter_point_count": int(np.count_nonzero(height_keep)),
        "sor_point_count": int(points_xyz.shape[0]),
        "sor_mean_distance_cutoff_mm": sor_cutoff,
        "cluster_count": int(len(cluster_sizes)),
        "main_cluster_id": int(main_cluster),
        "main_cluster_retention_ratio": float(plant_xyz.shape[0] / max(points_xyz.shape[0], 1)),
        "ground_fit_inlier_points": plane["inlier_points"],
        "ground_fit_candidate_points": plane["candidate_points"],
        "ground_fit_median_abs_residual_mm": plane["median_abs_residual_mm"],
        "valid_depth_ratio_in_roi": float(np.count_nonzero(valid & roi_mask) / max(np.count_nonzero(roi_mask), 1)),
        "depth_edge_ratio_in_roi": float(
            np.count_nonzero((depth_gradient >= 12.0) & valid & roi_mask) / max(np.count_nonzero(valid & roi_mask), 1)
        ),
    }

    point_cloud = np.column_stack((plant_xyz, plant_rgb)).astype(np.float64)
    readbin.save_pointcloud_as_ply(point_cloud, output_dir / f"{sample_id}_plant_main_cluster.ply")
    save_pixel_preview(color_bgr, plant_pixels, output_dir / f"{sample_id}_plant_rgb_preview.png")
    save_cluster_preview(color_bgr, pixel_indices, labels, main_cluster, output_dir / f"{sample_id}_clusters.png")
    crop.save_height_map(height_map, height_keep, output_dir / f"{sample_id}_height_map.png", max_height_mm=max_height_mm)

    csv_path = output_dir / f"{sample_id}_traits.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(traits))
        writer.writeheader()
        writer.writerow(traits)

    qc = {
        "parameters": {
            "camera": camera,
            "min_depth_mm": min_depth_mm,
            "max_depth_mm": max_depth_mm,
            "min_height_mm": min_height_mm,
            "max_height_mm": max_height_mm,
            "sor_neighbors": sor_neighbors,
            "sor_std_ratio": sor_std_ratio,
            "cluster_eps_mm": cluster_eps_mm,
            "cluster_min_samples": cluster_min_samples,
            "cluster_min_points": cluster_min_points,
            "grid_cell_mm": grid_cell_mm,
            "ground_method": ground_method,
            "ground_ring_inner_px": ground_ring_inner_px,
            "ground_ring_outer_px": ground_ring_outer_px,
            "ground_grid_size_px": ground_grid_size_px,
            "ground_deep_quantile": ground_deep_quantile,
            "labelme_roi": str(json_file) if json_file else None,
            "label": label,
        },
        "ground_plane": plane,
        "rgb_depth_alignment_diagnostic": alignment_qc,
        "cluster_sizes": cluster_sizes,
        "traits": traits,
    }
    qc_path = output_dir / f"{sample_id}_processing_qc.json"
    with qc_path.open("w", encoding="utf-8") as file:
        json.dump(qc, file, ensure_ascii=False, indent=2)

    return traits, csv_path, qc_path


def parse_args():
    parser = argparse.ArgumentParser(description="Process one RGB-D plant through filtering, ground removal, clustering, and trait extraction.")
    parser.add_argument("--bin", dest="bin_file", type=Path, required=True)
    parser.add_argument("--image", dest="image_file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json", dest="json_file", type=Path, default=None, help="Optional LabelMe ROI for robust plant isolation.")
    parser.add_argument("--label", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera", choices=sorted(readbin.CAMERA_PRESETS), default="D435i")
    parser.add_argument("--min-depth", type=float, default=300.0)
    parser.add_argument("--max-depth", type=float, default=3000.0)
    parser.add_argument("--min-height", type=float, default=20.0)
    parser.add_argument("--max-height", type=float, default=1000.0)
    parser.add_argument("--sor-neighbors", type=int, default=20)
    parser.add_argument("--sor-std-ratio", type=float, default=2.0)
    parser.add_argument("--cluster-eps", type=float, default=25.0)
    parser.add_argument("--cluster-min-samples", type=int, default=10)
    parser.add_argument("--cluster-min-points", type=int, default=200)
    parser.add_argument("--grid-cell", type=float, default=5.0)
    parser.add_argument("--ground-method", choices=["global_ransac", "local_ring_grid"], default="global_ransac")
    parser.add_argument("--ground-ring-inner", type=int, default=20)
    parser.add_argument("--ground-ring-outer", type=int, default=80)
    parser.add_argument("--ground-grid-size", type=int, default=24)
    parser.add_argument("--ground-deep-quantile", type=float, default=90.0)
    return parser.parse_args()


def main():
    args = parse_args()
    traits, csv_path, qc_path = process_rgbd_plant(
        bin_file=args.bin_file,
        image_file=args.image_file,
        output_dir=args.output_dir,
        json_file=args.json_file,
        label=args.label,
        width=args.width,
        height=args.height,
        camera=args.camera,
        min_depth_mm=args.min_depth,
        max_depth_mm=args.max_depth,
        min_height_mm=args.min_height,
        max_height_mm=args.max_height,
        sor_neighbors=args.sor_neighbors,
        sor_std_ratio=args.sor_std_ratio,
        cluster_eps_mm=args.cluster_eps,
        cluster_min_samples=args.cluster_min_samples,
        cluster_min_points=args.cluster_min_points,
        grid_cell_mm=args.grid_cell,
        ground_method=args.ground_method,
        ground_ring_inner_px=args.ground_ring_inner,
        ground_ring_outer_px=args.ground_ring_outer,
        ground_grid_size_px=args.ground_grid_size,
        ground_deep_quantile=args.ground_deep_quantile,
    )
    print(f"Saved traits: {csv_path}")
    print(f"Saved quality control: {qc_path}")
    print(
        f"Plant points={traits['point_count']}, H95={traits['h95_mm']:.2f} mm, "
        f"crown={traits['crown_length_pca_p98_mm']:.2f} x {traits['crown_width_pca_p98_mm']:.2f} mm"
    )


if __name__ == "__main__":
    main()
