"""批量提取人工确认单株的 RGB-D 株高、冠幅长和冠幅宽。

输入为“预测图像数据集”中的同名 JPG/BIN，以及“prediction”中的
``*.target.txt``。直接在 VS Code 中运行本文件，不需要终端参数。

方法定义：
1. 将人工确认的 YOLO 分割多边形映射到同分辨率深度图；
2. 按目标掩膜尺度生成自适应局部环带，提取候选点并用RANSAC拟合地面；
3. 去除高度范围外点和统计离群点；
4. 以局部深地面候选点残差P90校正土面，株高取校正点云的H99.9；
5. 冠幅使用离地20 mm以上点云，经PCA坐标稳健裁剪后，计算穿过投影
   点质心的PCA主轴/副轴与凸包边界形成的两条正交弦；
6. 默认输出未经人工测量校准的算法性状，直接供最终TabPFN模型使用。

注意：脚本沿用项目之前的直接像素映射假设，即保存的 RGB 与深度已经对齐。
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from . import depth_io as readbin
from . import pointcloud
from . import rgbd_processing as pipeline


# =====================================================================
#                         性状提取参数配置区
# =====================================================================

# 输入数据：同一编号必须同时存在 JPG、BIN 和 target.txt。
RGBD_DIR = PROJECT_ROOT / "examples" / "rgbd"
TARGET_LABEL_DIR = PROJECT_ROOT / "output" / "segmentation"

# 输出只生成一个汇总 CSV。质控图默认关闭，避免产生大量中间文件。
OUTPUT_DIR = PROJECT_ROOT / "output" / "traits"
OUTPUT_CSV = OUTPUT_DIR / "target_rgbd_traits.csv"
EXTENDED_OUTPUT_CSV = OUTPUT_DIR / "target_rgbd_extended_traits.csv"
CORE_OUTPUT_CSV = OUTPUT_DIR / "target_rgbd_core_traits.csv"
SAVE_QC_PREVIEWS = False
QC_PREVIEW_DIR = OUTPUT_DIR / "qc_previews"

# 相机与图像设置。当前 BIN 为 1280x720 uint32 深度，单位为毫米。
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CAMERA_PRESET = "D455"

# 只使用该深度范围内的有效点。
MIN_DEPTH_MM = 300.0
MAX_DEPTH_MM = 3000.0

# 地面归一化后保留的植株高度范围。
MIN_PLANT_HEIGHT_MM = 15.0
MAX_PLANT_HEIGHT_MM = 1000.0

# 冠幅边界阈值和稳健裁剪参数。
MIN_CROWN_HEIGHT_MM = 20.0
CROWN_TRIM_LOWER_PERCENTILE = 0.5
CROWN_TRIM_UPPER_PERCENTILE = 99.5

# 地点A 200株测量校准：人工实测 = scale * 质心弦原始值 + offset。
# 方法改变时必须重新拟合，禁止复用旧Feret系数。
CROWN_METHOD = "H20_PCA_P0.5_P99.5_centroid_hull_chords"
CROWN_CALIBRATION_JSON = PROJECT_ROOT / "weights" / "site_a_crown_calibration.json"
CROWN_CALIBRATION_MODE = os.environ.get("CROWN_CALIBRATION_MODE", "raw").lower()
CROWN_LENGTH_SCALE = 1.0
CROWN_LENGTH_OFFSET_CM = 0.0
CROWN_WIDTH_SCALE = 1.0
CROWN_WIDTH_OFFSET_CM = 0.0

# 局部地面环带参数。半径按掩膜外接框最大边长自适应计算。
# 15%/75%由地点A 200株的预设比例网格和5折稳定性评价锁定；
# 地点B 50株仅在锁定后做一次外部验证。上下限仅防止极端掩膜失控。
GROUND_INNER_RADIUS_RATIO = 0.15
GROUND_OUTER_RADIUS_RATIO = 0.60
GROUND_INNER_RADIUS_MIN_PX = 8
GROUND_INNER_RADIUS_MAX_PX = 48
GROUND_OUTER_RADIUS_MIN_PX = 30
GROUND_OUTER_RADIUS_MAX_PX = 160
GROUND_GRID_SIZE_PX = 24
GROUND_DEEP_QUANTILE = 90.0
GROUND_RANSAC_INLIER_MM = 15.0

# 统计离群点去除参数。
SOR_NEIGHBORS = 20
SOR_STD_RATIO = 2.0
MIN_REMAINING_POINTS = 100

# 调试时可设为正整数，例如5；None表示处理全部250株。
MAX_SAMPLES: int | None = None

# =====================================================================


CSV_FIELDS = [
    "编号",
    "株高(cm)",
    "冠幅长(cm)",
    "冠幅宽(cm)",
]

EXTENDED_CSV_FIELDS = [
    "编号",
    "株高(cm)",
    "冠幅长(cm)",
    "冠幅宽(cm)",
    "高度P10(cm)",
    "高度P25(cm)",
    "高度P50(cm)",
    "高度P75(cm)",
    "高度P90(cm)",
    "高度P95(cm)",
    "高度P99(cm)",
    "高度均值(cm)",
    "高度标准差(cm)",
    "高度四分位距(cm)",
    "高度变异系数",
    "冠幅平均直径(cm)",
    "冠幅长宽比",
    "冠层椭圆面积(cm2)",
    "点云投影凸包面积(cm2)",
    "点云投影栅格凸包面积(cm2)",
    "点云投影占据面积(cm2)",
    "冠层投影填充率",
    "冠层点密度(点/cm2)",
    "椭球体积代理(cm3)",
    "冠层面积株高代理(cm3)",
    "冠层高度模型体积(cm3)",
    "三维凸包体积(cm3)",
    "体素占据体积指数(cm3)",
    "分层凸包体积(cm3)",
    "深度校正可见投影面积(cm2)",
    "高度加权可见体量(cm3)",
    "可见面积冠层密度",
    "归一化ExG均值",
    "RGBVI均值",
    "绿色点比例(ExG大于0)",
    "绿色可见投影面积(cm2)",
    "冠层栅格顶高变异系数",
    "上半冠层体积占比",
    "垂直层0-25%点比例",
    "垂直层25-50%点比例",
    "垂直层50-75%点比例",
    "垂直层75-100%点比例",
    "原始质心弦冠幅长(cm)",
    "原始质心弦冠幅宽(cm)",
    "有效深度比例",
    "高度点数",
    "冠层点数",
    "地面拟合中位绝对残差(mm)",
    "掩膜特征长度(px)",
    "地面内环半径(px)",
    "地面外环半径(px)",
    "质控需复核",
    "质控标记",
]

# 正式生物量建模使用的低冗余性状。扩展表仅作为候选性状库和质控记录。
CORE_CSV_FIELDS = [
    "编号",
    "株高(cm)",
    "点云投影凸包面积(cm2)",
    "冠幅长宽比",
    "高度变异系数",
    "冠层投影填充率",
    "质控需复核",
    "质控标记",
]


def natural_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def configure_crown_calibration() -> None:
    """Load calibration only when it was fitted for the current crown method."""
    global CROWN_LENGTH_SCALE, CROWN_LENGTH_OFFSET_CM
    global CROWN_WIDTH_SCALE, CROWN_WIDTH_OFFSET_CM

    if CROWN_CALIBRATION_MODE == "raw":
        print("冠幅校准：原始模式（用于重拟合质心弦校准参数）")
        return
    if CROWN_CALIBRATION_MODE != "required":
        raise ValueError("CROWN_CALIBRATION_MODE只能是required或raw")
    if not CROWN_CALIBRATION_JSON.is_file():
        raise FileNotFoundError(f"缺少冠幅校准文件：{CROWN_CALIBRATION_JSON}")
    calibration = json.loads(CROWN_CALIBRATION_JSON.read_text(encoding="utf-8-sig"))
    if calibration.get("method") != CROWN_METHOD:
        raise ValueError(
            "冠幅校准方法与当前算法不一致；请先使用"
            "CROWN_CALIBRATION_MODE=raw提取，再运行analyze_traits_r2_by_site.py重拟合"
        )
    traits = calibration.get("traits", {})
    length = traits.get("冠幅长(cm)", {})
    width = traits.get("冠幅宽(cm)", {})
    CROWN_LENGTH_SCALE = float(length["slope"])
    CROWN_LENGTH_OFFSET_CM = float(length["intercept_cm"])
    CROWN_WIDTH_SCALE = float(width["slope"])
    CROWN_WIDTH_OFFSET_CM = float(width["intercept_cm"])
    print(
        "冠幅校准：已加载地点A质心弦参数 "
        f"L={CROWN_LENGTH_SCALE:.6f}x+{CROWN_LENGTH_OFFSET_CM:.6f}, "
        f"W={CROWN_WIDTH_SCALE:.6f}x+{CROWN_WIDTH_OFFSET_CM:.6f}"
    )


def robust_trimmed_crown_points(uv: np.ndarray) -> np.ndarray:
    """按PCA坐标裁剪边缘点，和质心弦计算使用同一批冠层点。"""
    return uv[robust_crown_mask(uv)]


def robust_crown_mask(uv: np.ndarray) -> np.ndarray:
    """返回PCA稳健边界内的冠层点掩膜。"""
    centered = uv - np.mean(uv, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh.T
    low, high = np.percentile(
        projected,
        [CROWN_TRIM_LOWER_PERCENTILE, CROWN_TRIM_UPPER_PERCENTILE],
        axis=0,
    )
    keep = np.all((projected >= low) & (projected <= high), axis=1)
    if np.count_nonzero(keep) < 3:
        raise ValueError("冠层稳健裁剪后点数少于3")
    return keep


def canopy_height_model_traits(
    uv: np.ndarray,
    heights_mm: np.ndarray,
    cell_mm: float = 5.0,
) -> tuple[float, float, float]:
    """计算单视角2.5D冠层高度模型的体量和垂直结构性状。"""
    if uv.shape[0] == 0 or heights_mm.shape[0] != uv.shape[0]:
        return np.nan, np.nan, np.nan
    origin = np.min(uv, axis=0)
    grid = np.floor((uv - origin) / cell_mm).astype(np.int64)
    _, inverse = np.unique(grid, axis=0, return_inverse=True)
    top_heights = np.full(int(inverse.max()) + 1, -np.inf, dtype=float)
    np.maximum.at(top_heights, inverse, heights_mm)
    top_heights = top_heights[np.isfinite(top_heights) & (top_heights > 0.0)]
    if top_heights.size == 0:
        return np.nan, np.nan, np.nan

    # 每格以顶高乘5 mm x 5 mm面积积分；mm3换算为cm3。
    volume_cm3 = float(np.sum(top_heights) * cell_mm**2 / 1000.0)
    mean_top = float(np.mean(top_heights))
    top_height_cv = (
        float(np.std(top_heights) / mean_top) if mean_top > 0 else np.nan
    )
    half_height = 0.5 * float(np.percentile(top_heights, 99.9))
    total_volume = float(np.sum(top_heights))
    upper_volume_ratio = (
        float(np.sum(top_heights[top_heights >= half_height]) / total_volume)
        if total_volume > 0
        else np.nan
    )
    return volume_cm3, top_height_cv, upper_volume_ratio


def pointcloud_volume_traits(
    uv: np.ndarray,
    heights_mm: np.ndarray,
    voxel_mm: float = 5.0,
    slice_mm: float = 10.0,
) -> tuple[float, float, float]:
    """计算三维凸包、体素占据量和分层凸包积分体积。"""
    coordinates = np.column_stack((uv, heights_mm))
    origin = np.min(coordinates, axis=0)
    voxel_index = np.floor((coordinates - origin) / voxel_mm).astype(np.int64)
    unique_voxels = np.unique(voxel_index, axis=0)
    voxel_volume_index_cm3 = float(
        unique_voxels.shape[0] * voxel_mm**3 / 1000.0
    )
    voxel_centers = origin + (unique_voxels.astype(float) + 0.5) * voxel_mm

    hull_volume_cm3 = np.nan
    if voxel_centers.shape[0] >= 4:
        try:
            hull_volume_cm3 = float(ConvexHull(voxel_centers).volume / 1000.0)
        except QhullError:
            pass

    upper = float(np.percentile(heights_mm, 99.9))
    layered_volume_mm3 = 0.0
    for lower in np.arange(0.0, upper, slice_mm):
        keep = (heights_mm >= lower) & (heights_mm < lower + slice_mm)
        if np.count_nonzero(keep) >= 3:
            layered_volume_mm3 += (
                pipeline.convex_hull_area(uv[keep]) * slice_mm
            )
    layered_hull_volume_cm3 = float(layered_volume_mm3 / 1000.0)
    return hull_volume_cm3, voxel_volume_index_cm3, layered_hull_volume_cm3


def parse_yolo_polygon_line(
    raw_line: str, width: int, height: int
) -> tuple[np.ndarray, float | None]:
    """解析一行带可选置信度的 YOLO 分割标签。"""
    tokens = raw_line.strip().split()
    if len(tokens) < 7:
        raise ValueError("YOLO分割标签坐标不足")
    try:
        float(tokens[0])
        values = [float(value) for value in tokens[1:]]
    except ValueError as error:
        raise ValueError("YOLO分割标签包含非数值内容") from error

    confidence = values.pop() if len(values) % 2 == 1 else None
    if len(values) < 6 or len(values) % 2:
        raise ValueError("YOLO分割标签的多边形坐标数量无效")

    points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] * width, 0, width - 1)
    points[:, 1] = np.clip(points[:, 1] * height, 0, height - 1)
    return np.rint(points).astype(np.int32), confidence


def load_target_mask(
    label_path: Path, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, float | None]:
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise ValueError(
            f"目标标签必须且只能包含一个实例，当前为{len(lines)}个：{label_path.name}"
        )
    polygon, confidence = parse_yolo_polygon_line(lines[0], width, height)
    binary_mask_path = label_path.with_name(
        label_path.name.removesuffix(".target.txt") + ".target.png"
    )
    if binary_mask_path.is_file():
        mask = readbin.imread_unicode(binary_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (height, width):
            raise ValueError(f"RTMDet目标掩膜尺寸无效：{binary_mask_path.name}")
        mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            polygon = max(contours, key=cv2.contourArea).reshape(-1, 2)
    else:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
    if not np.any(mask):
        raise ValueError(f"目标标签生成了空掩膜：{label_path.name}")
    return mask.astype(bool), polygon, confidence


def load_all_instance_mask(
    label_path: Path,
    target_mask: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """读取全部实例，用于防止周围狼毒点参与目标植株的地面拟合。"""
    binary_mask_path = label_path.with_name(label_path.stem + ".all.png")
    if binary_mask_path.is_file():
        union = readbin.imread_unicode(binary_mask_path, cv2.IMREAD_GRAYSCALE)
        if union is None or union.shape != (height, width):
            raise ValueError(f"RTMDet全实例掩膜尺寸无效：{binary_mask_path.name}")
        union = union > 0
        union[target_mask] = True
        return union
    if not label_path.is_file():
        return target_mask.copy()

    union = np.zeros((height, width), dtype=np.uint8)
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        polygon, _ = parse_yolo_polygon_line(raw_line, width, height)
        cv2.fillPoly(union, [polygon], 1)
    union[target_mask] = 1
    return union.astype(bool)


def fit_ground_plane(
    depth: np.ndarray,
    valid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    target_mask: np.ndarray,
    all_instance_mask: np.ndarray,
) -> tuple[dict, str, int]:
    """优先拟合目标附近地面，局部点不足时退回全图背景地面。"""
    mask_y, mask_x = np.nonzero(target_mask)
    if mask_x.size == 0:
        raise ValueError("目标掩膜为空，无法计算自适应地面环带")
    mask_width = int(mask_x.max() - mask_x.min() + 1)
    mask_height = int(mask_y.max() - mask_y.min() + 1)
    mask_scale_px = max(mask_width, mask_height)
    inner_radius_px = int(
        np.clip(
            round(mask_scale_px * GROUND_INNER_RADIUS_RATIO),
            GROUND_INNER_RADIUS_MIN_PX,
            GROUND_INNER_RADIUS_MAX_PX,
        )
    )
    outer_radius_px = int(
        np.clip(
            round(mask_scale_px * GROUND_OUTER_RADIUS_RATIO),
            GROUND_OUTER_RADIUS_MIN_PX,
            GROUND_OUTER_RADIUS_MAX_PX,
        )
    )
    outer_radius_px = max(outer_radius_px, inner_radius_px + 1)

    local_candidates, _ = pipeline.select_local_ring_ground_candidates(
        depth,
        valid,
        target_mask,
        inner_radius_px=inner_radius_px,
        outer_radius_px=outer_radius_px,
        grid_size_px=GROUND_GRID_SIZE_PX,
        deep_quantile=GROUND_DEEP_QUANTILE,
    )
    nearby_plants = cv2.dilate(
        all_instance_mask.astype(np.uint8),
        np.ones((7, 7), dtype=np.uint8),
    ) > 0
    local_candidates &= ~nearby_plants
    local_count = int(np.count_nonzero(local_candidates))

    if local_count >= 100:
        try:
            plane = pointcloud.fit_ground_plane_ransac(
                x,
                y,
                z,
                local_candidates,
                max_points=30000,
                iterations=200,
                inlier_mm=GROUND_RANSAC_INLIER_MM,
            )
            signed_residuals = z[local_candidates] - (
                plane["a"] * x[local_candidates]
                + plane["b"] * y[local_candidates]
                + plane["c"]
            )
            for quantile in (25, 50, 75, 90):
                plane[f"local_signed_residual_p{quantile}_mm"] = float(
                    np.percentile(signed_residuals, quantile)
                )
            plane["mask_scale_px"] = mask_scale_px
            plane["ground_inner_radius_px"] = inner_radius_px
            plane["ground_outer_radius_px"] = outer_radius_px
            return plane, "local_ring_ransac", local_count
        except ValueError:
            pass

    expanded_instances = cv2.dilate(
        all_instance_mask.astype(np.uint8),
        np.ones((31, 31), dtype=np.uint8),
    ) > 0
    global_candidates = valid & (~expanded_instances)
    plane = pointcloud.fit_ground_plane_ransac(
        x,
        y,
        z,
        global_candidates,
        max_points=50000,
        iterations=300,
        inlier_mm=20.0,
    )
    plane["mask_scale_px"] = mask_scale_px
    plane["ground_inner_radius_px"] = inner_radius_px
    plane["ground_outer_radius_px"] = outer_radius_px
    return plane, "global_background_fallback", local_count


def touches_border(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def save_qc_preview(
    image_bgr: np.ndarray,
    polygon: np.ndarray,
    final_mask: np.ndarray,
    output_path: Path,
    row: dict,
) -> None:
    overlay = image_bgr.copy()
    green = np.zeros_like(image_bgr)
    green[final_mask] = (0, 210, 0)
    overlay = cv2.addWeighted(overlay, 1.0, green, 0.45, 0)
    cv2.polylines(overlay, [polygon], True, (0, 255, 255), 2, cv2.LINE_AA)
    text = (
        f"H={row['plant_height_cm']:.1f} cm  "
        f"L={row['crown_length_cm']:.1f} cm  "
        f"W={row['crown_width_cm']:.1f} cm"
    )
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(
        overlay,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    readbin.imwrite_unicode(output_path, overlay)


def extract_one(
    instance_id: str,
    source_id: str,
) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    image_path = RGBD_DIR / f"{source_id}.jpg"
    depth_path = RGBD_DIR / f"{source_id}.bin"
    target_path = TARGET_LABEL_DIR / f"{instance_id}.target.txt"
    all_instances_path = TARGET_LABEL_DIR / f"{source_id}.txt"

    missing = [
        str(path)
        for path in (image_path, depth_path, target_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("缺少配对文件：" + "; ".join(missing))

    image_bgr = readbin.imread_unicode(image_path)
    if image_bgr is None:
        raise ValueError(f"无法读取RGB图像：{image_path}")
    if image_bgr.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(
            f"RGB尺寸为{image_bgr.shape[1]}x{image_bgr.shape[0]}，"
            f"预期{IMAGE_WIDTH}x{IMAGE_HEIGHT}：{image_path.name}"
        )

    target_mask, polygon, confidence = load_target_mask(
        target_path, IMAGE_WIDTH, IMAGE_HEIGHT
    )
    all_instance_mask = load_all_instance_mask(
        all_instances_path,
        target_mask,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )
    depth = readbin.read_depth_bin(
        depth_path,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        dtype=np.uint32,
    )
    valid = (
        np.isfinite(depth)
        & (depth >= MIN_DEPTH_MM)
        & (depth <= MAX_DEPTH_MM)
    )
    x, y, z = pointcloud.depth_to_xyz(
        depth, readbin.CAMERA_PRESETS[CAMERA_PRESET]
    )
    plane, ground_method, local_candidate_count = fit_ground_plane(
        depth,
        valid,
        x,
        y,
        z,
        target_mask,
        all_instance_mask,
    )

    # 深度候选点代表局部土面下包络。RANSAC平面可能落在周围草层表面，
    # 因此使用有符号残差P90将平面沿Z方向移到较深且仍稳健的局部土面。
    ground_z_shift_mm = max(
        float(plane.get("local_signed_residual_p90_mm", 0.0)), 0.0
    )
    measurement_plane = dict(plane)
    measurement_plane["c"] = plane["c"] + ground_z_shift_mm

    all_xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    _, all_heights = pipeline.ground_coordinates(all_xyz, measurement_plane)
    height_map = all_heights.reshape(depth.shape)
    height_keep = (
        valid
        & target_mask
        & (height_map >= MIN_PLANT_HEIGHT_MM)
        & (height_map <= MAX_PLANT_HEIGHT_MM)
    )
    pixel_indices = np.flatnonzero(height_keep.ravel())
    if pixel_indices.size < MIN_REMAINING_POINTS:
        raise ValueError(
            f"高度过滤后仅剩{pixel_indices.size}个点，少于{MIN_REMAINING_POINTS}"
        )

    points_before_sor = all_xyz[pixel_indices]
    _, heights_before_sor = pipeline.ground_coordinates(
        points_before_sor, measurement_plane
    )
    pre_sor_h99_mm, pre_sor_h99_5_mm, pre_sor_h99_9_mm = [
        float(value)
        for value in np.percentile(heights_before_sor, [99.0, 99.5, 99.9])
    ]
    pre_sor_hmax_mm = float(np.max(heights_before_sor))
    sor_keep, _ = pipeline.statistical_outlier_mask(
        points_before_sor,
        neighbors=min(SOR_NEIGHBORS, points_before_sor.shape[0] - 1),
        std_ratio=SOR_STD_RATIO,
    )
    points_xyz = points_before_sor[sor_keep]
    pixel_indices = pixel_indices[sor_keep]
    if points_xyz.shape[0] < MIN_REMAINING_POINTS:
        raise ValueError(
            f"统计离群点去除后仅剩{points_xyz.shape[0]}个点，少于{MIN_REMAINING_POINTS}"
        )

    _, plant_heights = pipeline.ground_coordinates(
        points_xyz, measurement_plane
    )
    h10_mm, h25_mm, h50_mm, h75_mm, h90_mm, h95_mm, h99_mm, h99_5_mm, h99_9_mm = [
        float(value)
        for value in np.percentile(
            plant_heights,
            [10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.5, 99.9],
        )
    ]
    hmax_mm = float(np.max(plant_heights))
    height_mean_mm = float(np.mean(plant_heights))
    height_std_mm = float(np.std(plant_heights))
    height_iqr_mm = h75_mm - h25_mm
    height_cv = height_std_mm / height_mean_mm if height_mean_mm > 0 else np.nan

    # 以H99.9为植株有效顶端，计算四个等高层的点比例。
    clipped_heights = np.clip(plant_heights, 0.0, h99_9_mm)
    layer_edges = np.linspace(0.0, h99_9_mm, 5)
    layer_counts, _ = np.histogram(clipped_heights, bins=layer_edges)
    layer_ratios = layer_counts / max(int(np.sum(layer_counts)), 1)

    # 冠幅单独使用更高的离地阈值，避免掩膜内部地面和低矮杂草扩大边界。
    crown_keep = (
        valid
        & target_mask
        & (height_map >= MIN_CROWN_HEIGHT_MM)
        & (height_map <= MAX_PLANT_HEIGHT_MM)
    )
    crown_indices = np.flatnonzero(crown_keep.ravel())
    if crown_indices.size < MIN_REMAINING_POINTS:
        raise ValueError(
            f"冠幅高度过滤后仅剩{crown_indices.size}个点，少于{MIN_REMAINING_POINTS}"
        )
    crown_points = all_xyz[crown_indices]
    crown_sor_keep, _ = pipeline.statistical_outlier_mask(
        crown_points,
        neighbors=min(SOR_NEIGHBORS, crown_points.shape[0] - 1),
        std_ratio=SOR_STD_RATIO,
    )
    crown_points = crown_points[crown_sor_keep]
    if crown_points.shape[0] < MIN_REMAINING_POINTS:
        raise ValueError(
            f"冠幅离群点去除后仅剩{crown_points.shape[0]}个点，少于{MIN_REMAINING_POINTS}"
        )
    crown_uv, crown_heights = pipeline.ground_coordinates(
        crown_points, measurement_plane
    )
    crown_trim_keep = robust_crown_mask(crown_uv)
    trimmed_crown_uv = crown_uv[crown_trim_keep]
    trimmed_crown_heights = crown_heights[crown_trim_keep]
    crown_geometry = pipeline.centroid_chord_geometry(trimmed_crown_uv)
    raw_crown_length_mm = crown_geometry["length"]
    raw_crown_width_mm = crown_geometry["width"]
    crown_hull_area_mm2 = pipeline.convex_hull_area(trimmed_crown_uv)
    crown_grid_area_mm2, crown_grid_hull_area_mm2 = pipeline.projected_grid_areas(
        trimmed_crown_uv, cell_mm=5.0
    )
    crown_fill_ratio = (
        crown_grid_area_mm2 / crown_grid_hull_area_mm2
        if crown_grid_hull_area_mm2 > 0
        else np.nan
    )
    raw_crown_length_cm = raw_crown_length_mm / 10.0
    raw_crown_width_cm = raw_crown_width_mm / 10.0
    crown_length_cm = (
        CROWN_LENGTH_SCALE * raw_crown_length_cm + CROWN_LENGTH_OFFSET_CM
    )
    crown_width_cm = CROWN_WIDTH_SCALE * raw_crown_width_cm + CROWN_WIDTH_OFFSET_CM
    plant_height_cm = h99_9_mm / 10.0
    mean_crown_diameter_cm = (crown_length_cm + crown_width_cm) / 2.0
    crown_aspect_ratio = (
        crown_length_cm / crown_width_cm if crown_width_cm > 0 else np.nan
    )
    crown_ellipse_area_cm2 = math.pi * crown_length_cm * crown_width_cm / 4.0
    crown_hull_area_cm2 = crown_hull_area_mm2 / 100.0
    crown_grid_hull_area_cm2 = crown_grid_hull_area_mm2 / 100.0
    crown_grid_area_cm2 = crown_grid_area_mm2 / 100.0
    crown_point_density_cm2 = (
        crown_points.shape[0] / crown_hull_area_cm2
        if crown_hull_area_cm2 > 0
        else np.nan
    )
    ellipsoid_volume_proxy_cm3 = (
        math.pi * crown_length_cm * crown_width_cm * plant_height_cm / 6.0
    )
    crown_area_height_proxy_cm3 = crown_ellipse_area_cm2 * plant_height_cm
    camera = readbin.CAMERA_PRESETS[CAMERA_PRESET]
    pixel_footprint_mm2 = (
        points_xyz[:, 2] ** 2 / (float(camera["fx"]) * float(camera["fy"]))
    )
    depth_corrected_visible_area_cm2 = float(
        np.sum(pixel_footprint_mm2) / 100.0
    )
    height_weighted_visible_volume_cm3 = float(
        np.sum(pixel_footprint_mm2 * plant_heights) / 1000.0
    )
    visible_area_canopy_density = (
        depth_corrected_visible_area_cm2 / crown_hull_area_cm2
        if crown_hull_area_cm2 > 0
        else np.nan
    )
    color_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    plant_rgb = color_rgb[pixel_indices].astype(float) / 255.0
    channel_sum = np.sum(plant_rgb, axis=1) + 1e-12
    red = plant_rgb[:, 0] / channel_sum
    green = plant_rgb[:, 1] / channel_sum
    blue = plant_rgb[:, 2] / channel_sum
    exg = 2.0 * green - red - blue
    rgbvi = (green**2 - red * blue) / (green**2 + red * blue + 1e-12)
    green_keep = exg > 0.0
    normalized_exg_mean = float(np.mean(exg))
    rgbvi_mean = float(np.mean(rgbvi))
    green_point_ratio = float(np.mean(green_keep))
    green_visible_projected_area_cm2 = float(
        np.sum(pixel_footprint_mm2[green_keep]) / 100.0
    )
    (
        canopy_height_model_volume_cm3,
        canopy_grid_top_height_cv,
        upper_canopy_volume_ratio,
    ) = canopy_height_model_traits(
        trimmed_crown_uv,
        trimmed_crown_heights,
        cell_mm=5.0,
    )
    (
        crown_3d_hull_volume_cm3,
        occupied_voxel_volume_index_cm3,
        layered_hull_volume_cm3,
    ) = pointcloud_volume_traits(
        trimmed_crown_uv,
        trimmed_crown_heights,
        voxel_mm=5.0,
        slice_mm=10.0,
    )

    final_mask = np.zeros(target_mask.shape, dtype=bool)
    final_mask.ravel()[pixel_indices] = True
    mask_pixels = int(np.count_nonzero(target_mask))
    valid_in_mask = int(np.count_nonzero(valid & target_mask))
    depth_gradient = pipeline.compute_depth_gradient(depth, valid)
    depth_edge_ratio = float(
        np.count_nonzero((depth_gradient >= 12.0) & valid & target_mask)
        / max(valid_in_mask, 1)
    )
    ground_inlier_ratio = float(
        plane["inlier_points"] / max(plane["candidate_points"], 1)
    )
    border_touch = touches_border(target_mask)

    qc_flags: list[str] = []
    if valid_in_mask / max(mask_pixels, 1) < 0.50:
        qc_flags.append("low_valid_depth_ratio")
    if plane["median_abs_residual_mm"] > 20.0:
        qc_flags.append("high_ground_residual")
    if ground_inlier_ratio < 0.40:
        qc_flags.append("low_ground_inlier_ratio")
    if h99_mm - h95_mm > 75.0:
        qc_flags.append("large_h99_h95_gap")
    if depth_edge_ratio > 0.03:
        qc_flags.append("high_depth_edge_ratio")
    if points_xyz.shape[0] < 1500:
        qc_flags.append("low_point_count")
    if border_touch:
        qc_flags.append("target_touches_image_border")
    if ground_method != "local_ring_ransac":
        qc_flags.append("global_ground_fallback")

    row = {
        "sample_id": instance_id,
        "status": "success",
        "error": "",
        "plant_height_cm": plant_height_cm,
        "crown_length_cm": crown_length_cm,
        "crown_width_cm": crown_width_cm,
        "raw_crown_length_cm": raw_crown_length_cm,
        "raw_crown_width_cm": raw_crown_width_cm,
        "crown_point_count": int(crown_points.shape[0]),
        "height_h95_cm": h95_mm / 10.0,
        "height_h10_cm": h10_mm / 10.0,
        "height_h25_cm": h25_mm / 10.0,
        "height_h50_cm": h50_mm / 10.0,
        "height_h75_cm": h75_mm / 10.0,
        "height_h90_cm": h90_mm / 10.0,
        "height_h99_cm": h99_mm / 10.0,
        "height_h99_5_cm": h99_5_mm / 10.0,
        "height_h99_9_cm": h99_9_mm / 10.0,
        "height_max_cm": hmax_mm / 10.0,
        "height_mean_cm": height_mean_mm / 10.0,
        "height_std_cm": height_std_mm / 10.0,
        "height_iqr_cm": height_iqr_mm / 10.0,
        "height_cv": height_cv,
        "vertical_layer_0_25_ratio": float(layer_ratios[0]),
        "vertical_layer_25_50_ratio": float(layer_ratios[1]),
        "vertical_layer_50_75_ratio": float(layer_ratios[2]),
        "vertical_layer_75_100_ratio": float(layer_ratios[3]),
        "mean_crown_diameter_cm": mean_crown_diameter_cm,
        "crown_aspect_ratio": crown_aspect_ratio,
        "crown_ellipse_area_cm2": crown_ellipse_area_cm2,
        "crown_hull_area_cm2": crown_hull_area_cm2,
        "crown_grid_hull_area_cm2": crown_grid_hull_area_cm2,
        "crown_grid_area_cm2": crown_grid_area_cm2,
        "crown_fill_ratio": crown_fill_ratio,
        "crown_point_density_cm2": crown_point_density_cm2,
        "ellipsoid_volume_proxy_cm3": ellipsoid_volume_proxy_cm3,
        "crown_area_height_proxy_cm3": crown_area_height_proxy_cm3,
        "canopy_height_model_volume_cm3": canopy_height_model_volume_cm3,
        "crown_3d_hull_volume_cm3": crown_3d_hull_volume_cm3,
        "occupied_voxel_volume_index_cm3": occupied_voxel_volume_index_cm3,
        "layered_hull_volume_cm3": layered_hull_volume_cm3,
        "depth_corrected_visible_area_cm2": depth_corrected_visible_area_cm2,
        "height_weighted_visible_volume_cm3": height_weighted_visible_volume_cm3,
        "visible_area_canopy_density": visible_area_canopy_density,
        "normalized_exg_mean": normalized_exg_mean,
        "rgbvi_mean": rgbvi_mean,
        "green_point_ratio": green_point_ratio,
        "green_visible_projected_area_cm2": green_visible_projected_area_cm2,
        "canopy_grid_top_height_cv": canopy_grid_top_height_cv,
        "upper_canopy_volume_ratio": upper_canopy_volume_ratio,
        "pre_sor_height_h99_cm": pre_sor_h99_mm / 10.0,
        "pre_sor_height_h99_5_cm": pre_sor_h99_5_mm / 10.0,
        "pre_sor_height_h99_9_cm": pre_sor_h99_9_mm / 10.0,
        "pre_sor_height_max_cm": pre_sor_hmax_mm / 10.0,
        "target_confidence": confidence if confidence is not None else "",
        "target_mask_pixels": mask_pixels,
        "valid_depth_points_in_mask": valid_in_mask,
        "valid_depth_ratio_in_mask": valid_in_mask / max(mask_pixels, 1),
        "height_filtered_points": int(points_before_sor.shape[0]),
        "sor_retained_points": int(points_xyz.shape[0]),
        "sor_retention_ratio": points_xyz.shape[0]
        / max(points_before_sor.shape[0], 1),
        "depth_edge_ratio_in_mask": depth_edge_ratio,
        "ground_method": ground_method,
        "local_ground_candidate_points": local_candidate_count,
        "ground_candidate_points": plane["candidate_points"],
        "ground_inlier_points": plane["inlier_points"],
        "ground_inlier_ratio": ground_inlier_ratio,
        "ground_median_abs_residual_mm": plane["median_abs_residual_mm"],
        "mask_scale_px": int(plane["mask_scale_px"]),
        "ground_inner_radius_px": int(plane["ground_inner_radius_px"]),
        "ground_outer_radius_px": int(plane["ground_outer_radius_px"]),
        "ground_z_shift_mm": ground_z_shift_mm,
        "ground_signed_residual_p25_mm": plane.get(
            "local_signed_residual_p25_mm", 0.0
        ),
        "ground_signed_residual_p50_mm": plane.get(
            "local_signed_residual_p50_mm", 0.0
        ),
        "ground_signed_residual_p75_mm": plane.get(
            "local_signed_residual_p75_mm", 0.0
        ),
        "ground_signed_residual_p90_mm": plane.get(
            "local_signed_residual_p90_mm", 0.0
        ),
        "ground_normal_scale": float(
            np.sqrt(plane["a"] ** 2 + plane["b"] ** 2 + 1.0)
        ),
        "target_touches_image_border": border_touch,
        "qc_review_required": bool(qc_flags),
        "qc_flags": ";".join(qc_flags),
        "height_method": "adaptive_mask_scale_ground_ring_P90_H99_9",
        "crown_method": CROWN_METHOD + (
            "_calibrated_on_siteA_200"
            if CROWN_CALIBRATION_MODE == "required"
            else "_raw"
        ),
        "camera_preset": CAMERA_PRESET,
        "rgb_depth_mapping": "direct_aligned_pixel",
    }
    return row, final_mask, polygon


def failure_row(sample_id: str, error: Exception) -> dict:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "sample_id": sample_id,
            "status": "error",
            "error": str(error),
            "camera_preset": CAMERA_PRESET,
            "rgb_depth_mapping": "direct_aligned_pixel",
            "height_method": "adaptive_mask_scale_ground_ring_P90_H99_9",
            "crown_method": CROWN_METHOD + (
                "_calibrated_on_siteA_200"
                if CROWN_CALIBRATION_MODE == "required"
                else "_raw"
            ),
            "qc_review_required": True,
            "qc_flags": "extraction_error",
        }
    )
    return row


def extended_csv_row(sample_id: str, row: dict) -> dict:
    """将内部计算结果整理为扩展性状CSV的一行。"""
    output = {field: "" for field in EXTENDED_CSV_FIELDS}
    output["编号"] = sample_id
    if row.get("status") != "success":
        output["质控需复核"] = True
        output["质控标记"] = row.get("qc_flags", "extraction_error")
        return output

    mapping = {
        "株高(cm)": "plant_height_cm",
        "冠幅长(cm)": "crown_length_cm",
        "冠幅宽(cm)": "crown_width_cm",
        "高度P10(cm)": "height_h10_cm",
        "高度P25(cm)": "height_h25_cm",
        "高度P50(cm)": "height_h50_cm",
        "高度P75(cm)": "height_h75_cm",
        "高度P90(cm)": "height_h90_cm",
        "高度P95(cm)": "height_h95_cm",
        "高度P99(cm)": "height_h99_cm",
        "高度均值(cm)": "height_mean_cm",
        "高度标准差(cm)": "height_std_cm",
        "高度四分位距(cm)": "height_iqr_cm",
        "高度变异系数": "height_cv",
        "冠幅平均直径(cm)": "mean_crown_diameter_cm",
        "冠幅长宽比": "crown_aspect_ratio",
        "冠层椭圆面积(cm2)": "crown_ellipse_area_cm2",
        "点云投影凸包面积(cm2)": "crown_hull_area_cm2",
        "点云投影栅格凸包面积(cm2)": "crown_grid_hull_area_cm2",
        "点云投影占据面积(cm2)": "crown_grid_area_cm2",
        "冠层投影填充率": "crown_fill_ratio",
        "冠层点密度(点/cm2)": "crown_point_density_cm2",
        "椭球体积代理(cm3)": "ellipsoid_volume_proxy_cm3",
        "冠层面积株高代理(cm3)": "crown_area_height_proxy_cm3",
        "冠层高度模型体积(cm3)": "canopy_height_model_volume_cm3",
        "三维凸包体积(cm3)": "crown_3d_hull_volume_cm3",
        "体素占据体积指数(cm3)": "occupied_voxel_volume_index_cm3",
        "分层凸包体积(cm3)": "layered_hull_volume_cm3",
        "深度校正可见投影面积(cm2)": "depth_corrected_visible_area_cm2",
        "高度加权可见体量(cm3)": "height_weighted_visible_volume_cm3",
        "可见面积冠层密度": "visible_area_canopy_density",
        "归一化ExG均值": "normalized_exg_mean",
        "RGBVI均值": "rgbvi_mean",
        "绿色点比例(ExG大于0)": "green_point_ratio",
        "绿色可见投影面积(cm2)": "green_visible_projected_area_cm2",
        "冠层栅格顶高变异系数": "canopy_grid_top_height_cv",
        "上半冠层体积占比": "upper_canopy_volume_ratio",
        "垂直层0-25%点比例": "vertical_layer_0_25_ratio",
        "垂直层25-50%点比例": "vertical_layer_25_50_ratio",
        "垂直层50-75%点比例": "vertical_layer_50_75_ratio",
        "垂直层75-100%点比例": "vertical_layer_75_100_ratio",
        "原始质心弦冠幅长(cm)": "raw_crown_length_cm",
        "原始质心弦冠幅宽(cm)": "raw_crown_width_cm",
        "有效深度比例": "valid_depth_ratio_in_mask",
        "高度点数": "sor_retained_points",
        "冠层点数": "crown_point_count",
        "地面拟合中位绝对残差(mm)": "ground_median_abs_residual_mm",
        "掩膜特征长度(px)": "mask_scale_px",
        "地面内环半径(px)": "ground_inner_radius_px",
        "地面外环半径(px)": "ground_outer_radius_px",
        "质控需复核": "qc_review_required",
        "质控标记": "qc_flags",
    }
    for output_name, internal_name in mapping.items():
        value = row.get(internal_name, "")
        if isinstance(value, (float, np.floating)) and np.isfinite(value):
            value = round(float(value), 6)
        output[output_name] = value
    return output


def core_csv_row(sample_id: str, row: dict) -> dict:
    """输出跨形态维度的低冗余核心性状。"""
    extended = extended_csv_row(sample_id, row)
    return {field: extended[field] for field in CORE_CSV_FIELDS}


def validate_configuration() -> list[Path]:
    if CAMERA_PRESET not in readbin.CAMERA_PRESETS:
        raise ValueError(f"未知相机预设：{CAMERA_PRESET}")
    if not RGBD_DIR.is_dir():
        raise FileNotFoundError(f"RGB-D目录不存在：{RGBD_DIR}")
    if not TARGET_LABEL_DIR.is_dir():
        raise FileNotFoundError(f"目标标签目录不存在：{TARGET_LABEL_DIR}")
    if not 0 < GROUND_INNER_RADIUS_RATIO < GROUND_OUTER_RADIUS_RATIO:
        raise ValueError("地面环带比例必须满足0 < inner < outer")
    if GROUND_INNER_RADIUS_MIN_PX > GROUND_INNER_RADIUS_MAX_PX:
        raise ValueError("地面内环半径上下限无效")
    if GROUND_OUTER_RADIUS_MIN_PX > GROUND_OUTER_RADIUS_MAX_PX:
        raise ValueError("地面外环半径上下限无效")
    if MIN_DEPTH_MM >= MAX_DEPTH_MM:
        raise ValueError("最小深度必须小于最大深度")
    if MIN_PLANT_HEIGHT_MM >= MAX_PLANT_HEIGHT_MM:
        raise ValueError("最小植株高度必须小于最大植株高度")

    labels = sorted(
        TARGET_LABEL_DIR.glob("*__instance_*.target.txt"), key=natural_key
    )
    if MAX_SAMPLES is not None:
        if MAX_SAMPLES <= 0:
            raise ValueError("MAX_SAMPLES必须为正整数或None")
        labels = labels[:MAX_SAMPLES]
    if not labels:
        raise FileNotFoundError(f"没有找到*.target.txt：{TARGET_LABEL_DIR}")
    return labels


def main() -> int:
    configure_crown_calibration()
    target_labels = validate_configuration()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_QC_PREVIEWS:
        QC_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("人工确认单株 RGB-D 性状提取")
    print(f"RGB-D目录：{RGBD_DIR}")
    print(f"目标标签：{TARGET_LABEL_DIR}")
    print(f"样本数量：{len(target_labels)}")
    print(f"相机预设：{CAMERA_PRESET}，深度单位：mm")
    print(
        "株高：掩膜尺度自适应地面环带"
        f"({GROUND_INNER_RADIUS_RATIO:.0%}--{GROUND_OUTER_RADIUS_RATIO:.0%})，"
        "局部地面包络P90校正后的H99.9"
    )
    print(
        "冠幅：H20、PCA P0.5--P99.5稳健裁剪，质心处正交凸包弦；"
        f"校准模式={CROWN_CALIBRATION_MODE}"
    )
    print(f"基础性状：{OUTPUT_CSV}")
    print(f"核心性状：{CORE_OUTPUT_CSV}")
    print(f"扩展性状：{EXTENDED_OUTPUT_CSV}")
    print("=" * 76)

    success_count = 0
    review_count = 0
    started = time.perf_counter()
    with (
        OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file,
        CORE_OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as core_file,
        EXTENDED_OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as extended_file,
    ):
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        core_writer = csv.DictWriter(core_file, fieldnames=CORE_CSV_FIELDS)
        extended_writer = csv.DictWriter(
            extended_file, fieldnames=EXTENDED_CSV_FIELDS
        )
        writer.writeheader()
        core_writer.writeheader()
        extended_writer.writeheader()

        for index, target_path in enumerate(target_labels, start=1):
            sample_id = target_path.name.removesuffix(".target.txt")
            source_id = sample_id.rsplit("__instance_", 1)[0]
            try:
                row, final_mask, polygon = extract_one(sample_id, source_id)
                success_count += 1
                review_count += int(bool(row["qc_review_required"]))
                if SAVE_QC_PREVIEWS and final_mask is not None and polygon is not None:
                    image = readbin.imread_unicode(RGBD_DIR / f"{source_id}.jpg")
                    save_qc_preview(
                        image,
                        polygon,
                        final_mask,
                        QC_PREVIEW_DIR / f"{sample_id}_traits.jpg",
                        row,
                    )
                message = (
                    f"H={row['plant_height_cm']:.1f} "
                    f"L={row['crown_length_cm']:.1f} "
                    f"W={row['crown_width_cm']:.1f} cm"
                )
            except Exception as error:
                row = failure_row(sample_id, error)
                review_count += 1
                message = f"ERROR: {error}"

            writer.writerow(
                {
                    "编号": sample_id,
                    "株高(cm)": round(float(row["plant_height_cm"]), 4)
                    if row["status"] == "success"
                    else "",
                    "冠幅长(cm)": round(float(row["crown_length_cm"]), 4)
                    if row["status"] == "success"
                    else "",
                    "冠幅宽(cm)": round(float(row["crown_width_cm"]), 4)
                    if row["status"] == "success"
                    else "",
                }
            )
            core_writer.writerow(core_csv_row(sample_id, row))
            extended_writer.writerow(extended_csv_row(sample_id, row))
            file.flush()
            core_file.flush()
            extended_file.flush()
            print(f"[{index:03d}/{len(target_labels):03d}] {sample_id}: {message}")

    elapsed = time.perf_counter() - started
    print("-" * 76)
    print(f"处理完成：成功{success_count}，失败{len(target_labels) - success_count}")
    print(f"建议复核：{review_count}")
    print(f"总用时：{elapsed:.1f}秒")
    print(f"基础性状：{OUTPUT_CSV}")
    print(f"核心性状：{CORE_OUTPUT_CSV}")
    print(f"扩展性状：{EXTENDED_OUTPUT_CSV}")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
