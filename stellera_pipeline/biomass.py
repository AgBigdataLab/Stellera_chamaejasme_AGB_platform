"""使用已训练的最佳模型预测RGB-D单株地上部干生物量。

直接修改下方路径后运行即可。输入必须是扩展性状提取脚本生成的CSV。
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import cv2
import numpy as np
import pandas as pd

from . import depth_io


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ------------------------------ 路径配置 ------------------------------
INPUT_TRAITS_CSV = (
    PROJECT_ROOT / "output" / "traits" / "target_rgbd_extended_traits.csv"
)
MODEL_PATH = PROJECT_ROOT / "weights" / "tabpfn_biomass_model.joblib"
OUTPUT_DIR = PROJECT_ROOT / "output" / "biomass"
OUTPUT_CSV = OUTPUT_DIR / "predicted_aboveground_dry_biomass.csv"
IMAGE_DIR = PROJECT_ROOT / "examples" / "rgbd"
SEGMENTATION_DIR = PROJECT_ROOT / "output" / "segmentation"
SEGMENTATION_MANIFEST = SEGMENTATION_DIR / "instance_manifest.csv"
VISUALIZATION_DIR = OUTPUT_DIR
MODEL_NAME = "TabPFN"
FEATURES = [
    "株高(cm)",
    "体素占据体积指数(cm3)",
    "冠幅长(cm)",
    "冠幅宽(cm)",
    "冠幅长宽比",
    "高度变异系数",
    "冠层投影填充率",
]
OUTPUT_FEATURES = [
    ("plant height", FEATURES[0]),
    ("crown length", FEATURES[2]),
    ("crown width", FEATURES[3]),
    ("occupied-voxel volume index", FEATURES[1]),
    ("coefficient of variation of canopy height", FEATURES[5]),
    ("crown length-to-width ratio", FEATURES[4]),
    ("projected crown fill ratio", FEATURES[6]),
]
# ---------------------------------------------------------------------


def load_traits() -> tuple[pd.DataFrame, str]:
    """读取最新提取的七性状结果。"""
    if INPUT_TRAITS_CSV.is_file():
        return pd.read_csv(INPUT_TRAITS_CSV, encoding="utf-8-sig"), str(INPUT_TRAITS_CSV)
    raise FileNotFoundError(f"缺少性状文件：{INPUT_TRAITS_CSV}")


def normalize_id(value: object) -> int | None:
    if pd.isna(value):
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def instance_color(instance_id: str) -> tuple[int, int, int]:
    match = re.search(r"__instance_(\d+)$", instance_id)
    index = int(match.group(1)) if match else 1
    hue = int((index * 47) % 180)
    bgr = cv2.cvtColor(
        np.uint8([[[hue, 190, 235]]]), cv2.COLOR_HSV2BGR
    )[0, 0]
    return tuple(int(value) for value in bgr)


def save_prediction_visualizations(output: pd.DataFrame) -> None:
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    for source_id, group in output.groupby("source_image"):
        image_path = IMAGE_DIR / f"{source_id}.jpg"
        image = depth_io.imread_unicode(image_path)
        if image is None:
            raise ValueError(f"无法读取可视化源图像：{image_path}")
        overlay = image.copy()
        annotations: list[tuple[np.ndarray, tuple[int, int, int], str, tuple[int, int]]] = []
        for row in group.itertuples(index=False):
            instance_id = str(row.instance_id)
            mask_path = SEGMENTATION_DIR / f"{instance_id}.target.png"
            mask_image = depth_io.imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                raise FileNotFoundError(mask_path)
            mask = mask_image > 0
            color = instance_color(instance_id)
            overlay[mask] = color
            y, x = np.nonzero(mask)
            if not len(x):
                continue
            prediction = float(row.predicted_biomass_g)
            value_text = f"{prediction:.2f} g" if np.isfinite(prediction) else "trait failed"
            label = f"{instance_id.rsplit('__instance_', 1)[-1]} AGB: {value_text}"
            annotations.append(
                (mask, color, label, (int(np.median(x)), int(np.median(y))))
            )
        canvas = cv2.addWeighted(image, 0.72, overlay, 0.28, 0)
        for mask, color, label, origin in annotations:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, color, 2)
            text_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            text_x = int(np.clip(
                origin[0] - text_size[0] / 2,
                4,
                max(4, canvas.shape[1] - text_size[0] - 4),
            ))
            text_y = int(np.clip(
                origin[1] + text_size[1] / 2,
                text_size[1] + baseline + 4,
                canvas.shape[0] - baseline - 4,
            ))
            text_origin = (text_x, text_y)
            cv2.putText(
                canvas, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 3, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA,
            )
        depth_io.imwrite_unicode(
            VISUALIZATION_DIR / f"{source_id}_biomass.jpg", canvas
        )


def main() -> None:
    for path in (MODEL_PATH,):
        if not path.is_file():
            raise FileNotFoundError(f"缺少文件：{path}")

    traits, traits_source = load_traits()
    required = {"编号", *FEATURES}
    missing = required.difference(traits.columns)
    if missing:
        raise ValueError(f"输入性状缺少模型字段：{sorted(missing)}")

    X = traits[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid = ~X.isna().any(axis=1)
    if not bool(valid.any()):
        raise ValueError("No instances contain all seven valid traits.")

    model = joblib.load(MODEL_PATH)
    predicted = np.full(len(X), np.nan, dtype=float)
    valid_prediction = np.asarray(
        model.predict(X.loc[valid].to_numpy(dtype=float))
    ).reshape(-1)
    predicted[valid.to_numpy()] = np.maximum(valid_prediction, 0.0)

    instance_ids = traits["编号"].astype(str)
    output = pd.DataFrame(
        {
            "source_image": instance_ids.str.rsplit("__instance_", n=1).str[0],
            "instance_id": instance_ids,
            "predicted_biomass_g": predicted,
            "status": np.where(valid.to_numpy(), "success", "trait_extraction_failed"),
        }
    )
    for feature in FEATURES:
        output[feature] = X[feature].to_numpy(dtype=float)
    for output_name, source_feature in OUTPUT_FEATURES:
        output[output_name] = X[source_feature].to_numpy(dtype=float)
    if SEGMENTATION_MANIFEST.is_file():
        manifest = pd.read_csv(SEGMENTATION_MANIFEST, encoding="utf-8-sig")
        output = output.merge(
            manifest[["instance_id", "confidence"]],
            on="instance_id",
            how="left",
            validate="one_to_one",
        )
    if "质控需复核" in traits.columns:
        output["点云质控需复核"] = traits["质控需复核"].astype(bool)
    if "质控标记" in traits.columns:
        output["点云质控标记"] = traits["质控标记"].fillna("")

    final_columns = [
        "source_image",
        "instance_id",
        "predicted_biomass_g",
        "status",
        *(name for name, _ in OUTPUT_FEATURES),
    ]
    if "confidence" in output.columns:
        final_columns.append("confidence")
    output = output[final_columns]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if VISUALIZATION_DIR.is_dir():
        for stale_image in VISUALIZATION_DIR.glob("*_biomass.jpg"):
            stale_image.unlink()
    output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", float_format="%.4f")
    save_prediction_visualizations(output)
    print(f"模型：{MODEL_NAME}")
    print(f"性状数据：{traits_source}")
    print(f"预测样本：{len(output)}")
    print(f"点云质控需复核：{int(output.get('点云质控需复核', pd.Series(False)).sum())}株")
    print(f"结果：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
