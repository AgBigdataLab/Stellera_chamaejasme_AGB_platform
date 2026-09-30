"""Predict measured plants with the final BC-DCA RF-DETR-Seg-Large model."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from .mask_utils import (
    IMAGE_SUFFIXES, mask_iou, read_image, write_image, yolo_line,
)


os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_KEY = "rfdetr_seg_large_bcdca"
MODEL_WEIGHT = PROJECT_ROOT / "weights" / "stellera_segnet_bcdca.pth"
SCORE_THRESHOLD = 0.4591708
IMAGE_DIR = PROJECT_ROOT / "examples" / "rgbd"
OUTPUT_DIR = PROJECT_ROOT / "output" / "segmentation"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASK_NMS_IOU = 0.60
MASK_NMS_CONTAINMENT = 0.85
SAVE_QC_PREVIEWS = True


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def load_threshold() -> float:
    threshold = float(SCORE_THRESHOLD)
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Invalid score threshold: {threshold}")
    return threshold


def validate_inputs() -> list[Path]:
    for path in (MODEL_WEIGHT,):
        if not path.is_file():
            raise FileNotFoundError(path)
    images = sorted(
        (path for path in IMAGE_DIR.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )
    if not images:
        raise FileNotFoundError(f"No RGB images found in: {IMAGE_DIR}")
    missing_depth = [
        path.name for path in images
        if not (IMAGE_DIR / f"{path.stem}.bin").is_file()
    ]
    if missing_depth:
        raise FileNotFoundError(
            f"Missing same-stem depth files for {len(missing_depth)} images: "
            f"{missing_depth[:5]}"
        )
    return images


def mask_nms(scores: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove duplicates using mask IoU and intersection-over-smaller mask."""
    order = np.argsort(-scores)
    kept: list[int] = []
    for index in order:
        duplicate = False
        for other in kept:
            intersection = np.count_nonzero(masks[index] & masks[other])
            smaller = min(
                np.count_nonzero(masks[index]),
                np.count_nonzero(masks[other]),
            )
            containment = float(intersection / smaller) if smaller else 0.0
            if (
                mask_iou(masks[index], masks[other]) >= MASK_NMS_IOU
                or containment >= MASK_NMS_CONTAINMENT
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(int(index))
    indices = np.asarray(kept, dtype=np.int64)
    return scores[indices], masks[indices]


def save_instance_preview(
    image: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    path: Path,
) -> None:
    canvas = image.copy()
    overlay = image.copy()
    for index, (mask, score) in enumerate(zip(masks, scores), start=1):
        hue = int((index * 47) % 180)
        color = tuple(
            int(value)
            for value in cv2.cvtColor(
                np.uint8([[[hue, 190, 235]]]), cv2.COLOR_HSV2BGR
            )[0, 0]
        )
        overlay[mask] = color
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, color, 2)
        y, x = np.nonzero(mask)
        if len(x):
            origin = (int(np.median(x)), int(np.median(y)))
            cv2.putText(
                canvas,
                f"#{index:03d} {score:.2f}",
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"#{index:03d} {score:.2f}",
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )
    canvas = cv2.addWeighted(canvas, 0.72, overlay, 0.28, 0)
    write_image(path, canvas)


def build_model():
    import rfdetr.inference as inference
    from rfdetr import RFDETRSegLarge
    from .models.bcdca import install_bcdca_modules

    official_builder = inference.build_model

    def custom_builder(args):
        return install_bcdca_modules(official_builder(args))

    with patch.object(inference, "build_model", custom_builder):
        model = RFDETRSegLarge(
            pretrain_weights=str(MODEL_WEIGHT), num_classes=1,
            resolution=504, device=DEVICE, trust_checkpoint=True,
        )
    if getattr(model.model.model, "bcdca_architecture_id", None) is None:
        raise RuntimeError("BC-DCA decoder was not installed")
    return model


def clear_outputs() -> None:
    for pattern in ("*.txt", "*.target.png", "*.all.png", "*.csv", "*.json"):
        for path in OUTPUT_DIR.glob(pattern):
            path.unlink()
    preview = OUTPUT_DIR / "visualizations"
    if preview.is_dir():
        for path in preview.glob("*.jpg"):
            path.unlink()


def main() -> int:
    images = validate_inputs()
    threshold = load_threshold()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_outputs()
    preview_dir = OUTPUT_DIR / "visualizations"
    preview_dir.mkdir(parents=True, exist_ok=True)
    model = build_model()
    rows = []
    started = time.perf_counter()

    for number, image_path in enumerate(images, 1):
        image = read_image(image_path)
        height, width = image.shape[:2]
        detections = model.predict(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            threshold=threshold, include_source_image=False,
        )
        scores = np.asarray(detections.confidence, dtype=np.float32)
        labels = np.asarray(detections.class_id, dtype=np.int64)
        masks = np.asarray(detections.mask, dtype=bool)
        keep = labels == 0
        scores, masks = scores[keep], masks[keep]
        if not len(masks):
            if SAVE_QC_PREVIEWS:
                preview = image.copy()
                cv2.putText(
                    preview, "No detections", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA,
                )
                write_image(preview_dir / f"{image_path.stem}_instances.jpg", preview)
            print(f"[{number:03d}/{len(images):03d}] {image_path.stem}: 0 instances")
            continue

        raw_count = len(masks)
        scores, masks = mask_nms(scores, masks)
        union_mask = np.any(masks, axis=0)
        (OUTPUT_DIR / f"{image_path.stem}.txt").write_text(
            "\n".join(yolo_line(mask, float(score)) for mask, score in zip(masks, scores)) + "\n",
            encoding="utf-8",
        )
        write_image(OUTPUT_DIR / f"{image_path.stem}.all.png", union_mask.astype(np.uint8) * 255)
        for instance_index, (mask, score) in enumerate(zip(masks, scores), start=1):
            instance_id = f"{image_path.stem}__instance_{instance_index:03d}"
            label_path = OUTPUT_DIR / f"{instance_id}.target.txt"
            mask_path = OUTPUT_DIR / f"{instance_id}.target.png"
            label_path.write_text(
                yolo_line(mask, float(score)) + "\n", encoding="utf-8"
            )
            write_image(mask_path, mask.astype(np.uint8) * 255)
            y, x = np.nonzero(mask)
            rows.append(
                {
                    "source_image": image_path.name,
                    "instance_id": instance_id,
                    "confidence": float(score),
                    "mask_pixels": int(mask.sum()),
                    "centroid_x_px": float(x.mean()),
                    "centroid_y_px": float(y.mean()),
                    "label_file": label_path.name,
                    "mask_file": mask_path.name,
                }
            )
        if SAVE_QC_PREVIEWS:
            save_instance_preview(
                image,
                masks,
                scores,
                preview_dir / f"{image_path.stem}_instances.jpg",
            )
        print(
            f"[{number:03d}/{len(images):03d}] {image_path.stem}: "
            f"{len(masks)} instances ({raw_count} before mask NMS)"
        )

    fields = [
        "source_image", "instance_id", "confidence", "mask_pixels",
        "centroid_x_px", "centroid_y_px", "label_file", "mask_file",
    ]
    with (OUTPUT_DIR / "instance_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_DIR / "prediction_config.json").write_text(json.dumps({
        "model": "BC-DCA-RF-DETR-Seg-Large", "checkpoint": str(MODEL_WEIGHT),
        "score_threshold": threshold,
        "score_threshold_source": "embedded validation-set maximum-F1 threshold",
        "mask_nms_iou": MASK_NMS_IOU,
        "mask_nms_intersection_over_smaller": MASK_NMS_CONTAINMENT,
        "images": len(images), "instances": len(rows),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Completed {len(images)} images, {len(rows)} instances, "
        f"elapsed {time.perf_counter() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
