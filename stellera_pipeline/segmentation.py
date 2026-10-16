"""Predict measured plants with the final BC-DCA RF-DETR-Seg-Large model."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from .mask_utils import (
    IMAGE_SUFFIXES, read_image, write_image, yolo_line,
)


os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_KEY = "rfdetr_seg_large_bcdca"
MODEL_WEIGHT = PROJECT_ROOT / "weights" / "stellera_segnet_bcdca.pth"
IMAGE_DIR = PROJECT_ROOT / "examples" / "rgbd"
OUTPUT_DIR = PROJECT_ROOT / "output" / "segmentation"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_QC_PREVIEWS = True


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def load_threshold() -> float:
    calibration_path = Path(MODEL_WEIGHT).with_suffix(".threshold.json")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    threshold = float(calibration["confidence_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Invalid score threshold: {threshold}")
    return threshold


def validate_inputs() -> list[Path]:
    for path in (MODEL_WEIGHT, Path(MODEL_WEIGHT).with_suffix(".experiment.json"),
                 Path(MODEL_WEIGHT).with_suffix(".threshold.json")):
        if not path.is_file():
            raise FileNotFoundError(path)
    images = sorted(
        (path for path in IMAGE_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
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
    if torch.device(DEVICE).type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
        try:
            with torch.inference_mode():
                probe = torch.ones((1, 300), device=DEVICE)
                torch.topk(probe, 100, dim=1)
                torch.argsort(probe, dim=1, descending=True, stable=True)
                torch.cuda.synchronize(DEVICE)
        except RuntimeError as error:
            raise RuntimeError(
                f"CUDA selection kernels failed in PyTorch {torch.__version__} "
                f"(CUDA {torch.version.cuda}). Check the GPU installation instructions "
                f"in README.md, or explicitly select --device cpu. Original error: {error}"
            ) from error
    return NativeV2Predictor(MODEL_WEIGHT, DEVICE)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_settings(weight):
    weight = Path(weight)
    config_path = weight.with_suffix(".experiment.json")
    threshold_path = weight.with_suffix(".threshold.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    calibration = json.loads(threshold_path.read_text(encoding="utf-8"))
    if config["architecture"] != "BC-DCA-RF-DETR-Seg-Large-v1":
        raise ValueError("The packaged loader requires the saved V2 BC-DCA v1 architecture")
    if int(config["resolution"]) != 672:
        raise ValueError("The V2 checkpoint requires 672 x 672 model inputs")
    if (calibration["schema"] != "validation-mask-f1-v1"
            or calibration["model_key"] != MODEL_KEY
            or calibration["selection_split"] != "val"):
        raise ValueError("Expected the frozen V2 validation-set threshold record")
    if not 0.0 < float(calibration["confidence_threshold"]) < 1.0:
        raise ValueError("Invalid frozen validation confidence threshold")
    if (file_sha256(weight) != calibration["checkpoint_sha256"]
            or file_sha256(config_path) != calibration["config_sha256"]
            or file_sha256(SCRIPT_DIR / "models" / "bcdca.py") != config["module_source_sha256"]):
        raise ValueError("Segmentation checkpoint, configuration, threshold and BC-DCA source do not match")
    if calibration["native_postprocessing"] != {"num_select": 100, "additional_mask_nms": False, "mask_logit_cutoff": 0.0}:
        raise ValueError("Unsupported V2 segmentation postprocessing protocol")
    return config, calibration


class NativeV2Predictor:
    def __init__(self, weight, device):
        if importlib.metadata.version("rfdetr") != "1.10.1":
            raise RuntimeError("The packaged V2 checkpoint requires rfdetr==1.10.1")
        from rfdetr.config import RFDETRSegLargeConfig, SegmentationTrainConfig
        from rfdetr.models.lwdetr import build_model_from_config
        from rfdetr.models.postprocess import PostProcess
        from torchvision.transforms import functional as transforms
        from .models.bcdca import install_bcdca_modules

        config, calibration = model_settings(weight)
        self.calibration = calibration
        self.resolution, self.device = int(config["resolution"]), device
        self.transforms = transforms
        mc = RFDETRSegLargeConfig(resolution=self.resolution, positional_encoding_size=self.resolution // 12,
                                 num_classes=1, pretrain_weights=str(weight), device="cpu", amp=False,
                                 fused_optimizer=False, gradient_checkpointing=False)
        tc = SegmentationTrainConfig(dataset_dir=str(PROJECT_ROOT), output_dir=str(OUTPUT_DIR))

        def offline(*args, **kwargs):
            raise RuntimeError("Network access is disabled during local checkpoint loading")

        with patch("requests.sessions.Session.request", offline), patch("torch.hub.download_url_to_file", offline):
            model = build_model_from_config(mc, tc)
        model = install_bcdca_modules(model, **{key: config[key] for key in (
            "boundary_points", "boundary_radii", "boundary_residual_scale", "boundary_logit_bias")})
        payload = torch.load(weight, map_location="cpu", weights_only=False)
        if payload.get("rfdetr_version", "1.10.1") != "1.10.1":
            raise ValueError("RF-DETR checkpoint version mismatch")
        model.load_state_dict(payload["model"], strict=True)
        self.model = model.float().to(device).eval()
        self.postprocess = PostProcess(num_select=calibration["native_postprocessing"]["num_select"],
                                       upsample_masks_to_image_size=True)

    @torch.inference_mode()
    def predict(self, image_rgb, *, threshold, include_source_image=False):
        height, width = image_rgb.shape[:2]
        tensor = self.transforms.to_tensor(image_rgb)
        tensor = self.transforms.resize(tensor, [self.resolution, self.resolution], antialias=False)
        tensor = self.transforms.normalize(tensor, [.485, .456, .406], [.229, .224, .225])[None].to(self.device)
        result = self.postprocess(self.model(tensor), torch.tensor([[height, width]], device=self.device),
                                  score_threshold=self.calibration["min_candidate_score"])[0]
        masks = result["masks"].cpu().numpy()
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        scores, labels = result["scores"].float().cpu().numpy(), result["labels"].cpu().numpy()
        keep = (labels == 0) & (scores >= threshold)
        return SimpleNamespace(mask=masks[keep], confidence=scores[keep], class_id=labels[keep])


def clear_outputs() -> None:
    for pattern in ("*.txt", "*.target.png", "*.all.png", "*.csv", "*.json"):
        for path in OUTPUT_DIR.glob(pattern):
            path.unlink()
    preview = OUTPUT_DIR / "visualizations"
    if preview.is_dir():
        for path in preview.glob("*.jpg"):
            path.unlink()


def predict_arrays(model, image):
    """Predict one BGR frame without creating intermediate image or mask files."""
    threshold = load_threshold()
    detections = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                               threshold=threshold, include_source_image=False)
    scores = np.asarray(detections.confidence, dtype=np.float32)
    labels = np.asarray(detections.class_id, dtype=np.int64)
    if not len(scores):
        return scores, np.zeros((0, *image.shape[:2]), dtype=bool)
    masks = np.asarray(detections.mask)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.shape != (len(scores), *image.shape[:2]) or labels.shape != scores.shape:
        raise ValueError("Predicted masks must match the RGB frame dimensions and confidence rows")
    keep = (labels == 0) & (scores >= threshold) & np.any(masks, axis=(1, 2))
    if masks.dtype != np.bool_:
        raise ValueError("Expected native binary masks, not mask logits")
    return scores[keep], masks[keep]


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
        scores, masks = predict_arrays(model, image)
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
            f"{len(masks)} instances (native V2 postprocessing)"
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
        "score_threshold_source": ("validation-set maximum-F1 threshold" if np.isclose(
            threshold, model.calibration["confidence_threshold"], rtol=0, atol=1e-12) else "user override"),
        "validation_score_threshold": model.calibration["confidence_threshold"],
        "resolution": model.resolution,
        "additional_mask_nms": False,
        "images": len(images), "instances": len(rows),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Completed {len(images)} images, {len(rows)} instances, "
        f"elapsed {time.perf_counter() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
