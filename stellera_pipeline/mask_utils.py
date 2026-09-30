"""Shared mask IO, matching and YOLO-polygon helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.count_nonzero(mask_a | mask_b)
    return float(np.count_nonzero(mask_a & mask_b) / union) if union else 0.0


def largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Predicted mask contains no valid contour")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    epsilon = max(1.0, 0.001 * cv2.arcLength(contour.reshape(-1, 1, 2), True))
    simplified = cv2.approxPolyDP(contour.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2)
    return simplified if len(simplified) >= 3 else contour


def yolo_line(mask: np.ndarray, score: float) -> str:
    height, width = mask.shape
    contour = largest_contour(mask).astype(np.float64)
    contour[:, 0] /= width
    contour[:, 1] /= height
    coordinates = " ".join(f"{value:.6f}" for value in contour.reshape(-1))
    return f"0 {coordinates} {score:.6f}"


