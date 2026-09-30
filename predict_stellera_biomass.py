"""End-to-end StelleraSegNet RGB-D aboveground-biomass inference."""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import cv2
import numpy as np

from stellera_pipeline import biomass, segmentation, traits


ROOT = Path(__file__).resolve().parent
STAGES = ("segmentation", "traits", "biomass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--input-dir",
        dest="input_source",
        type=Path,
        default=ROOT / "examples",
        help=(
            "RGB-D source: a paired RGB image or .bin file, a directory of "
            "same-stem RGB/.bin pairs, or an Intel RealSense .bag recording."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
        help="Directory for the final biomass CSV and labelled RGB images.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device, for example auto, cuda, cuda:0 or cpu.",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=0.3,
        help="Sampling interval in seconds for RealSense .bag input (default: 0.3).",
    )
    return parser.parse_args()


def _paired_image(depth_path: Path) -> Path:
    candidates = [
        path
        for path in depth_path.parent.iterdir()
        if path.is_file()
        and path.stem.lower() == depth_path.stem.lower()
        and path.suffix.lower() in segmentation.IMAGE_SUFFIXES
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one same-stem RGB image for {depth_path.name}; found {len(candidates)}."
        )
    return candidates[0]


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise OSError(f"Unable to encode RGB image: {path}")
    encoded.tofile(path)


def _stage_single_pair(source: Path, destination: Path) -> Path:
    if source.suffix.lower() == ".bin":
        depth_path = source
        image_path = _paired_image(depth_path)
    elif source.suffix.lower() in segmentation.IMAGE_SUFFIXES:
        image_path = source
        depth_path = source.with_suffix(".bin")
        if not depth_path.is_file():
            raise FileNotFoundError(
                f"Missing same-stem depth file for {image_path.name}: {depth_path}"
            )
    else:
        raise ValueError(
            "Single-sample input must be an RGB image or its same-stem .bin depth file."
        )

    destination.mkdir(parents=True, exist_ok=True)
    output_image = destination / f"{image_path.stem}.jpg"
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(image_path, output_image)
    else:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to convert RGB image: {image_path}")
        _write_jpeg(output_image, image)
    shutil.copy2(depth_path, destination / f"{image_path.stem}.bin")
    return destination


def _extract_realsense_bag(
    bag_path: Path,
    destination: Path,
    frame_interval: float,
) -> Path:
    if frame_interval <= 0:
        raise ValueError("--frame-interval must be greater than zero.")
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "RealSense .bag input requires pyrealsense2: pip install pyrealsense2"
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    frame_count = 0
    last_saved_ms: float | None = None
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        profile.get_device().as_playback().set_real_time(False)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)

        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                break
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            timestamp_ms = float(frames.get_timestamp())
            if (
                last_saved_ms is not None
                and timestamp_ms - last_saved_ms < frame_interval * 1000.0
            ):
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            if color.shape[:2] != (720, 1280) or depth_raw.shape != (720, 1280):
                raise ValueError(
                    "RGB-D video must provide aligned 1280 x 720 D455 frames; "
                    f"received RGB {color.shape[:2]} and depth {depth_raw.shape}."
                )
            color_format = color_frame.profile.format()
            if color_format == rs.format.rgb8:
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            elif color_format != rs.format.bgr8:
                raise ValueError(f"Unsupported RealSense color format: {color_format}")

            frame_count += 1
            stem = f"{bag_path.stem}_frame_{frame_count:06d}"
            _write_jpeg(destination / f"{stem}.jpg", color)
            depth_mm = np.rint(
                depth_raw.astype(np.float64) * depth_scale * 1000.0
            ).astype(np.uint32)
            depth_mm.tofile(destination / f"{stem}.bin")
            last_saved_ms = timestamp_ms
    finally:
        if started:
            pipeline.stop()

    if frame_count == 0:
        raise ValueError(f"No synchronized RGB-D frames were extracted from: {bag_path}")
    return destination


def prepare_input(args: argparse.Namespace, temporary_root: Path) -> Path:
    source = args.input_source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    staging_dir = temporary_root / "prepared_input"

    if source.is_dir():
        images = [
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in segmentation.IMAGE_SUFFIXES
        ]
        if images:
            if all(path.suffix.lower() == ".jpg" for path in images):
                return source
            for image_path in sorted(images):
                _stage_single_pair(image_path, staging_dir)
            return staging_dir
        bags = sorted(source.glob("*.bag"))
        if len(bags) == 1:
            return _extract_realsense_bag(
                bags[0], staging_dir, args.frame_interval
            )
        raise FileNotFoundError(
            f"No RGB/.bin pairs found in {source}; a video directory must contain exactly one .bag file."
        )
    if source.suffix.lower() == ".bag":
        return _extract_realsense_bag(source, staging_dir, args.frame_interval)
    return _stage_single_pair(source, staging_dir)


def configure(
    args: argparse.Namespace,
    temporary_root: Path,
    input_dir: Path,
) -> None:
    output_dir = args.output_dir.resolve()

    segmentation.IMAGE_DIR = input_dir
    segmentation.OUTPUT_DIR = temporary_root / "segmentation"
    segmentation.SAVE_QC_PREVIEWS = False
    if args.device != "auto":
        segmentation.DEVICE = args.device

    traits.RGBD_DIR = input_dir
    traits.TARGET_LABEL_DIR = segmentation.OUTPUT_DIR
    traits.OUTPUT_DIR = temporary_root / "traits"
    traits.OUTPUT_CSV = traits.OUTPUT_DIR / "target_rgbd_traits.csv"
    traits.EXTENDED_OUTPUT_CSV = (
        traits.OUTPUT_DIR / "target_rgbd_extended_traits.csv"
    )
    traits.CORE_OUTPUT_CSV = traits.OUTPUT_DIR / "target_rgbd_core_traits.csv"
    traits.QC_PREVIEW_DIR = traits.OUTPUT_DIR / "qc_previews"

    biomass.INPUT_TRAITS_CSV = traits.EXTENDED_OUTPUT_CSV
    biomass.OUTPUT_DIR = output_dir
    biomass.IMAGE_DIR = input_dir
    biomass.SEGMENTATION_DIR = segmentation.OUTPUT_DIR
    biomass.SEGMENTATION_MANIFEST = (
        segmentation.OUTPUT_DIR / "instance_manifest.csv"
    )
    biomass.VISUALIZATION_DIR = output_dir
    biomass.OUTPUT_CSV = (
        biomass.OUTPUT_DIR / "predicted_AGB.csv"
    )


def run_stage(name: str) -> None:
    captured = io.StringIO()
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            if name == "segmentation":
                status = segmentation.main()
            elif name == "traits":
                status = traits.main()
            else:
                biomass.main()
                status = 0
    except Exception:
        diagnostics = captured.getvalue().strip()
        if diagnostics:
            print(diagnostics, file=sys.stderr)
        raise
    if status not in (None, 0):
        raise RuntimeError(f"Stage failed: {name} (exit status {status})")


def main() -> int:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="stellera_biomass_") as temp_dir:
        temporary_root = Path(temp_dir)
        input_dir = prepare_input(args, temporary_root)
        configure(args, temporary_root, input_dir)
        images = segmentation.validate_inputs()
        for index, image_path in enumerate(images, start=1):
            print(f"[{index:03d}/{len(images):03d}] Processing image: {image_path.name}")
        for name in STAGES:
            run_stage(name)
    print(f"Results saved to: {biomass.OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
