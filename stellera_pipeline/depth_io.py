import argparse
from pathlib import Path

import cv2
import numpy as np


def find_default_data_dir():
    """Find the sample data directory without depending on non-ASCII path literals."""
    project_dir = Path(__file__).resolve().parent
    for data_dir in project_dir.rglob("data"):
        if (data_dir / "001.1.bin").exists() and (data_dir / "001.1.jpg").exists():
            return data_dir
    return project_dir / "data"


DEFAULT_DATA_DIR = find_default_data_dir()
DEFAULT_BIN_FILE = DEFAULT_DATA_DIR / "277.1.bin"
DEFAULT_COLOR_FILE = DEFAULT_DATA_DIR / "277.1.jpg"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


CAMERA_PRESETS = {
    "D435i": {"fx": 903.0325, "fy": 903.0325, "cx": 654.9388, "cy": 369.5530},
    "D455": {"fx": 643.6724, "fy": 643.6724, "cx": 649.2437, "cy": 370.9462},
}


def imread_unicode(image_file, flags=cv2.IMREAD_COLOR):
    """Read an image from paths that may contain non-ASCII characters."""
    image_path = Path(image_file)
    if not image_path.exists():
        return None

    data = np.fromfile(image_path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(image_file, image):
    """Write an image to paths that may contain non-ASCII characters."""
    image_path = Path(image_file)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    ext = image_path.suffix or ".png"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        return False

    encoded.tofile(image_path)
    return True


def read_depth_bin(bin_file, width=1280, height=720, dtype=np.uint32):
    """Read a RealSense depth .bin file into a float64 array."""
    bin_path = Path(bin_file)
    if not bin_path.exists():
        raise FileNotFoundError(f"Depth file does not exist: {bin_path}")

    data = np.fromfile(bin_path, dtype=dtype)
    expected_pixels = width * height
    if data.size != expected_pixels:
        raise ValueError(
            f"Unexpected depth size: got {data.size} values, expected "
            f"{expected_pixels} for {width}x{height}. File: {bin_path}"
        )

    depth = data.reshape((height, width)).astype(np.float64)
    depth[depth == 0] = np.nan
    return depth


def print_bin_file_info(bin_file, dtype=np.uint32):
    """Print basic file size and possible image resolution information."""
    bin_path = Path(bin_file)
    item_size = np.dtype(dtype).itemsize
    file_size = bin_path.stat().st_size

    if file_size % item_size != 0:
        print(f"Warning: file size {file_size} is not divisible by {item_size}.")

    num_pixels = file_size // item_size
    print(f"Depth file: {bin_path}")
    print(f"File size: {file_size} bytes")
    print(f"Pixel count: {num_pixels}")

    for width, height in [(1280, 720), (1920, 1080), (848, 480), (640, 480)]:
        if width * height == num_pixels:
            print(f"Possible resolution: {width} x {height}")

    first_values = np.fromfile(bin_path, dtype=dtype, count=10)
    print(f"First 10 depth values: {first_values}")


def normalize_depth_to_uint8(depth, min_depth=None, max_depth=None):
    """Normalize valid depth values to 0-255 for visualization."""
    valid = np.isfinite(depth)
    if not np.any(valid):
        raise ValueError("Depth image has no valid non-zero values.")

    if min_depth is None:
        min_depth = float(np.nanmin(depth))
    if max_depth is None:
        max_depth = float(np.nanmax(depth))
    if max_depth <= min_depth:
        raise ValueError(f"Invalid depth range: min={min_depth}, max={max_depth}")

    normalized = (depth - min_depth) / (max_depth - min_depth)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0)
    return (normalized * 255).astype(np.uint8), min_depth, max_depth


def save_depth_colormap(depth, output_file, colormap=cv2.COLORMAP_JET, min_depth=None, max_depth=None):
    """Save a pseudo-color depth image."""
    depth_uint8, min_depth, max_depth = normalize_depth_to_uint8(depth, min_depth, max_depth)
    colored = cv2.applyColorMap(depth_uint8, colormap)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite_unicode(output_path, colored):
        raise OSError(f"Failed to write image: {output_path}")

    print(f"Saved depth colormap: {output_path}")
    print(f"Visualization depth range: [{min_depth:.3f}, {max_depth:.3f}]")
    return colored


def read_color_aligned(color_file, shape):
    """Read the RGB image and resize it to match the depth image."""
    color_path = Path(color_file)
    color_bgr = imread_unicode(color_path, cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"Cannot read color image: {color_path}")

    height, width = shape
    return cv2.resize(color_bgr, (width, height), interpolation=cv2.INTER_AREA)


def save_color_point_overlay(depth, color_file, output_file, point_size=1):
    """Save the camera-view RGB image masked by valid depth points on a white background."""
    color_bgr = read_color_aligned(color_file, depth.shape)
    valid = np.isfinite(depth) & (depth > 0)

    if point_size > 1:
        kernel = np.ones((point_size, point_size), dtype=np.uint8)
        color_layer = np.zeros_like(color_bgr)
        color_layer[valid] = color_bgr[valid]
        mask = cv2.dilate(valid.astype(np.uint8), kernel) > 0
        color_layer = cv2.dilate(color_layer, kernel)
        canvas = np.full_like(color_bgr, 255)
        canvas[mask] = color_layer[mask]
    else:
        canvas = np.full_like(color_bgr, 255)
        canvas[valid] = color_bgr[valid]

    output_path = Path(output_file)
    if not imwrite_unicode(output_path, canvas):
        raise OSError(f"Failed to write image: {output_path}")

    print(f"Saved camera-view point overlay: {output_path}")
    return canvas


def save_pointcloud_topdown_image(
    point_cloud,
    output_file,
    image_width=1600,
    image_height=1000,
    percentile_clip=1.0,
    point_size=2,
):
    """Render a white-background top-down X/Y projection from colored point cloud rows."""
    if point_cloud.size == 0:
        raise ValueError("Cannot render an empty point cloud.")

    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    rgb = point_cloud[:, 3:6].astype(np.uint8)

    low = float(percentile_clip)
    high = 100.0 - low
    x_min, x_max = np.percentile(x, [low, high])
    y_min, y_max = np.percentile(y, [low, high])

    in_range = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    x = x[in_range]
    y = y[in_range]
    rgb = rgb[in_range]

    if x.size == 0 or x_max <= x_min or y_max <= y_min:
        raise ValueError("Point cloud projection range is invalid after clipping.")

    px = ((x - x_min) / (x_max - x_min) * (image_width - 1)).astype(np.int32)
    py = ((y_max - y) / (y_max - y_min) * (image_height - 1)).astype(np.int32)

    color_layer = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    color_layer[py, px] = rgb[:, ::-1]
    mask[py, px] = 255

    if point_size > 1:
        kernel = np.ones((point_size, point_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
        color_layer = cv2.dilate(color_layer, kernel)

    canvas = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
    canvas[mask > 0] = color_layer[mask > 0]

    output_path = Path(output_file)
    if not imwrite_unicode(output_path, canvas):
        raise OSError(f"Failed to write image: {output_path}")

    print(f"Saved top-down point-cloud render: {output_path}")
    print(f"Rendered point count after clipping: {x.size}")
    return canvas


def save_pointcloud_as_ply(point_cloud, filename):
    """Save x y z r g b point cloud as an ASCII PLY file."""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {point_cloud.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        np.savetxt(f, point_cloud, fmt="%.6f %.6f %.6f %d %d %d")

    print(f"Saved PLY point cloud: {output_path}")


def save_pointcloud_as_binary_ply(xyz, rgb, filename):
    """Save XYZ float32 plus RGB uint8 rows as a binary little-endian PLY."""
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz shape (N, 3), got {xyz.shape}")
    if rgb.shape != xyz.shape:
        raise ValueError(f"Expected rgb shape {xyz.shape}, got {rgb.shape}")

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(xyz.shape[0], dtype=vertex_dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with output_path.open("wb") as file:
        file.write(header)
        vertices.tofile(file)

    return output_path


def depth_to_colored_pointcloud(depth, color_file, camera_params):
    """Convert a depth map and an RGB image into x y z r g b point cloud rows."""
    height, width = depth.shape
    color_bgr = read_color_aligned(color_file, depth.shape)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    y_grid, x_grid = np.mgrid[0:height, 0:width]
    z = depth.ravel()
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        raise ValueError("No valid depth points found.")

    fx = camera_params["fx"]
    fy = camera_params["fy"]
    cx = camera_params["cx"]
    cy = camera_params["cy"]

    x = ((x_grid.ravel() - cx) * z / fx)[valid]
    y = ((y_grid.ravel() - cy) * z / fy)[valid]
    z = z[valid]

    rgb = color_rgb.reshape((-1, 3))[valid]
    point_cloud = np.column_stack((x, y, z, rgb)).astype(np.float64)

    print(f"Valid point count: {point_cloud.shape[0]}")
    print(f"X range: [{point_cloud[:, 0].min():.3f}, {point_cloud[:, 0].max():.3f}]")
    print(f"Y range: [{point_cloud[:, 1].min():.3f}, {point_cloud[:, 1].max():.3f}]")
    print(f"Z range: [{point_cloud[:, 2].min():.3f}, {point_cloud[:, 2].max():.3f}]")
    return point_cloud


def convert_depth_bin_to_outputs(
    bin_file,
    color_file,
    output_dir=DEFAULT_OUTPUT_DIR,
    width=1280,
    height=720,
    camera="D435i",
    prefix="pointcloud_from_bin",
):
    """Read depth/color files and write visualization plus PLY point cloud."""
    camera_params = CAMERA_PRESETS[camera]
    output_dir = Path(output_dir)

    print_bin_file_info(bin_file)
    depth = read_depth_bin(bin_file, width=width, height=height)
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{np.nanmin(depth):.3f}, {np.nanmax(depth):.3f}]")

    colormap_path = output_dir / f"{prefix}_depth_colormap_jet.png"
    save_depth_colormap(depth, colormap_path)

    overlay_path = output_dir / f"{prefix}_rgb_point_overlay.png"
    save_color_point_overlay(depth, color_file, overlay_path, point_size=2)

    point_cloud = depth_to_colored_pointcloud(depth, color_file, camera_params)
    ply_path = output_dir / f"{prefix}.ply"
    save_pointcloud_as_ply(point_cloud, ply_path)

    topdown_path = output_dir / f"{prefix}_topdown_pointcloud.png"
    save_pointcloud_topdown_image(point_cloud, topdown_path)

    return {
        "depth": depth,
        "point_cloud": point_cloud,
        "colormap_path": colormap_path,
        "overlay_path": overlay_path,
        "ply_path": ply_path,
        "topdown_path": topdown_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read a RealSense depth .bin file and generate a depth colormap plus colored PLY point cloud."
    )
    parser.add_argument("--bin", dest="bin_file", type=Path, default=DEFAULT_BIN_FILE, help="Path to depth .bin file.")
    parser.add_argument("--color", dest="color_file", type=Path, default=DEFAULT_COLOR_FILE, help="Path to color image.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated files.")
    parser.add_argument("--width", type=int, default=1280, help="Depth image width.")
    parser.add_argument("--height", type=int, default=720, help="Depth image height.")
    parser.add_argument("--camera", choices=sorted(CAMERA_PRESETS), default="D435i", help="Camera intrinsics preset.")
    parser.add_argument("--prefix", default="pointcloud_from_bin", help="Output filename prefix.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = convert_depth_bin_to_outputs(
        bin_file=args.bin_file,
        color_file=args.color_file,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        camera=args.camera,
        prefix=args.prefix,
    )

    print("Conversion complete.")
    print(f"Generated: {result['colormap_path']}")
    print(f"Generated: {result['overlay_path']}")
    print(f"Generated: {result['ply_path']}")
    print(f"Generated: {result['topdown_path']}")


if __name__ == "__main__":
    main()
