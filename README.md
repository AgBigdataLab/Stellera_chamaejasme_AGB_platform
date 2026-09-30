# Stellera RGB-D Biomass Estimation

An end-to-end framework for individual-plant segmentation, three-dimensional phenotypic trait extraction, and aboveground biomass (AGB) estimation of *Stellera chamaejasme* from RGB-D data.

Shuchen Liu<sup>1,2</sup>, Yunpeng Cui<sup>1,2*</sup>

## Main files

The main files and directories are as follows:

1. `examples/` - Example RGB-D data included with the current package.

```text
*.jpg  - RGB images.
*.bin  - Registered depth images with the same file stems.
```

2. `output/` - Default directory for biomass prediction results.

```text
predicted_AGB.csv  - Instance-level trait and AGB prediction results.
*_biomass.jpg      - Visualized individual-plant predictions.
```

3. `stellera_pipeline/` - Source code for instance segmentation, RGB-D processing, three-dimensional trait extraction, and biomass prediction.

4. `weights/` - Model weight directory.

```text
stellera_segnet_bcdca.pth       - StelleraSegNet instance-segmentation checkpoint.
tabpfn_biomass_model.joblib     - TabPFN model for seven-trait AGB prediction.
```

5. `predict_stellera_biomass.py` - End-to-end inference script.

6. `requirements.txt` - Python dependency list.

## Default behavior

Without command-line arguments, the script:

- reads paired RGB and depth files from `examples/`;
- loads the segmentation and biomass models from `weights/`;
- detects each individual *S. chamaejasme* plant;
- extracts seven RGB-D-derived phenotypic traits;
- predicts the AGB of every detected plant; and
- saves the CSV results and labelled images to `output/`.

Run the included examples:

```bash
python predict_stellera_biomass.py
```

## Input data

Each still sample requires a registered RGB image and a same-stem depth file:

```text
my_data/001.jpg
my_data/001.bin
```

The `.bin` file must contain a `1280 x 720` `uint32` depth map in millimetres. The script supports a single RGB-D pair, a directory containing multiple RGB-D pairs, or an Intel RealSense `.bag` recording.

Single RGB-D pair:

```bash
python predict_stellera_biomass.py --input my_data/001.jpg
```

Directory of RGB-D pairs:

```bash
python predict_stellera_biomass.py --input my_data --output-dir output
```

Intel RealSense RGB-D recording:

```bash
python predict_stellera_biomass.py --input recording.bag --frame-interval 0.3
```

GPU inference:

```bash
python predict_stellera_biomass.py --input my_data --device cuda:0
```

Standard RGB-only videos are not supported because they do not contain metric depth information.

## Phenotypic traits

The biomass model uses seven RGB-D-derived traits: plant height, crown length, crown width, occupied-voxel volume index, coefficient of variation of canopy height, crown length-to-width ratio, and projected crown fill ratio.

## Environment

Python 3.10 is recommended. Install a CUDA-compatible PyTorch build first for GPU inference, and then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies include:

- Python 3.10 or later
- PyTorch and torchvision
- RF-DETR
- TabPFN
- NumPy, pandas, SciPy, and scikit-learn
- OpenCV-Python
- pyrealsense2
- joblib
