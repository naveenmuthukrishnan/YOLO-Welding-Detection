# YOLO Welding Detection

Real-time weld quality inspection with **YOLO11** ([Ultralytics](https://docs.ultralytics.com/)). The model locates welds in an image, video or live camera feed and classifies each one as a **Bad Weld**, **Good Weld** or **Defect**. The live-camera script turns those detections into a simple per-frame verdict: **Good**, **Defective** or **No Weld**.

## Features

- Object detection of weld seams with three classes (see below)
- Fine-tuning of a pretrained `yolo11s.pt` model on a custom weld dataset, with augmentation tuned for weld imagery
- Dataset sanity check: image and label counts per split
- Validation with mAP@50 and mAP@50-95
- Batch prediction on the test images, with annotated results saved to disk
- Live inference from a webcam or a video file, with colour-coded boxes, a verdict overlay and an FPS counter
- Export to **TensorRT**, **OpenVINO** and **ONNX** for deployment on NVIDIA GPUs, Intel CPUs and other runtimes

## Classes

| ID | Class      | Treated as |
|----|------------|------------|
| 0  | Bad Weld   | Defective  |
| 1  | Good Weld  | Good       |
| 2  | Defect     | Defective  |

Classes 0 and 2 are both counted as defective by `dataset_check.py` and `webcam.py`.

## Project structure

```
welding_project/
├── data.yaml           # Dataset paths and class names
├── dataset_check.py    # Image / label counts for train, test and valid splits
├── train.py            # Fine-tune YOLO11s on the weld dataset
├── validation.py       # Evaluate a trained model (mAP50, mAP50-95)
├── prediction.py       # Run inference on the test images and save results
├── webcam.py           # Live inference on a webcam or video file
├── model_save.py       # Export the trained model (TensorRT / OpenVINO / ONNX)
├── requirements.txt    # Python dependencies
├── datasets/           # NOT included, add your dataset here (see below)
└── runs/               # Created automatically by Ultralytics (weights, plots, predictions)
```

## Dataset

The dataset is **not included** in this repository because of its size. Add it to a `datasets/` folder in the project root, in standard YOLO format (the layout exported by tools such as Roboflow):

```
datasets/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each image has a matching `.txt` label file with one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates are normalised to the 0-1 range. The class IDs must match `data.yaml`:

```yaml
path: datasets
train: train/images
test: test/images
val: valid/images

nc: 3
names: ['Bad Weld', 'Good Weld', 'Defect']
```

<!-- Add the dataset download link here once it is uploaded, e.g.: [Download the dataset](https://...) -->

## Installation

1. Clone the repository and move into the project folder:

   ```bash
   git clone <your-repo-url>
   cd welding_project
   ```

2. (Recommended) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   This installs `ultralytics`, `opencv-python`, `pandas`, `numpy`, `matplotlib` and `seaborn`. YOLO11 needs a recent Ultralytics release (8.3 or newer).

> **GPU training:** `train.py` uses `device = 0` (the first CUDA GPU), so you need a CUDA build of PyTorch. If `torch.cuda.is_available()` returns `False`, install the right build from [pytorch.org](https://pytorch.org/get-started/locally/). To train without a GPU, change `device = 0` to `device = "cpu"` (much slower).

## Usage

Run all commands from the project root, since `data.yaml` and the scripts use relative paths.

### 1. Check the dataset

```bash
python dataset_check.py
```

Prints the number of images in the train, test and validation folders, and the number of good and defective labels in each split.

### 2. Train

```bash
python train.py
```

Fine-tunes the pretrained `yolo11s.pt` model (downloaded automatically on first run). Results are written to `runs/detect/yolo_weld_training/`, and the best weights are saved to `runs/detect/yolo_weld_training/weights/best.pt`.

If a folder with that name already exists, Ultralytics creates `yolo_weld_training2`, `yolo_weld_training3` and so on, so check the actual folder name before using the weights.

To resume an interrupted run:

```python
from ultralytics import YOLO

YOLO("runs/detect/yolo_weld_training/weights/last.pt").train(resume=True)
```

### 3. Set the weights path

`validation.py`, `prediction.py` and `webcam.py` load the model with an empty path, `YOLO("")`. Before running them, set it to your trained weights:

```python
model = YOLO("runs/detect/yolo_weld_training/weights/best.pt")
```

### 4. Validate

```bash
python validation.py
```

Evaluates the model on the validation split and prints the full results object, **mAP@50** (`results.box.map50`) and **mAP@50-95** (`results.box.map`).

### 5. Predict on test images

```bash
python prediction.py
```

Runs inference on `datasets/test/images` with a confidence threshold of 0.50. Annotated images are saved by Ultralytics under `runs/detect/predict*/`, and each detection's class and confidence is printed to the console.

### 6. Live detection (webcam or video)

```bash
python webcam.py
```

- Uses the default camera (`cv2.VideoCapture(0)`). To process a video file instead, pass its path: `cv2.VideoCapture("weld.mp4")`.
- Frames are resized to 640×480 and run at a 0.50 confidence threshold.
- Boxes are **green** for Good Weld and **red** for Bad Weld or Defect.
- The frame verdict is **Defective** if any defective class is detected, otherwise **Good** if a good weld is detected, otherwise **No Weld**.
- The current FPS is shown in the corner.
- Press **q** to quit.

### 7. Export for deployment

```bash
python model_save.py
```

| Format   | Target                                  | Options                    |
|----------|-----------------------------------------|----------------------------|
| TensorRT | NVIDIA GPUs (CUDA)                      | `half=True`, `imgsz=640`   |
| OpenVINO | Intel CPUs and integrated GPUs          | `half=True`                |
| ONNX     | Portable, runs on many runtimes         | `simplify=True`            |

Exported models are written next to `best.pt`. The TensorRT export must be run on the machine with the NVIDIA GPU it will be deployed on. If training was extended or renamed, update the path to `best.pt` in `model_save.py`.

## Training configuration

Defined in `train.py`:

| Parameter                       | Value                                                    |
|---------------------------------|----------------------------------------------------------|
| Base model                      | `yolo11s.pt` (small; `n`, `s`, `m`, `l` are available)  |
| Epochs / early-stopping patience| 200 / 50                                                 |
| Batch size / image size         | 6 / 640                                                  |
| Device                          | GPU 0                                                    |
| Rotation (`degrees`)            | 20                                                       |
| Horizontal flip (`fliplr`)      | 0.20 (vertical flip is disabled)                         |
| Scale (`scale`)                 | 0.50                                                     |
| Perspective (`perspective`)     | 0.0005                                                   |
| HSV hue / saturation / value    | 0.015 / 0.5 / 0.5                                        |
| Mosaic / `close_mosaic`         | 1.0 / last 10 epochs without mosaic                      |
| Mixup                           | 0.1                                                      |
| Output                          | `runs/detect/yolo_weld_training`                         |

Adjust these values to suit your dataset. Set `imgsz` to match your camera feed, and keep `imgsz` consistent between training, validation, prediction and export.

## Results

<!--
Fill in after training. Example table:

| Metric        | Value |
|---------------|-------|
| mAP@50        |       |
| mAP@50-95     |       |
| Precision     |       |
| Recall        |       |

You can also add images from runs/detect/yolo_weld_training/ (confusion_matrix.png, results.png, val_batch0_pred.jpg).
-->

## Notes

- If you run the scripts on Google Colab, the dataset lives under `/content/`, so prefix the paths (for example `/content/datasets/train/images`).
- Treat the verdict from `webcam.py` as decision support. Validate the model on your own welding setup, lighting and camera before relying on it for production quality control.

## Acknowledgements

Built with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) and [OpenCV](https://opencv.org/).
