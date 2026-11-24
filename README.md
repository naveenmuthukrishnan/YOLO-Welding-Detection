# YOLO Welding Detection

An AI-based weld quality inspection system built on **YOLO11** ([Ultralytics](https://docs.ultralytics.com/)). The model finds welds in images, video or a live camera feed and classifies each one, so defective welds can be flagged automatically instead of by eye alone.

## What it detects

| ID | Class     | Treated as |
|----|-----------|------------|
| 0  | Bad Weld  | Defective  |
| 1  | Good Weld | Good       |
| 2  | Defect    | Defective  |

For video and live camera input, each frame gets a simple verdict: **Defective** if any Bad Weld or Defect is detected, **Good** if only Good Weld is detected, and **No Weld** if nothing is detected.

## How the project fits together

```
Dataset (YOLO format)  →  Train on Google Colab (T4 GPU)  →  Validate  →  Export (ONNX / OpenVINO / TensorRT)  →  Test on a local laptop
```

1. **Train** a YOLO11 model on a labelled weld dataset (up to 200 epochs, with early stopping).
2. **Validate** it with mAP@50 and mAP@50-95, and run predictions on the test images.
3. **Export** the best weights to ONNX, OpenVINO and TensorRT.
4. **Deploy and benchmark** the exports locally on an NVIDIA MX350 GPU, an Intel GPU and a CPU.

## Repository contents

| Folder | What it contains |
|--------|------------------|
| [`welding_project/`](welding_project/) | The **training project**: scripts to check the dataset, train, validate, predict on test images, run a live webcam demo and export the model, plus `data.yaml` and its requirements. See its [README](welding_project/README.md). |
| [`deployment and testing/`](deployment%20and%20testing/) | The **local deployment test**: the trained weights and exported models (PyTorch, ONNX, OpenVINO, TensorRT), `webcam.py` for video/camera inference, sample weld videos, and the hardware benchmark. See its [README](deployment%20and%20testing/README.md). |
| [`200 epochs/`](200%20epochs/) | Output of the 200-epoch training run. <!-- Confirm and describe the contents: weights, result plots, confusion matrix, metrics. --> |

## Local benchmark at a glance

Inference time per frame on the test laptop (lower is better):

| Format and hardware                    | Inference time | Approx. FPS |
|----------------------------------------|----------------|-------------|
| ONNX on NVIDIA MX350                   | ~230.8 ms      | ~4.3        |
| OpenVINO on Intel GPU                  | ~433.5 ms      | ~2.3        |
| OpenVINO, `device="cpu"` (see note)    | ~464.0 ms      | ~2.2        |
| ONNX on CPU                            | ~1225.0 ms     | ~0.8        |

The OpenVINO `device="cpu"` row may not be a CPU-only measurement. Details are in the [deployment README](deployment%20and%20testing/README.md#benchmark).

> [!IMPORTANT]
> **TensorRT was run on Google Colab (T4 GPU) only.** The local NVIDIA MX350 is not supported by TensorRT, so TensorRT is not part of the local benchmark. See the deployment README for details.

## Getting started

- To **train or re-train** the model, start with [`welding_project/README.md`](welding_project/README.md).
- To **run the model locally** on a video or webcam, start with [`deployment and testing/README.md`](deployment%20and%20testing/README.md).

## Dataset

The weld dataset is not included in this repository because of its size. Add it as described in the [training README](welding_project/README.md#dataset).

## Tech stack

Python, Ultralytics YOLO11, PyTorch, OpenCV, ONNX Runtime, OpenVINO, TensorRT (Colab only).
