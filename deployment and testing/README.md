# YOLO Weld Detection: Local Inference and Benchmark

Running the trained weld-detection model (YOLO, via [Ultralytics](https://docs.ultralytics.com/)) on a local Windows machine. This folder has a live/video inference script, `webcam.py`, and a speed comparison of the ONNX and OpenVINO exports on an NVIDIA MX350 GPU, an Intel GPU and a CPU.

The model detects three classes: **Bad Weld**, **Good Weld** and **Defect**. `webcam.py` turns the detections in each frame into a verdict: **Good**, **Defective** or **No Weld**.

This README covers local inference only. The model was trained separately on Google Colab.

> [!IMPORTANT]
> **TensorRT was run on Google Colab (T4 GPU) only. It is not supported on the local machine.**
>
> - `best.engine` was built on a Colab **T4** (Turing, compute capability 7.5).
> - The local GPU, an **NVIDIA GeForce MX350** (Pascal, compute capability 6.1), is **not supported by TensorRT**. Current TensorRT 10 releases require compute capability 7.5 or higher ([support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix-10/10.15.1.html)).
> - A TensorRT engine also only runs on the type of GPU it was built on ([engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/inference-library/engine-compatibility.html)), so `best.engine` cannot be loaded on the MX350 even if TensorRT were installed.
> - Because of this, TensorRT is **not part of the local benchmark** and `tensorrt` is **not in `requirements.txt`**. To use TensorRT, export the engine on the GPU you will deploy to (compute capability 7.5 or higher). See [Re-exporting the models](#re-exporting-the-models).

## Test machine

| Component        | Details                                                         |
|------------------|-----------------------------------------------------------------|
| GPU (local)      | NVIDIA GeForce MX350 (Pascal, compute capability 6.1)           |
| Integrated GPU   | Intel graphics, used through OpenVINO as `intel:gpu`            |
| CPU              | Used for the `device="cpu"` runs                                |
| OS               | Windows                                                         |
| GPU (training)   | Google Colab T4 (Turing, compute capability 7.5)                |

<!-- Add the CPU model, RAM, Windows version and driver versions here if you want the benchmark to be reproducible. -->

## Folder contents

| File / folder          | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `webcam.py`            | Live or video inference with box overlay, verdict and FPS counter           |
| `best_weld.pt`         | Trained PyTorch weights (the default model in `webcam.py`)                  |
| `best.onnx`            | ONNX export, used for the MX350 and CPU rows of the benchmark               |
| `best.fp16.onnx`       | FP16 ONNX export (not benchmarked)                                          |
| `best_openvino_model/` | OpenVINO export, used for the Intel GPU and CPU rows of the benchmark       |
| `best.engine`          | TensorRT engine built on the Colab T4 (~390 MB). Cannot run on the MX350    |
| `weld_1.mp4`, `weld_2.mp4`, `weld_3.mp4` | Sample videos for testing (`webcam.py` reads `weld_2.mp4`) |
| `benchmark.png`        | Benchmark results, reproduced in the table below                            |
| `requirements.txt`     | Python dependencies for this local setup                                    |

## Benchmark

Inference time per frame for each export and device on the test machine. Lower is better.

| Rank | Format and hardware                                          | Inference time | Approx. FPS |
|------|--------------------------------------------------------------|----------------|-------------|
| 1    | `best.onnx` on NVIDIA MX350 (`device=0`)                     | ~230.8 ms      | ~4.3        |
| 2    | `best_openvino_model/` on Intel GPU (`device="intel:gpu"`)   | ~433.5 ms      | ~2.3        |
| 3    | `best_openvino_model/` on CPU (`device="cpu"`) †             | ~464.0 ms      | ~2.2        |
| 4    | `best.onnx` on CPU (`device="cpu"`)                          | ~1225.0 ms     | ~0.8        |
| n/a  | `best.engine` (TensorRT) on MX350                            | not supported  | n/a         |

FPS is calculated as 1000 ÷ inference time in ms. The original chart is in [`benchmark.png`](benchmark.png).

† **Row 3 may not be a CPU-only measurement.** In Ultralytics 8.3.238, only `device="intel:gpu"` and `device="intel:cpu"` select a specific OpenVINO device. `device="cpu"` leaves OpenVINO on its automatic (`AUTO`) device selection, which can use the Intel GPU. Row 3 is close to row 2 (464.0 ms vs 433.5 ms), which is consistent with that but does not prove it. Re-run with `device="intel:cpu"` to get a pure CPU figure.

<!-- Add the benchmark settings here: input size, number of runs, warm-up runs, FP32/FP16, test image or video. -->

**What the numbers show**

- **ONNX on the MX350 is the fastest option here**, about 1.9× faster than OpenVINO on the Intel GPU and about 5.3× faster than ONNX on the CPU.
- **ONNX on the CPU is by far the slowest** (about 1.2 s per frame). OpenVINO on the Intel GPU is about 2.8× faster, so prefer OpenVINO when no NVIDIA GPU is available.
- **None of the local options reach real-time video rates.** The best result is about 4 FPS, and the FPS shown in the `webcam.py` window will be lower, because it also counts frame decoding, resizing and drawing.
- The PyTorch `.pt` model and TensorRT were not benchmarked locally. The TensorRT engine was built and run on the Colab T4 only.

## Setup

**Requirements:** Windows, Python 3.10 or newer, an NVIDIA driver for the MX350 (for GPU runs) and Intel graphics drivers (for OpenVINO on the Intel GPU).

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Check that each device is visible:

```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

You should see `True` with the MX350 name, `CUDAExecutionProvider` in the ONNX Runtime list, and `GPU` (the Intel GPU) alongside `CPU` in the OpenVINO list.

### About `requirements.txt`

| Entry | Why |
|-------|-----|
| `--extra-index-url .../whl/cu126` and `torch>=2.9,<2.15` | CUDA 12.6 PyTorch wheels. The default PyPI `torch` on Windows is CPU-only. It is capped below 2.15 because PyTorch stops publishing CUDA 12.6 wheels after 2.14, and Pascal GPUs such as the MX350 have no prebuilt wheels beyond that ([PyTorch notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)) |
| `ultralytics==8.3.238` | YOLO inference and export. Pinned so the behaviour described in this README stays the same |
| `onnx`, `onnxruntime-gpu` | ONNX export and inference, including on the MX350 |
| `openvino` | OpenVINO inference on the Intel GPU and CPU |
| `opencv-python`, `numpy` | Video capture, drawing and arrays |
| *(no `tensorrt`)* | The MX350 does not support TensorRT (see the note at the top) |

## Run

```bat
python webcam.py
```

Key settings in `webcam.py`:

| Setting | Current value | Meaning |
|---------|---------------|---------|
| Model | `best_weld.pt` | Weights or exported model to load |
| Source | `weld_2.mp4` | Video file, or `0` for the default webcam |
| `conf` | `0.25` | Minimum confidence for a detection |
| `imgsz` | `1024` | Inference size. Each frame is also resized to 1024×1024 |
| `device` | `0` | GPU index, `"cpu"`, or an OpenVINO device such as `"intel:gpu"` or `"intel:cpu"` |

What you see in the window:

- **Green** boxes for Good Weld and **red** boxes for Bad Weld and Defect, each labelled with class and confidence.
- A frame **verdict**: *Defective* if any Bad Weld or Defect is detected, otherwise *Good* if a Good Weld is detected, otherwise *No Weld*.
- The **FPS** of each loop iteration (frame read, resize, inference and drawing).
- Press **q** (or **Q**) to quit.

The script runs one black frame through the model before the loop to initialise it.

### Switching model and device

| Run on                | Model path             | `device`       |
|-----------------------|------------------------|----------------|
| NVIDIA MX350          | `best.onnx`            | `0`            |
| Intel GPU             | `best_openvino_model`  | `"intel:gpu"`  |
| CPU (OpenVINO)        | `best_openvino_model`  | `"intel:cpu"`  |
| CPU (ONNX, slowest)   | `best.onnx`            | `"cpu"`        |

For OpenVINO models, `device="cpu"` does not select the CPU. Use `"intel:cpu"` (see † in the benchmark).

> [!NOTE]
> **Set `device` in both the warm-up call and the main inference call.** In Ultralytics 8.3.238 the model is loaded on a device the first time it is called, and a different `device=` in later calls does not move it. The warm-up call (`model(np.zeros(...))`) is the first call, so if it has no `device`, the model is loaded on the default device and the `device` in the loop has no effect. The default is the CUDA GPU if one is available, and for OpenVINO models it is OpenVINO's `AUTO` device selection.

Exported models have a fixed input size, so `imgsz` in `webcam.py` must match the `imgsz` used when the model was exported.

## Re-exporting the models

Exports are created from the PyTorch weights. Ultralytics names each export after the weights file, so `best_weld.pt` produces `best_weld.onnx` and `best_weld_openvino_model/`. Rename them or update the path in `webcam.py`.

```python
from ultralytics import YOLO

model = YOLO("best_weld.pt")

model.export(format="onnx", imgsz=1024, simplify=True)   # runs on the MX350 and CPU
model.export(format="openvino", imgsz=1024, half=True)   # runs on Intel GPU and CPU

# TensorRT: run this ONLY on the GPU you will deploy to (compute capability 7.5+),
# for example a Colab T4. It will not work on the MX350.
model.export(format="engine", imgsz=1024, half=True)
```

Use the same `imgsz` as in `webcam.py`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| ONNX is slow (around 1.2 s per frame) and `CUDAExecutionProvider` is missing | ONNX Runtime is running on the CPU. Make sure only `onnxruntime-gpu` is installed: `pip uninstall onnxruntime onnxruntime-gpu`, then `pip install onnxruntime-gpu`. If it is still missing, check the [CUDA execution provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) |
| `torch.cuda.is_available()` is `False` | A CPU-only PyTorch was installed. Reinstall with the `--extra-index-url` line from `requirements.txt` and keep `torch<2.15` |
| `OpenVINO device 'GPU' not available. Using 'AUTO' instead.` | OpenVINO cannot see the Intel GPU. Update the Intel graphics driver and check `ov.Core().available_devices` |
| Shape or size error when loading an export | `imgsz` in `webcam.py` does not match the size the model was exported with |

## Notes

- Each frame is stretched to a 1024×1024 square, so the aspect ratio of non-square videos is not preserved.
- The verdict is decided per frame, with no smoothing across frames.
- The verdict is decision support. Validate the model on your own welds, lighting and camera before using it for quality control.

## Acknowledgements

Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [ONNX Runtime](https://onnxruntime.ai/), [OpenVINO](https://docs.openvino.ai/) and [OpenCV](https://opencv.org/).
