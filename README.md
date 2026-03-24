# Gas Cylinder Auto-Labeling with YOLO Segmentation

This repository contains the code from a graduation-project workflow focused on gas-cylinder auto-labeling, dataset preparation, training, web inference, and localization experiments.

## Core Auto-Labeling Idea

The main pipeline uses a generic YOLO segmentation model and then applies domain-specific rules to recover gas-cylinder masks from common COCO misclassifications.

1. Run YOLO segmentation with a very low confidence threshold (`0.01`).
2. Keep only classes that often resemble a gas cylinder, such as `bottle`, `fire hydrant`, `sink`, `toilet`, `cup`, `chair`, and `vase`.
3. Reject candidates that are too small or too far from the image center.
4. Score the remaining candidates with:

```text
score = confidence * class_weight * centrality_bonus
```

5. Select the best polygon and export it as a YOLO segmentation label or a visualization overlay.

## Why `process_E_to_F.py` Matters

`process_E_to_F.py` is the clearest batch-processing version of the robust pipeline:

- it loads `yolo11x-seg.pt`
- it selects GPU automatically when CUDA is available
- it reads images with `numpy + cv2.imdecode`, which is safer on Windows and Unicode paths
- it reuses the `GasCylinderSegmenter` logic
- it writes processed results from one disk location to another (`E:\gas` to `F:\guoji` in the original local setup)

The same segmentation logic also appears in:

- `segmentation_engine.py`
- `app_v2.py`
- `auto_label_final_robust.py`
- `batch_process_gpu_v3.py`

## Repository Map

- `process_E_to_F.py`: batch inference and export script
- `segmentation_engine.py`: reusable segmentation engine
- `app_v2.py`: FastAPI demo for browser-based inference
- `auto_label_final_robust.py`: robust auto-labeling script that writes YOLO segmentation labels
- `auto_segment_*.py` and `batch_process_gpu*.py`: iterative versions of the auto-labeling pipeline
- `prepare_dataset.py`, `visualize_dataset.py`, `train_yolo.py`: local dataset-preparation and training helpers
- `colab_auto_label.py`, `colab_train.py`, `colab_dataset_clean.py`: Colab workflows exported as Python scripts
- `colab_*.ipynb`: notebook versions of the Colab workflows
- `localization/`: stereo and 3D localization experiments

## Installation

```bash
pip install -r requirements.txt
```

## Notes Before Running

- Many scripts still use the original local Windows paths such as `D:\biyesheji\...`, `E:\gas`, and `F:\guoji`.
- Update those paths before running the scripts on another machine.
- Model weights (`*.pt`), local datasets, generated visualizations, logs, and thesis materials are intentionally not committed to GitHub.
- `data.yaml` is provided as a lightweight example and expects a local `gas_dataset` directory.

