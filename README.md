# Runway Model — Marine Object Detection

A small **computer-vision** workspace for **marine and coastal monitoring**: detect **fish**, **marine waste**, and **oil spills** in uploaded videos. Object detection in this project is built around **rft-etr**; trained checkpoints are shipped as Ultralytics-compatible `.pt` weights and are loaded at inference time for fast, practical deployment.

---

## Table of contents

- [Overview](#overview)
- [What this repository includes](#what-this-repository-includes)
- [Project layout](#project-layout)
- [Object detection models](#object-detection-models)
- [How the Streamlit app works](#how-the-streamlit-app-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the web application](#running-the-web-application)
- [Command-line inference (`detect.py`)](#command-line-inference-detectpy)
- [Training new weights (`main.py`)](#training-new-weights-mainpy)
- [Tips for accuracy and performance](#tips-for-accuracy-and-performance)
- [Troubleshooting](#troubleshooting)
- [Data and licensing](#data-and-licensing)

---

## Overview

**Goal:** Take a user-uploaded video, run one or more detection models frame by frame, visualize boxes and labels (optionally in near–real time), aggregate detection counts, and offer a downloadable annotated MP4.

**Typical workflow:**

1. Install Python dependencies in a virtual environment.
2. Ensure `final_models/` contains the three weight files.
3. Start `streamlit_app.py`, upload a video, choose **Single model** or **All models**, adjust confidence, and run detection.
4. Review on-screen results and download the processed video if needed.

---

## What this repository includes

| Component | Purpose |
|-----------|---------|
| **`final_models/`** | Frozen checkpoints: fish, marine waste, oil spill (`.pt`). |
| **`streamlit_app.py`** | Browser UI: upload → inference → live frame preview → summary table → download. |
| **`main.py`** | Example **training** script using Ultralytics YOLO and a dataset YAML. |
| **`detect.py`** | Minimal **CLI-style** example: load one `best.pt` and run `model.predict()` on a video path. |
| **`data/`** | Dataset folder structure and `data.yaml` (often aligned with Roboflow exports). |
| **`runs/`** | Default output from YOLO training/validation (if you train locally). |

---

## Project layout

```
runway_model/
├── final_models/
│   ├── best_fish.pt      # Fish dataset
│   ├── best_waste.pt     # Marine waste dataset
│   └── best_oil.pt       # Oil spill dataset
├── streamlit_app.py      # Streamlit entrypoint
├── main.py               # Training example (YOLO)
├── detect.py             # Single-model video predict example
├── data/
│   └── data.yaml         # Paths, class names, optional Roboflow metadata
├── runs/                 # Training artifacts (when you train)
└── README.md
```

Sample MP4 files or extra folders (e.g. `oil spill/`) may exist for local experiments; the app only **requires** the files above plus your Python environment.

---

## Object detection models

Detection is described in this project as using **rft-etr** for object detection. For **inference**, the Streamlit application loads the checkpoints below with **Ultralytics `YOLO`**, which expects the standard `.pt` layout produced by typical YOLO training or conversion pipelines.

| Model key in UI | File | Suggested use case |
|-----------------|------|------------------|
| **Fish** | `final_models/best_fish.pt` | Fish / aquatic life in video |
| **Marine Waste** | `final_models/best_waste.pt` | Plastic and debris in water |
| **Oil Spill** | `final_models/best_oil.pt` | Oil spill–related patterns |

**Single model:** Only the selected head runs — faster CPU/GPU use, single-domain labels.  
**All models:** Each frame is passed through every selected model; boxes are drawn in distinct colors and labels are prefixed with the model name (e.g. `Fish: rohu 0.82`).

---

## How the Streamlit app works

1. **Upload:** User provides a video (`mp4`, `avi`, `mov`, `mkv`).
2. **Configuration:** Detection mode (single vs all) and **confidence threshold** (e.g. `0.25`).
3. **Inference loop:** For each frame:
   - Run the active model(s) with `model.predict(..., conf=..., verbose=False)`.
   - Draw bounding boxes and text; accumulate per-label counts.
   - Update a **live preview** (`st.image`) and optional rolling **stats** table.
4. **Output file:** Frames are written with OpenCV to an MP4 on disk (`detected_output.mp4` in the working directory during the run), then read back into memory for **download**.
5. **After run:** A **final summary** table sorts detections by count.

Models are wrapped in **`@st.cache_resource`** so weights are loaded once per Streamlit process, not on every interaction.

---

## Requirements

- **OS:** Windows, Linux, or macOS (examples below use **Windows PowerShell**).
- **Python:** 3.10+ recommended (matches common PyTorch / Ultralytics stacks).
- **Hardware:** **GPU** (CUDA) strongly recommended for long videos and “All models”; CPU will work but is slower.
- **FFmpeg (optional but recommended):** Helps if you need browser-friendly H.264 outputs or post-processing; the current app primarily uses OpenCV’s MP4 writer.

---

## Installation

### 1. Create and activate a virtual environment

```powershell
cd D:\runway_model
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1
```

If execution policy blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install Python packages

```powershell
pip install --upgrade pip
pip install ultralytics streamlit opencv-python pandas torch
```

Ultralytics will pull other dependencies as needed (`numpy`, `pillow`, etc.).

### 3. Verify checkpoints

Confirm these paths exist:

- `final_models\best_fish.pt`
- `final_models\best_waste.pt`
- `final_models\best_oil.pt`

If any are missing, the app will stop with an error listing which files are absent.

---

## Running the web application

From the **repository root** (so relative paths to `final_models` resolve correctly):

```powershell
.\myvenv\Scripts\python.exe -m streamlit run .\streamlit_app.py
```

**Important:** Use Streamlit’s CLI (`streamlit run ...`). A command like `run streamlit_app.py` is **not** valid in PowerShell — `run` is not a built-in.

When the terminal prints a URL (typically **`http://localhost:8501`**), open it in a browser. Then:

1. Choose **Single Model** or **All Models**.
2. Pick a model when in single mode.
3. Adjust **Confidence Threshold** (higher = fewer but more confident boxes).
4. Upload a video and click **Run Detection**.
5. When finished, use **Download Detected Video** if you want the MP4 file locally.

---

## Command-line inference (`detect.py`)

`detect.py` is a short template: it loads **one** `best.pt` from `runs/detect/...` and calls `model.predict()` on a hard-coded video path, with optional display and saved output.

Before running:

- Point `YOLO(...)` to an existing weights file.
- Point `source=` to your video (or use `0` for a webcam index).

Run with the same virtual environment:

```powershell
python detect.py
```

Use this when you prefer **no UI** or want to script batch runs.

---

## Training new weights (`main.py`)

`main.py` demonstrates **training** with **Ultralytics YOLO** (e.g. `yolo11n.pt`) on the dataset defined in `data/data.yaml`:

- Adjust **`data=`** to your YAML path.
- Set **`epochs`**, **`imgsz`**, and **`device`** (`0` for first GPU, `cpu` for CPU).
- Training writes under `runs/detect/` by default; copy or export your best weights into `final_models/` if you want the Streamlit app to use them (update names/paths in `streamlit_app.py` if you rename files).

After training, you can run **`model.val()`** and **`model.export(...)`** as needed (ONNX, etc.).

---

## Tips for accuracy and performance

- **Confidence:** Start around `0.25`; raise it if you see too many false positives, lower if you miss small objects.
- **All models mode:** Runs three detectors per frame — expect roughly **3×** the compute vs one model.
- **Resolution / length:** Long 4K videos are expensive; consider trimming clips for experiments.
- **Streamlit + OpenCV:** Writing `detected_output.mp4` uses the process working directory; keep the app’s cwd predictable (run from repo root).
- **GPU:** Install the correct **PyTorch + CUDA** build from [pytorch.org](https://pytorch.org/) for your driver; then Ultralytics will use the GPU when `device` allows it internally.

---

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| **`Missing models: ...`** | Restore `.pt` files under `final_models/` or fix `MODEL_PATHS` in `streamlit_app.py`. |
| **`streamlit` not found** | Use `python -m streamlit run streamlit_app.py` inside the activated venv, or `pip install streamlit`. |
| **`run` is not recognized** | Use `streamlit run streamlit_app.py`, not `run streamlit_app.py`. |
| **Video won’t play in browser** | Download the MP4 and open in VLC; optional: transcode with FFmpeg to H.264/yuv420p. |
| **Very slow inference** | Use **Single model**, shorter clips, smaller resolution, or GPU. |
| **Cannot open video** | Check codec support; try re-encoding the input to H.264 MP4. |

---

## Data and licensing

Dataset layout and provenance are described in **`data/data.yaml`** (including optional **Roboflow** blocks: workspace, project URL, license such as **CC BY 4.0**). Always verify **terms of use** before redistributing images or weights derived from a dataset.

---

## Summary

This project ties together **rft-etr**-oriented **object detection** with **three domain-specific weights** (fish, marine waste, oil spill), a **Streamlit** front end for interactive video analysis, and optional **YOLO training / CLI** scripts for iteration. Keep `final_models` populated, run Streamlit from the repo root, and tune confidence and mode to match your deployment scenario.
