# Poker Analysis Interface

This repository is a fork of [yolo11-poker-hand-detection-and-analysis](https://github.com/Gholamrezadar/yolo11-poker-hand-detection-and-analysis) and contains an extended interface for analyzing poker hands at the end of a round using a pretrained YOLO model.

## Introduction

`poker_analysis.py` leverages a YOLO-based playing-card detection model to process both video and image inputs of a poker table after a round has ended. The script identifies the five community cards (the “board”) and each player’s two hole cards, evaluates the ranking of each hand using the Treys library, and determines the round’s winner.

## Features

* **Interactive ROI Definition**: Users draw bounding boxes to define Regions of Interest (ROIs) for the community cards and each player before analysis. There is no hard limit on the number of players; at least one player ROI is required.
* **YOLO Inference**: Runs YOLO inference on each video frame (at a reduced frame rate for performance) or on a single high-resolution pass for images.
* **Label Deduplication**: Within each ROI, only the highest-confidence detection per card label is kept, preventing duplicate boxes for the same card.
* **Temporal Smoothing (Video Mode)**: Detections persist for up to `PERSISTENCE_TIMEOUT` seconds to smooth out transient misdetections. A sliding window of `HISTORY_WINDOW` seconds aggregates labels and selects the most frequent cards for each ROI.
* **Draw Overlays**:

  * Green bounding boxes for every YOLO detection, labeled with the card and confidence score.
  * Static blue ROI for the board, showing detected community cards.
  * Static red ROIs for each player, displaying evaluated hand ranks below each ROI.
  * A “Winner” text overlay in the top-left corner when all hands are complete.
* **Interactive Buttons (Video Mode)**:

  * **Pause**: Halts video inference and reveals “Export Results” and “Resume” buttons.
  * **Export Results**: Generates a timestamped `.txt` file listing the board cards and each player’s hand and rank.
  * **Resume**: Continues inference from the paused state.
  * **Replay**: Available when the video ends; restarts from the first frame and clears history.
  * **Exit**: Closes the application.
* **Image Mode**: Performs a single YOLO inference pass at `IMGSZ_IMAGE` resolution and shows “Export Results” and “Exit” buttons for static images.
* **CUDA Support**: Automatically checks for GPU availability via PyTorch and moves the YOLO model to the GPU for faster inference if available.

## Requirements

The following Python packages and versions are required (listed in `poker_analysis_requirements.txt`):

```
torch==2.7.0+cu128
ultralytics==8.3.144
opencv-python==4.11.0.86
numpy==2.2.6
treys==0.1.8
```

## Installation

1. **Clone this fork**:

   ```bash
   git clone https://github.com/MistaPaulo/yolo11-poker-analysis-interface.git
   cd yolo11-poker-analysis-interface
   ```
2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: .\venv\Scripts\Activate
   ```
3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r poker_analysis_requirements.txt
   ```

## Usage

Run the analysis script with one of the following options:

* **Webcam or video file**:

  ```bash
  python poker_analysis.py --source 0                  # Use webcam as input
  python poker_analysis.py --source path/to/video.mp4  # Use video file as input
  
  ```

  1. Draw a bounding box for the Board (community cards), press `Enter` to confirm.
  2. Draw one bounding box for each player, pressing `Enter` to confirm each.
  3. Press `s` once you have defined at least one player ROI to start inference.
  4. Use the on-screen buttons to Pause, Resume, Export Results, Replay, or Exit.

* **Static image**:

  ```bash
  python poker_analysis.py --source path/to/image.jpg
  ```

  1. Draw the Board ROI (press `Enter` to confirm), then draw each Player ROI.
  2. Press `s` once all ROIs are defined; the script runs a single YOLO pass.
  3. Click “Export Results” to save detected cards and hand rankings to a timestamped `.txt` file, or click “Exit” to close.

All exported text files are saved to the `poker_analysis_results/` directory by default.

## Original Repository Acknowledgment

This project is based on the work from [yolo11-poker-hand-detection-and-analysis](https://github.com/Gholamrezadar/yolo11-poker-hand-detection-and-analysis). All credit for the original YOLO-based card detection model belongs to the original author.

