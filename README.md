# VigilantDrive

## What this project does
VigilantDrive uses a camera and a YOLO11 detector to look for closed eyes, open eyes, and yawns.
It combines detections over time to display NORMAL, WARNING, DROWSY, CRITICAL, DRIVER_NOT_FOUND, or CAMERA_UNCLEAR and play an alarm for configured alert conditions.
This is a learning demo, not a certified or medically validated vehicle safety system.

## Run the webcam demo
Open PowerShell in this folder. If `.venv` already exists, run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe realtime.py
```

If the virtual environment does not exist yet:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe realtime.py
```

Press **Q** in the video window to quit, **D** to hide/show diagnostics, and **C** to restart calibration. Startup calibration waits for a visible face and enough valid samples; alerts are silent while calibration is in progress. The checkpoint path and output folders are in `config.yaml`; numeric thresholds are in `src/config.py`. MediaPipe Face Mesh is added through `requirements.txt`. The demo uses the existing YOLO checkpoint without retraining it. For a video file, use `python realtime.py --source "path\to\clip.mp4"`.

## Simple architecture

1. `realtime.py` starts `src/main.py`.
2. `src/main.py` reads frames, runs calibration, uses the largest visible face, draws the display, and writes session measurements.
3. `src/detector.py` loads the unchanged YOLO checkpoint. `src/face_signals.py` adds ready-made MediaPipe eye and head landmarks; it does not train a model.
4. `src/eye_logic.py` combines YOLO and eye aspect ratio (EAR) and rejects brief frame-level changes.
5. `src/drowsiness.py` calculates time-based PERCLOS and maps eye, yawn, visibility, and head-pose signals to one state.
6. `src/alarm.py` maps that same state to sound patterns on a separate thread.
7. `src/metrics.py` writes eye/head values, alert state, FPS, total frame time, and capture/face-mesh/YOLO/JPEG/draw/display stage times to a CSV under `results/`.

The largest visible face is only a driver proxy, not identity tracking. When face landmarks are unavailable, the app ignores whole-frame eye/yawn detections and raises a driver-not-found alert after the configured delay.

## How the alarm works

Thresholds and alarm patterns are listed in [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) and configured in `src/config.py`. A plain-language explanation of fixes B1–B13, state transitions, checks, and remaining limits is in [docs/EXPLAINED.md](docs/EXPLAINED.md). The visible state is the same state used to choose the alarm pattern. It holds the alert until the face is visible, eyes are open, the image is clear, and the head is up for the required time. These are prototype settings, not validated medical thresholds.

## Evaluation and data checks

- `python scripts/evaluate.py --weights "runs/fcbam/weights/best.pt" --data "data/Driver_Drowsiness_YOLO.training_ready/data.yaml" --split test` evaluates object detection at the `imgsz` configured in `config.yaml`.
- To list alert start/stop times for one clip, run `python scripts/test_video.py "path\to\demo.mp4"`. Add `--labels "path\to\labels.csv"` with `start_sec,end_sec,label` columns to measure the longest silent gap inside labeled drowsy intervals.
- Put consented test clips and hand labels in `sample_videos/`, then run `python scripts/test_videos.py`. It writes `results/results.md` with event recall, alarm precision, false alarms per hour, and alert delay grouped by filename condition.
- `python scripts/check_split.py` checks train/validation files using exact image hashes, perceptual similarity, and source-like filename prefixes. A filename-prefix check is only a heuristic; it cannot prove that drivers are separated if the filenames do not identify them.
- The current train/validation audit flagged **100 exact duplicate hashes**, **178 shared filename prefixes**, and **12,280 possible perceptual-hash matches**. Some sampled candidates are frames from the same person and scene. Treat validation results as potentially optimistic; the prefix and perceptual checks need manual review, and driver-level split isolation is not established.

When the demo saves a review clip, it writes an overlay version and a clean version, then adds the clean filename to `false_alarm_dir/labels.csv` with the label left blank. Review the video and fill the label manually. The clip is not a training example until its contents and boxes are checked. Do not assume a false-alarm clip is mislabeled just because the system alerted.

## Limitations

- The largest face is a driver proxy, not driver identification. More than one person, occlusion, and camera movement can still confuse it.
- EAR and head angles are landmark estimates, not validated medical or vehicle-approved measurements. Camera orientation and calibration affect them; check the displayed angle direction on your own camera.
- Startup calibration uses at least 40 valid open-eye, forward-facing samples over about five seconds. Press **C** to recalibrate.
- Dark/blur checks use simple brightness and sharpness rules. Glasses, sunglasses, night lighting, camera placement, and face angles still need testing.
- The app does not use vehicle speed or GPS. Its head-pose logic is a prototype and has not been validated across drivers.
- Object-detection mAP does not show whether drowsiness alerts are caught reliably or how often false alarms occur.
- No event-level video results have been measured for this project yet. Do not use this prototype as a real driving safety device.

## Results

The following numbers are from the evaluation output previously provided for the `baseline_yolo11s` checkpoint, on the test split at **640-pixel** inference size. They describe object detection only and are **not** results for the FCBAM checkpoint or the webcam's configured 416-pixel run.

| Measure | Previously produced result |
|---|---:|
| Test images / labeled instances | 899 / 1,171 |
| Precision / recall | 0.703 / 0.940 |
| mAP@50 / mAP@50–95 | 0.854 / 0.561 |
| Eyeclosed mAP@50 | 0.980 |
| Eyeopen mAP@50 | 0.867 |
| Yawn mAP@50 | 0.717 |
| Inference time on Tesla T4 | 10.5 ms/image |
| YOLO11s parameters / GFLOPs | 9.41 million / 21.3 |
| Live webcam FPS and drowsiness-event metrics | **not measured yet** |

Rerun `scripts/evaluate.py` to produce metrics with the current configured image size. Keep the checkpoint name, data split, and image size with any number you report.
