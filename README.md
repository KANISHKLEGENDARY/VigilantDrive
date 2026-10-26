# VigilantDrive

## What this project does
VigilantDrive uses a camera and a YOLO11 detector to look for closed eyes, open eyes, and yawns.
It combines detections over time to display NORMAL, WARNING, or DROWSY and play an alarm for configured alert conditions.
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

Press **Q** in the video window to quit. The model path, camera index, image size, and thresholds are in `config.yaml`. The project does not require model retraining to run the demo. For video files, use `python realtime.py --source "path\to\clip.mp4"`.

## Simple architecture

1. `realtime.py` starts `src/main.py`.
2. `src/main.py` reads camera frames, finds the largest Haar-detected face, filters model detections around it, draws the display, and writes session measurements.
3. `src/detector.py` loads the trained YOLO checkpoint and interprets eye boxes.
4. `src/drowsiness.py` tracks PERCLOS, sustained yawns, eye coverage, and the optional head-drop signal over time.
5. `src/alarm.py` repeats and escalates the sound while an alarm condition remains active.
6. `src/metrics.py` records frame-level values to a CSV file in `results/`.

The face detector selects the largest detected face as a driver proxy. It is not face identity or driver tracking. When no face is found for the configured period, detections fall back to the full frame and the screen warns that the driver was not found.

## How the alarm works

A sustained eye closure can trigger an alert after `closed_eye_alarm_seconds`. A yawn must last `yawn_min_duration_seconds`; repeated yawns can trigger an alert according to `yawn_events_for_alarm` and `yawn_drowsy_hold_seconds`. A detected face center that moves down by the configured fraction of the frame and stays there for the configured duration now also triggers an alert. The alarm repeats while an alert remains active and stops after the clear delay. This is a face-position change, **not a measurement of head pitch or gaze**. These are prototype settings, not validated medical thresholds.

## Evaluation and data checks

- `python scripts/evaluate.py --weights "runs/fcbam/weights/best.pt" --data "data/Driver_Drowsiness_YOLO.training_ready/data.yaml" --split test` evaluates object detection at the `imgsz` configured in `config.yaml`.
- Put consented test clips and hand labels in `sample_videos/`, then run `python scripts/test_videos.py`. It writes `results/results.md` with event recall, alarm precision, false alarms per hour, and alert delay grouped by filename condition.
- `python scripts/check_split.py` checks train/validation files using exact image hashes, perceptual similarity, and source-like filename prefixes. A filename-prefix check is only a heuristic; it cannot prove that drivers are separated if the filenames do not identify them.
- The current train/validation audit flagged **100 exact duplicate hashes**, **178 shared filename prefixes**, and **12,280 possible perceptual-hash matches**. Some sampled candidates are frames from the same person and scene. Treat validation results as potentially optimistic; the prefix and perceptual checks need manual review, and driver-level split isolation is not established.

When the demo saves a review clip, it writes an overlay version and a clean version, then adds the clean filename to `false_alarm_dir/labels.csv` with the label left blank. Review the video and fill the label manually. The clip is not a training example until its contents and boxes are checked. Do not assume a false-alarm clip is mislabeled just because the system alerted.

## Limitations

- The largest Haar-detected face is only a driver proxy. Passengers, missed faces, camera motion, and non-frontal faces can still cause errors.
- The eye pairing and face-position drop checks are geometric heuristics. The head-drop rule can miss a head tilt if the face center does not move down, and can react to camera movement or normal posture changes.
- Per-driver calibration is not implemented. The detector provides class confidence scores and boxes, not eye landmarks or a validated personal eye measurement; changing the closed-eye threshold from open-eye confidence would be an unvalidated rule.
- The score thresholds and head-drop rule need testing across more drivers, lighting, glasses, and camera angles.
- The system does not use head pose, vehicle speed, or GPS. Speed/GPS and validated head-pose input are future work.
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
