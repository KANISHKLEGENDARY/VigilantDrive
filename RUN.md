# Run instructions

## Webcam quick start

Use Python 3.10 or 3.11 in the project virtual environment. Install the pinned packages once:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the environment already exists, only activate it and install the updated audio dependency if needed:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install pygame==2.6.1
```

From this project folder, start the webcam demo:

```powershell
python realtime.py
```

For a presentation-size window:

```powershell
python realtime.py --fullscreen
```

Press **Q** to close. Press **F** to save the previous ten seconds as a review clip under `results/false_alarms/`. The clip is created only after you press F; review and remove it yourself if it contains private footage. Session values, including eye visibility and alert reason, are logged to a CSV in `results/`.

The demo uses the weights path in `config.yaml`. Relative paths are resolved from the project folder. `device: auto` chooses CUDA only when the installed PyTorch has CUDA support; otherwise it uses CPU. `imgsz: 416` is a speed setting. Compare its detection behavior with 640 on your camera before claiming it is more accurate. Set `half: true` only on a compatible CUDA setup.

The eye logic returns closed, open, or unknown. It pairs eye boxes by location; a single closed-eye box must exceed `single_eye_closed_confidence`. Missing eye detections are shown as unknown and are not counted as open in PERCLOS. Low eye visibility also prevents stale PERCLOS from driving the score. A yawn must remain detected for `yawn_min_duration_seconds` before it counts. The alarm repeats while an alert condition is active and escalates its tone over time. These are software heuristics, not a certified vehicle safety system.

If you need to limit detections to one part of the image, change `driver_roi` in `config.yaml` to `[left, top, right, bottom]`, using values from `0.0` to `1.0`. This is a manual camera region; it does not identify a driver's face automatically. Keep the full-frame default for a single-person laptop demo.

To use a video file instead of the camera:

```powershell
python realtime.py --source "data/sample_clips/example.mp4"
```

Video files use their media timestamps for duration-based logic. Cameras use a monotonic clock.

### If Windows blocks Pillow/Ultralytics

If Python reports `An Application Control policy has blocked this file` while importing `PIL._imaging`, Windows is blocking a native dependency before the demo starts. Do not disable or bypass that policy. Use `notebooks/Colab_Webcam_Demo.ipynb` instead: put the current project folder or ZIP and the existing checkpoint in Google Drive, open the notebook in Colab, select a GPU if available, and run its cells. The browser accesses the webcam while Colab runs inference. This requires internet and may have more delay than local inference.

## Train and evaluate

```powershell
python scripts/train.py --data data/Driver_Drowsiness_YOLO.training_ready/data.yaml --model yolo11s.pt --epochs 100 --imgsz 640 --batch -1 --seed 42 --name baseline_yolo11s --project runs --device 0
python scripts/train.py --data data/Driver_Drowsiness_YOLO.training_ready/data.yaml --model configs/yolov11s_fcbam.yaml --pretrained yolo11s.pt --epochs 100 --imgsz 640 --batch -1 --seed 42 --name fcbam_yolo11s --project runs --device 0
python scripts/evaluate.py --weights runs/baseline_yolo11s/weights/best.pt --data data/Driver_Drowsiness_YOLO.training_ready/data.yaml --split test --device 0
python scripts/evaluate.py --weights runs/fcbam_yolo11s/weights/best.pt --data data/Driver_Drowsiness_YOLO.training_ready/data.yaml --split test --device 0
python scripts/verify_fcbam.py
```

Use the same training data, image size, epochs, seed, and `yolo11s.pt` initialization for a fair baseline comparison. The FCBAM run transfers only weights whose names and shapes match; inspect the Ultralytics transfer log to confirm how many were accepted and report that limitation.

If Colab disconnects before a run completes, reconnect the same Drive and resume from that run's `weights/last.pt`:

```powershell
python scripts/train.py --resume "runs/fcbam_yolo11s/weights/last.pt" --device 0
```

Use the exact path in your Drive-backed run directory. `best.pt` is for evaluation/inference; `last.pt` contains the most recent training state for resuming.

## Demo evaluation

Prepare short clips and hand-labeled `drowsy_intervals` in `data/sample_clips/ground_truth.json`. Add `normal: true` only to verified normal clips. Add a `condition` string (for example `day`, `night`, or `glasses`) to group results. Then run:

```powershell
python scripts/evaluate_demo.py --config config.yaml --clips data/sample_clips --labels data/sample_clips/ground_truth.json
```

For CSV-labeled clips named by recording condition (for example `day_normal.mp4`), use:

```powershell
python scripts/test_videos.py --videos sample_videos --labels sample_videos/labels.csv
python scripts/check_split.py --data data/Driver_Drowsiness_YOLO.training_ready/data.yaml
```

The evaluator uses the `imgsz` value from `config.yaml` and requires at least the configured fraction of a labeled drowsiness event to overlap an alert interval. The split check combines exact hashes, perceptual hashes, and source-like name prefixes; it cannot prove driver-level separation if file names do not encode driver identity.

The report contains event recall, detection delay, false alarms per hour for clips explicitly marked normal, and summaries by condition. Its event matching uses any temporal overlap; inspect the clips manually before reporting results. Do not use this short-clip evaluation as a substitute for a subject-separated road test.

## Unit checks

```powershell
python -m pytest -q
```

Do not describe detection mAP as classification accuracy. Keep the test split untouched until final evaluation.
