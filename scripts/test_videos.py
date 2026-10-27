"""Measure drowsiness alerts on short, manually labeled videos."""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import yaml
from src.detector import (Detector, expand_face_box, find_largest_face, infer_eye_state,
                          keep_detections_in_box, keep_driver_region, load_face_detector)
from src.drowsiness import DrowsinessMonitor
from src.main import load_config
from scripts.evaluate_demo import match_events, event_recall, alarm_precision, false_alarms_per_hour, average_detection_delay


# Map a video's name to a condition label such as day, night, or night_glasses.
def condition_from_name(filename):
    ignored = {"normal", "drowsy", "yawn", "alert", "driver"}
    parts = [part.lower() for part in Path(filename).stem.split("_") if part.lower() not in ignored]
    return "_".join(parts) or "unspecified"


# Convert per-frame alert states into start and end times.
def make_intervals(times, active):
    intervals, start = [], None
    for t, is_active in zip(times, active):
        if is_active and start is None:
            start = t
        elif not is_active and start is not None:
            intervals.append((start, t))
            start = None
    if start is not None and times:
        intervals.append((start, times[-1]))
    return intervals


# Run the configured detector and temporal alert logic against the labeled clips.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", default="sample_videos", help="Folder containing labeled test videos")
    parser.add_argument("--labels", default="sample_videos/labels.csv", help="CSV: video,start_sec,end_sec,label")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/results.md")
    args = parser.parse_args()
    video_dir, labels_path = Path(args.videos), Path(args.labels)
    output_path = Path(args.output)
    video_paths = sorted(path for path in video_dir.glob("*") if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not video_paths:
        report = "# Video evaluation\n\nNo test videos found. Event recall, alarm precision, false alarms per hour, and detection delay: **not measured yet**.\n"
        output_path.write_text(report, encoding="utf-8")
        print(report)
        print(f"Saved: {output_path}")
        return
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels CSV is required when videos exist: {labels_path}")
    with labels_path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    required_columns = {"video", "start_sec", "end_sec", "label"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError("labels CSV must have columns: video,start_sec,end_sec,label")
    labels_by_video = defaultdict(list)
    for row in rows:
        start, end = float(row["start_sec"]), float(row["end_sec"])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid interval for {row['video']}: {start}..{end}")
        label = row["label"].strip().lower()
        if label not in {"drowsy", "alert", "normal"}:
            raise ValueError(f"Unsupported label {label!r}; use drowsy or normal in {labels_path}")
        labels_by_video[Path(row["video"]).name].append((start, end, label))
    cfg, _ = load_config(Path(args.config))
    detector = Detector(cfg["model_weights"], str(cfg["device"]), cfg["confidence"], cfg["iou"],
                        cfg.get("class_confidence"), int(cfg["imgsz"]), cfg.get("half", False), cfg["class_names"])
    cascade = load_face_detector()
    by_condition = defaultdict(lambda: {"truth": 0, "predicted": 0, "matched": 0, "false": 0,
                                        "normal_seconds": 0.0, "delays": []})
    report_rows = []
    for path in video_paths:
        video_rows = labels_by_video.get(path.name, [])
        if not video_rows:
            raise ValueError(f"No labels found for video {path.name}")
        truth = [(start, end) for start, end, label in video_rows if label in {"drowsy", "alert"}]
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or float(cfg.get("camera_fps_fallback", 20))
        monitor, frame_count, times, active = DrowsinessMonitor(cfg), 0, [], []
        face_box, missed = None, 0
        refind_n = max(1, int(cfg.get("face_refind_interval_frames", 5)))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1
                if frame_count == 1 or frame_count % refind_n == 0:
                    found = find_largest_face(frame, cascade, float(cfg.get("face_scale_factor", 1.1)),
                                              int(cfg.get("face_min_neighbors", 5)),
                                              int(cfg.get("face_min_size_pixels", 60)))
                    if found is not None:
                        face_box, missed = found, 0
                    else:
                        missed += refind_n
                if missed >= int(cfg.get("face_lost_fallback_frames", 30)):
                    face_box = None
                detections = keep_driver_region(detector.detect(frame), frame.shape, cfg.get("driver_roi"))
                face_region = (expand_face_box(face_box, frame.shape, float(cfg.get("face_box_padding_fraction", .25)))
                               if face_box is not None else None)
                detections = keep_detections_in_box(detections, face_region)
                eyes = infer_eye_state(detections, float(cfg.get("single_eye_closed_confidence", .90)),
                                       float(cfg.get("eye_pair_max_size_ratio", 2.0)))
                yawn = any(item[0] == "Yawn" for item in detections)
                t = (frame_count - 1) / fps
                monitor.update(t, eyes, yawn, face_visible=face_box is not None)
                state = monitor.alert_level()
                active.append(state != "NORMAL")
                times.append(t)
        finally:
            cap.release()
        duration = max(times[-1] + 1 / fps, 0.0) if times else 0.0
        predictions = make_intervals(times, active)
        matches = match_events(predictions, truth, float(cfg.get("minimum_event_overlap_fraction", .5)))
        normal_seconds = sum(end - start for start, end, label in video_rows if label == "normal")
        condition = condition_from_name(path.name)
        result = {"video": path.name, "condition": condition, "frames": frame_count,
                  "true_events": len(truth), "predicted_events": len(predictions), "matched_events": len(matches),
                  "event_recall": event_recall(matches, len(truth)),
                  "alarm_precision": alarm_precision(matches, len(predictions)),
                  "false_alarms_per_hour": (false_alarms_per_hour(len(predictions), matches, normal_seconds)
                                            if normal_seconds else None),
                  "mean_delay_seconds": average_detection_delay(matches)}
        report_rows.append(result)
        agg = by_condition[condition]
        agg["truth"] += len(truth)
        agg["predicted"] += len(predictions)
        agg["matched"] += len(matches)
        agg["false"] += len(predictions) - len(matches)
        agg["normal_seconds"] += normal_seconds
        agg["delays"].extend(max(0.0, pred[0] - actual[0]) for pred, actual, _ in matches)
    lines = ["# Video evaluation results", "",
             "Uses the configured live image size and alert rules. Event matching requires the configured fraction of the labeled event to overlap.", "",
             "## By condition", "",
             "| Condition | Events caught / labeled | Alarm precision | False alarms/hour | Mean delay (s) |",
             "|---|---:|---:|---:|---:|"]
    for condition, data in sorted(by_condition.items()):
        precision = data["matched"] / data["predicted"] if data["predicted"] else None
        recall = data["matched"] / data["truth"] if data["truth"] else None
        false_per_hour = data["false"] / (data["normal_seconds"] / 3600) if data["normal_seconds"] else None
        delay = sum(data["delays"]) / len(data["delays"]) if data["delays"] else None
        values = (f"{data['matched']}/{data['truth']}",
                  f"{precision:.3f}" if precision is not None else "n/a",
                  f"{false_per_hour:.3f}" if false_per_hour is not None else "n/a",
                  f"{delay:.3f}" if delay is not None else "n/a")
        lines.append(f"| {condition} | " + " | ".join(values) + " |")
    lines += ["", "## Per video", "", "| Video | Condition | Events caught / labeled | Alarm precision | False alarms/hour | Mean delay (s) |",
              "|---|---|---:|---:|---:|---:|"]
    for row in report_rows:
        p = row["alarm_precision"]
        r = row["event_recall"]
        f = row["false_alarms_per_hour"]
        d = row["mean_delay_seconds"]
        lines.append(f"| {row['video']} | {row['condition']} | {row['matched_events']}/{row['true_events']} | "
                     f"{p:.3f} | {f:.3f} | {d:.3f} |" if p is not None and f is not None and d is not None
                     else f"| {row['video']} | {row['condition']} | {row['matched_events']}/{row['true_events']} | "
                          f"{p if p is not None else 'n/a'} | {f if f is not None else 'n/a'} | {d if d is not None else 'n/a'} |")
    lines += ["", "These are measurements only for the supplied labeled clips; they do not establish road-safety performance.", ""]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
