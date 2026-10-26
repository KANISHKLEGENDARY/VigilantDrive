"""Evaluate hand-labeled demo intervals and report event/condition outcomes."""
import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import yaml
from src.detector import (Detector, expand_face_box, find_largest_face, infer_eye_state,
                          keep_detections_in_box, keep_driver_region, load_face_detector)
from src.drowsiness import DrowsinessMonitor


def overlap_fraction(predicted, truth):
    """Return the fraction of a labeled event covered by a predicted interval."""
    start = max(predicted[0], truth[0])
    end = min(predicted[1], truth[1])
    duration = max(0.0, truth[1] - truth[0])
    return (max(0.0, end - start) / duration) if duration else float(predicted[0] <= truth[0] <= predicted[1])


def match_events(predicted, truth, minimum_overlap):
    """Match each prediction to at most one labeled event using the overlap threshold."""
    unmatched = set(range(len(truth)))
    matches = []
    for event in predicted:
        options = [(overlap_fraction(event, truth[index]), index) for index in unmatched]
        options = [item for item in options if item[0] >= minimum_overlap]
        if options:
            fraction, index = max(options)
            unmatched.remove(index)
            matches.append((event, truth[index], fraction))
    return matches


def event_recall(matches, truth_count):
    """Calculate the share of labeled drowsiness events that were matched."""
    return len(matches) / truth_count if truth_count else None


def alarm_precision(matches, prediction_count):
    """Calculate the share of predicted alert events that matched labels."""
    return len(matches) / prediction_count if prediction_count else None


def false_alarms_per_hour(prediction_count, matches, duration_seconds):
    """Calculate unmatched predictions per recorded video hour."""
    return (prediction_count - len(matches)) / (duration_seconds / 3600) if duration_seconds > 0 else None


def average_detection_delay(matches):
    """Calculate the mean nonnegative delay from labeled start to predicted start."""
    delays = [max(0.0, predicted[0] - truth[0]) for predicted, truth, _ in matches]
    return sum(delays) / len(delays) if delays else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--labels", default="data/sample_clips/ground_truth.json")
    parser.add_argument("--clips", default="data/sample_clips")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    labels_path = Path(args.labels)
    if not labels_path.is_file():
        raise FileNotFoundError(f"Add clips and hand-labeled intervals first: {labels_path}")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    detector = Detector(cfg["model_weights"], str(cfg["device"]), cfg["confidence"], cfg["iou"],
                        cfg.get("class_confidence"), cfg["imgsz"], cfg.get("half", False),
                        cfg.get("class_names", ["Eyeclosed", "Eyeopen", "Yawn"]))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    outputs = []
    by_condition = defaultdict(lambda: {"clips": 0, "truth_events": 0, "detected_events": 0,
                                        "normal_hours": 0.0, "false_alarms": 0, "durations": [],
                                        "predictions": 0, "matches": 0})
    face_cascade = load_face_detector()
    for clip in labels["clips"]:
        path = Path(args.clips) / clip["file"]
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open labeled clip: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = float(cfg.get("camera_fps_fallback", 20.0))
        frame_count, times, states, latencies = 0, [], [], []
        face_box = None
        face_missed_frames = 0
        refind_n = max(1, int(cfg.get("face_refind_interval_frames", 5)))
        monitor = DrowsinessMonitor(cfg)
        processing_started = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1
                if frame_count == 1 or frame_count % refind_n == 0:
                    found = find_largest_face(frame, face_cascade, float(cfg.get("face_scale_factor", 1.1)),
                                              int(cfg.get("face_min_neighbors", 5)),
                                              int(cfg.get("face_min_size_pixels", 60)))
                    if found is not None:
                        face_box, face_missed_frames = found, 0
                    else:
                        face_missed_frames += refind_n
                if face_missed_frames >= int(cfg.get("face_lost_fallback_frames", 30)):
                    face_box = None
                media_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                t = media_t if media_t > 0 else (frame_count - 1) / fps
                detections = keep_driver_region(detector.detect(frame), frame.shape, cfg.get("driver_roi"))
                if face_box is not None:
                    face_region = expand_face_box(face_box, frame.shape,
                                                  float(cfg.get("face_box_padding_fraction", 0.25)))
                    detections = keep_detections_in_box(detections, face_region)
                latencies.append(detector.last_latency_ms)
                eye = infer_eye_state(detections, float(cfg.get("single_eye_closed_confidence", .90)),
                                      float(cfg.get("eye_pair_max_size_ratio", 2.0)))
                yawn = any(d[0] == "Yawn" for d in detections)
                center_y = ((face_box[1] + face_box[3]) / 2 / frame.shape[0]
                            if face_box is not None else None)
                monitor.update(t, eye, yawn, face_center_y=center_y)
                state = monitor.state()
                alert = state == "DROWSY" and (monitor.eye_alarm_active() or monitor.yawn_alert_active()
                                                or monitor.head_drop_active)
                times.append(t)
                states.append(alert)
        finally:
            cap.release()
        predicted, event_start = [], None
        for t, active in zip(times, states):
            if active and event_start is None:
                event_start = t
            elif not active and event_start is not None:
                predicted.append((event_start, t))
                event_start = None
        if event_start is not None and times:
            predicted.append((event_start, times[-1]))
        truth = clip.get("drowsy_intervals", [])
        matches = match_events(predicted, truth, float(cfg.get("minimum_event_overlap_fraction", 0.5)))
        duration = max(times[-1] - times[0], 1.0 / fps) if times else 0.0
        normal = bool(clip.get("normal", False))
        condition = str(clip.get("condition", "unspecified"))
        elapsed = max(time.perf_counter() - processing_started, 1e-9)
        metrics = {"clip": clip["file"], "condition": condition, "normal_clip": normal,
                   "duration_seconds": duration, "frames": frame_count,
                   "processing_fps": frame_count / elapsed if frame_count else 0.0,
                   "mean_detector_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                   "true_drowsy_events": len(truth), "predicted_alert_events": len(predicted),
                   "detected_drowsy_events": len(matches),
                   "event_recall": event_recall(matches, len(truth)),
                   "alarm_precision": alarm_precision(matches, len(predicted)),
                   "mean_detection_delay_seconds": average_detection_delay(matches),
                   "false_drowsy_events": len(predicted) - len(matches),
                   "false_alarms_per_hour": false_alarms_per_hour(len(predicted), matches, duration) if normal else None,
                   "event_match_rule": f"one-to-one match; overlap must cover at least {cfg.get('minimum_event_overlap_fraction', 0.5):.0%} of labeled event"}
        outputs.append(metrics)
        agg = by_condition[condition]
        agg["clips"] += 1
        agg["truth_events"] += len(truth)
        agg["detected_events"] += len(matches)
        if normal:
            agg["normal_hours"] += duration / 3600
            agg["false_alarms"] += len(predicted) - len(matches)
        agg["predictions"] = agg.get("predictions", 0) + len(predicted)
        agg["matches"] = agg.get("matches", 0) + len(matches)
        agg["video_seconds"] = agg.get("video_seconds", 0.0) + duration
        agg["durations"].extend(max(0.0, pred[0] - true[0]) for pred, true, _ in matches)
    condition_results = {}
    for condition, agg in by_condition.items():
        condition_results[condition] = {"clips": agg["clips"],
            "truth_events": agg["truth_events"], "detected_events": agg["detected_events"],
            "event_recall": agg["detected_events"] / agg["truth_events"] if agg["truth_events"] else None,
            "false_alarms_per_hour_on_normal_clips": agg["false_alarms"] / agg["normal_hours"] if agg["normal_hours"] else None,
            "alarm_precision": agg["matches"] / agg["predictions"] if agg["predictions"] else None,
            "false_alarms_per_hour": agg["false_alarms"] / agg["normal_hours"] if agg["normal_hours"] else None,
            "mean_detection_delay_seconds": sum(agg["durations"]) / len(agg["durations"]) if agg["durations"] else None}
    payload = {"measurement_note": "Measurements apply only to supplied hand-labeled clips on this machine; not a road-safety certification.",
               "clips": outputs, "by_condition": condition_results}
    output = Path(args.output_dir) / f"demo_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output.read_text(encoding="utf-8"))
    print("Condition | events caught / truth | alarm precision | false alarms/hour on normal video | mean delay (s)")
    for condition, values in condition_results.items():
        print(f"{condition} | {values['detected_events']}/{values['truth_events']} | "
              f"{values['alarm_precision'] if values['alarm_precision'] is not None else 'n/a'} | "
              f"{values['false_alarms_per_hour'] if values['false_alarms_per_hour'] is not None else 'n/a'} | "
              f"{values['mean_detection_delay_seconds'] if values['mean_detection_delay_seconds'] is not None else 'n/a'}")


if __name__ == "__main__":
    main()
