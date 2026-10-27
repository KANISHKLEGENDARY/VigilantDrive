"""Run the live decision pipeline on a clip and report alert intervals."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from src.alarm import Alarm
from src.config import (ALERT_POLICIES, AUDIO_TONE_SECONDS,
                        HEAD_DOWN_DIRECTION, HEAD_DOWN_ANGLE_DEGREES, HEAD_TURN_ANGLE_DEGREES,
                        MINIMUM_BRIGHTNESS, MINIMUM_SHARPNESS)
from src.detector import Detector, expand_face_box, infer_eye_state, keep_detections_in_box
from src.drowsiness import DrowsinessMonitor
from src.eye_logic import EyeDebouncer, combine_eye_signals, ear_says_closed
from src.face_signals import FaceCalibration, FaceSignalExtractor, angle_delta
from src.main import load_config


def read_labels(path: Path | None) -> list[tuple[float, float]]:
    """Read hand-labeled drowsy time ranges when a labels file is supplied."""
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows or not {"start_sec", "end_sec", "label"}.issubset(rows[0]):
        raise ValueError("Labels CSV needs start_sec,end_sec,label columns")
    return [(float(row["start_sec"]), float(row["end_sec"])) for row in rows
            if row["label"].strip().lower() in {"drowsy", "alert"}]


def intervals(times: list[float], active: list[bool]) -> list[tuple[float, float]]:
    """Convert per-frame alert states into start and stop timestamps."""
    result, started = [], None
    for now, is_active in zip(times, active):
        if is_active and started is None:
            started = now
        elif not is_active and started is not None:
            result.append((started, now))
            started = None
    if started is not None and times:
        result.append((started, times[-1]))
    return result


def longest_silent_gap(truth: list[tuple[float, float]], alerts: list[tuple[float, float]]) -> float:
    """Measure the longest labeled-drowsy gap without a scheduled tone."""
    longest = 0.0
    for start, end in truth:
        cursor = start
        for alert_start, alert_end in alerts:
            if alert_end <= cursor or alert_start >= end:
                continue
            longest = max(longest, alert_start - cursor)
            cursor = max(cursor, min(end, alert_end))
        longest = max(longest, end - cursor)
    return longest


def tone_windows(times: list[float], states: list[str]) -> list[tuple[float, float]]:
    """Simulate tone windows from the same alert-policy table as the sound worker."""
    tones, index = [], 0
    while index < len(times):
        state = states[index]
        end_index = index + 1
        while end_index < len(times) and states[end_index] == state:
            end_index += 1
        if state != "NORMAL":
            policy = ALERT_POLICIES[state]
            next_tone = times[index]
            end_time = times[end_index] if end_index < len(times) else times[end_index - 1]
            while next_tone <= end_time:
                tones.append((next_tone, min(end_time, next_tone + AUDIO_TONE_SECONDS)))
                next_tone += policy["repeat_seconds"]
        index = end_index
    return tones


def run_video(video: Path, config_path: Path) -> tuple[list[tuple[float, float]], float, list[tuple[float, float]]]:
    """Run detection and the real temporal logic over a recorded video."""
    cfg, _ = load_config(config_path)
    detector = Detector(cfg["model_weights"], str(cfg["device"]), cfg["confidence"], cfg["iou"],
                        cfg["class_confidence"], cfg["imgsz"], cfg["half"],
                        cfg["class_names"])
    face_tools = FaceSignalExtractor()
    calibration = FaceCalibration()
    eyes = EyeDebouncer()
    monitor = DrowsinessMonitor(cfg)
    alarm = Alarm(audio_enabled=False)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        face_tools.close()
        alarm.close()
        raise RuntimeError(f"Could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    interval = max(1, int(cfg["inference_interval_frames"]))
    detections, times, active, states = [], [], [], []
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            now = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if now <= 0:
                now = frame_number / fps
            face = face_tools.measure(frame)
            face_box = face.box if face is not None else None
            if face_box is None:
                detections = []
            elif (frame_number - 1) % interval == 0:
                detections = detector.detect(frame)
            bounded = keep_detections_in_box(detections,
                expand_face_box(face_box, frame.shape) if face_box is not None else None)
            yolo_eye = infer_eye_state(bounded, float(cfg["single_eye_closed_confidence"]),
                                       float(cfg["eye_pair_max_size_ratio"]))
            calibrated = calibration.update(now, face)
            ear_closed = ear_says_closed(face.ear, calibration.ear_baseline) if calibrated and face else None
            candidate = combine_eye_signals(yolo_eye, ear_closed)
            eye_state = eyes.update(candidate, now)
            pitch = yaw = 0.0
            if calibrated and face:
                pitch = angle_delta(face.pitch_degrees, calibration.pitch_baseline) * HEAD_DOWN_DIRECTION
                yaw = angle_delta(face.yaw_degrees, calibration.yaw_baseline)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            camera_clear = (float(np.mean(gray)) >= MINIMUM_BRIGHTNESS
                            and float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= MINIMUM_SHARPNESS)
            has_face = face is not None
            yawn = calibrated and any(name == "Yawn" for name, _, _ in bounded)
            monitor_eye = (False if not calibrated and has_face else
                           None if candidate is None else eye_state)
            monitor.update(now, monitor_eye, yawn, face_visible=has_face,
                           head_pitch=pitch, head_yaw=yaw, camera_clear=camera_clear)
            requested = monitor.alert_level()
            safe = (calibrated and has_face and camera_clear and candidate is False and eye_state is False
                    and pitch < HEAD_DOWN_ANGLE_DEGREES and abs(yaw) < HEAD_TURN_ANGLE_DEGREES)
            actual = alarm.update(requested, now, safe_to_clear=safe)
            times.append(now)
            active.append(actual != "NORMAL")
            states.append(actual)
    finally:
        capture.release()
        face_tools.close()
        alarm.close()
    alerts = intervals(times, active)
    return alerts, (times[-1] if times else 0.0), tone_windows(times, states)


def main() -> None:
    """Parse clip and labels, then print alert timing and measured silence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Recorded video to evaluate")
    parser.add_argument("--labels", type=Path, help="CSV with start_sec,end_sec,label ground truth")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    alerts, duration, tones = run_video(args.video, args.config)
    print("Alert | start (s) | stop (s) | duration (s)")
    for number, (start, end) in enumerate(alerts, 1):
        print(f"{number} | {start:.2f} | {end:.2f} | {end - start:.2f}")
    truth = read_labels(args.labels)
    if truth:
        print(f"Longest silent gap inside labeled drowsy intervals: {longest_silent_gap(truth, tones):.2f} s")
        print("Silence is simulated from the configured tone schedule; it is not an audio-device recording.")
    else:
        print("Longest silent gap during true drowsiness: not measurable without hand-labeled drowsy intervals.")
    print(f"Video duration: {duration:.2f} s; predicted alert intervals: {len(alerts)}")


if __name__ == "__main__":
    main()
