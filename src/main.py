"""Live webcam/video drowsiness demonstration."""
from __future__ import annotations
import argparse
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import yaml
from src.alarm import Alarm
from src.detector import (Detector, expand_face_box, infer_eye_state,
                          keep_detections_in_box, keep_driver_region)
from src.drowsiness import DrowsinessMonitor
from src.metrics import Metrics
from src.config import (ALERT_POLICIES, CAMERA_LOST_ALARM_SECONDS, DEFAULT_INFERENCE_INTERVAL,
                        DEFAULT_RUNTIME_CONFIG,
                        HEAD_DOWN_DIRECTION, MINIMUM_BRIGHTNESS, MINIMUM_SHARPNESS,
                        OVERLAY_FONT_SCALE, CALIBRATION_SECONDS)
from src.eye_logic import EyeDebouncer, combine_eye_signals, ear_says_closed
from src.face_signals import FaceCalibration, FaceSignalExtractor, angle_delta


def put_text_fit(frame, text: str, y: int, color, font_scale: float = OVERLAY_FONT_SCALE) -> None:
    """Draw one overlay row with a font small enough to fit the video width."""
    available_width = frame.shape[1] - 36
    scale = font_scale
    while scale > 0.25 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > available_width:
        scale -= 0.025
    cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def load_config(config_path: Path) -> tuple[dict, Path]:
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    loaded_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_config, dict):
        raise ValueError("Config must contain a YAML mapping of setting names to values.")
    cfg = dict(DEFAULT_RUNTIME_CONFIG)
    cfg.update(loaded_config)
    required = ("model_weights", "device", "confidence", "iou", "camera_index", "class_names",
                "session_log_dir", "window_seconds", "perclos_warning", "perclos_drowsy",
                "yawn_window_seconds", "yawn_debounce_seconds", "eye_weight", "yawn_weight",
                "score_warning", "score_drowsy", "hysteresis_margin")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing required config settings: {', '.join(missing)}")
    if not 0 <= float(cfg["confidence"]) <= 1 or not 0 <= float(cfg["iou"]) <= 1:
        raise ValueError("confidence and iou must be between 0 and 1")
    if float(cfg["eye_weight"]) < 0 or float(cfg["yawn_weight"]) < 0 or float(cfg["eye_weight"]) + float(cfg["yawn_weight"]) <= 0:
        raise ValueError("eye_weight and yawn_weight must be non-negative and have a positive sum")
    for key in ("window_seconds", "yawn_window_seconds", "yawn_debounce_seconds", "closed_eye_alarm_seconds"):
        if key in cfg and float(cfg[key]) <= 0:
            raise ValueError(f"{key} must be greater than zero")
    if not 0 <= float(cfg["minimum_eye_coverage"]) <= 1:
        raise ValueError("minimum_eye_coverage must be between 0 and 1")
    if not isinstance(cfg["class_names"], list) or not cfg["class_names"]:
        raise ValueError("class_names must be a non-empty list matching the checkpoint classes")
    if cfg.get("driver_roi") is not None:
        roi = cfg["driver_roi"]
        if (not isinstance(roi, list) or len(roi) != 4 or any(not 0 <= float(v) <= 1 for v in roi)
                or float(roi[2]) <= float(roi[0]) or float(roi[3]) <= float(roi[1])):
            raise ValueError("driver_roi must be [left, top, right, bottom] normalized to 0..1")
    project_root = config_path.parent
    weights = Path(cfg["model_weights"])
    if not weights.is_absolute():
        weights = project_root / weights
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}\nSet model_weights in {config_path}")
    cfg["model_weights"] = str(weights)
    for key in ("session_log_dir", "false_alarm_dir"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(project_root / cfg[key])
    return cfg, project_root


def source_time(cap, source_is_file: bool, frame_index: int, fallback: float, fps: float) -> float:
    """Video files use media timestamps; cameras use monotonic elapsed time."""
    if source_is_file:
        media_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if np.isfinite(media_time) and media_time > 0:
            return media_time
        if fps > 0:
            return frame_index / fps
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VigilantDrive live drowsiness demo")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", help="Camera index or video path (default: configured camera)")
    parser.add_argument("--fullscreen", action="store_true", help="Show a presentation-size demo window")
    parser.add_argument("--no-calibration", action="store_true", help="Skip the five-second face baseline calibration")
    args = parser.parse_args()
    cfg, _ = load_config(Path(args.config))
    detector = Detector(cfg["model_weights"], str(cfg["device"]), cfg["confidence"], cfg["iou"],
                        cfg["class_confidence"], cfg["imgsz"], cfg["half"],
                        cfg["class_names"])
    monitor = DrowsinessMonitor(cfg)
    face_extractor = FaceSignalExtractor()
    calibration = FaceCalibration(float(cfg.get("calibration_seconds", CALIBRATION_SECONDS)))
    eye_debouncer = EyeDebouncer()
    alarm = Alarm()
    source = args.source if args.source is not None else cfg["camera_index"]
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    is_file = isinstance(source, str) and Path(source).is_file()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        alarm.close()
        raise RuntimeError(f"Could not open video source: {source}. Check the camera index, permissions, or file path.")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if source_fps <= 0:
        source_fps = float(cfg["camera_fps_fallback"])
    metrics = Metrics(cfg["session_log_dir"])
    clip_buffer: deque[tuple[float, bytes, bytes]] = deque()
    clip_seconds = float(cfg["false_alarm_buffer_seconds"])
    clip_max_bytes = int(float(cfg["false_alarm_buffer_max_mb"]) * 1024 * 1024)
    clip_bytes = 0
    alert_starts: deque[float] = deque()
    last_alarm_active = False
    lost_since: float | None = None
    last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    capture_enabled = bool(cfg.get("false_alarm_capture_enabled", True))
    frame_index = 0
    inference_interval = max(1, int(cfg.get("inference_interval_frames", DEFAULT_INFERENCE_INTERVAL)))
    last_detections = []
    debug_enabled = True
    face_box = None
    face_not_found = False
    labels_path = Path(cfg.get("false_alarm_dir", cfg["session_log_dir"])) / "labels.csv"
    window_name = "VigilantDrive — Driver Drowsiness Monitor"
    if args.fullscreen:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        while True:
            frame_started = time.perf_counter()
            ok, frame = cap.read()
            wall_t = time.monotonic()
            if not ok:
                if is_file:
                    break  # End of a video file is normal, not a camera-loss event.
                if lost_since is None:
                    lost_since = wall_t
                last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                elapsed_lost = wall_t - lost_since
                status = "CAMERA / VIDEO SIGNAL LOST" if elapsed_lost >= CAMERA_LOST_ALARM_SECONDS else "Reconnecting to camera..."
                cv2.putText(last_frame, status, (25, 220), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 255), 2)
                cv2.putText(last_frame, "Check connection. Press Q to exit.", (25, 270), cv2.FONT_HERSHEY_SIMPLEX, .65, (230, 230, 230), 2)
                incoming = "CAMERA_UNCLEAR" if elapsed_lost >= CAMERA_LOST_ALARM_SECONDS else "NORMAL"
                alarm_state = alarm.update(incoming, wall_t, safe_to_clear=False)
                cv2.putText(last_frame, ALERT_POLICIES[alarm_state]["message"], (25, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
                cv2.imshow(window_name, last_frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue
            clean_frame = frame.copy()
            if lost_since is not None:
                monitor.reset()
                eye_debouncer.reset()
                last_detections = []
                face_box = None
            lost_since = None
            frame_index += 1
            last_frame = frame
            t = source_time(cap, is_file, frame_index, wall_t, source_fps)
            try:
                face_measurement = face_extractor.measure(frame)
            except (ValueError, cv2.error):
                face_measurement = None
            face_box = face_measurement.box if face_measurement is not None else None
            face_visible = face_measurement is not None
            face_not_found = not face_visible
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            camera_clear = brightness >= MINIMUM_BRIGHTNESS and sharpness >= MINIMUM_SHARPNESS
            if face_visible and (frame_index - 1) % inference_interval == 0:
                last_detections = detector.detect(frame)
            detections = list(last_detections) if face_visible else []
            detections = keep_driver_region(detections, frame.shape, cfg["driver_roi"])
            active_face_box = (expand_face_box(
                                face_box, frame.shape, float(cfg["face_box_padding_fraction"]))
                               if face_visible and face_box is not None else None)
            detections = keep_detections_in_box(detections, active_face_box)
            yolo_eye_state = infer_eye_state(detections, float(cfg["single_eye_closed_confidence"]),
                                             float(cfg["eye_pair_max_size_ratio"]))
            calibrated = args.no_calibration or calibration.update(t, face_measurement)
            ear = face_measurement.ear if face_measurement is not None else None
            ear_closed = (ear_says_closed(ear, calibration.ear_baseline)
                          if calibrated and not args.no_calibration else None)
            eye_candidate = combine_eye_signals(yolo_eye_state, ear_closed)
            eye_state = eye_debouncer.update(eye_candidate, t)
            pitch_delta = 0.0
            yaw_delta = 0.0
            if calibrated and face_measurement is not None and not args.no_calibration:
                pitch_delta = angle_delta(face_measurement.pitch_degrees, calibration.pitch_baseline) * HEAD_DOWN_DIRECTION
                yaw_delta = angle_delta(face_measurement.yaw_degrees, calibration.yaw_baseline)
            yawn = calibrated and any(name == "Yawn" for name, _, _ in detections)
            monitor_eye_state = (False if not calibrated and face_visible else
                                 None if eye_candidate is None else eye_state)
            monitor.update(t, monitor_eye_state, yawn,
                           face_visible=face_visible, head_pitch=pitch_delta,
                           head_yaw=yaw_delta, camera_clear=camera_clear)
            incoming_state = monitor.alert_level()
            safe_to_clear = (calibrated and face_visible and camera_clear and eye_candidate is False
                             and eye_state is False
                             and pitch_delta < cfg["head_down_angle_degrees"]
                             and abs(yaw_delta) < cfg["head_turn_angle_degrees"])
            state = alarm.update(incoming_state, t, safe_to_clear=safe_to_clear)
            alarm_active = state != "NORMAL"
            display_reason = (monitor.reason() if state == incoming_state
                             else "Hold alert until eyes are open, head is up, and face is visible")
            if alarm_active and not last_alarm_active:
                alert_starts.append(wall_t)
            last_alarm_active = alarm_active
            while alert_starts and alert_starts[0] < wall_t - 1800:
                alert_starts.popleft()
            frame_ms = (time.perf_counter() - frame_started) * 1000 if 'frame_started' in locals() else 0.0
            fps = metrics.record(wall_t, monitor.score(), monitor.perclos(), monitor.yawn_events(), state,
                                 detector.last_latency_ms, monitor.eye_status(), display_reason, alarm_active,
                                 monitor.eye_coverage(), monitor.head_drop_active, ear,
                                 pitch_delta, yaw_delta, state, frame_ms)
            color = {"NORMAL": (50, 200, 50), "WARNING": (0, 180, 255), "DROWSY": (0, 0, 255),
                     "CRITICAL": (0, 0, 255), "DRIVER_NOT_FOUND": (0, 180, 255),
                     "CAMERA_UNCLEAR": (0, 180, 255)}[state]
            cv2.rectangle(frame, (0, 0), (frame.shape[1], min(160, frame.shape[0])), (18, 18, 18), -1)
            put_text_fit(frame, f"{state} | {display_reason}", 25, color, .58)
            put_text_fit(frame, ALERT_POLICIES[state]["message"], 49, color)
            if debug_enabled:
                put_text_fit(frame, f"Eyes: {monitor.eye_status()}  EAR: {ear if ear is not None else 'n/a'}  PERCLOS: {monitor.perclos():.2f}", 73, (245, 245, 245))
                put_text_fit(frame, f"Head pitch: {pitch_delta:.1f} deg  Head turn: {yaw_delta:.1f} deg  Yawns: {monitor.yawn_events()}", 97, (245, 245, 245))
                put_text_fit(frame, f"FPS: {fps:.1f}  Frame: {frame_ms:.1f} ms  YOLO: {detector.last_latency_ms:.1f} ms  Eye coverage: {monitor.eye_coverage():.0%}", 121, (245, 245, 245))
                camera_label = "CLEAR" if camera_clear else "DARK / BLURRED"
                put_text_fit(frame, f"Camera: {camera_label}  Calibration: {'READY' if calibrated else 'LOOK FORWARD, EYES OPEN'}", 145, (245, 245, 245))
            if active_face_box is not None:
                cv2.rectangle(frame, active_face_box[:2], active_face_box[2:], (255, 180, 0), 2)
            if cfg.get("show_detection_boxes", True):
                colors = {"Eyeclosed": (40, 70, 240), "Eyeopen": (60, 190, 70), "Yawn": (0, 180, 255)}
                for name, confidence, (x1, y1, x2, y2) in detections:
                    col = colors.get(name, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(frame, f"{name} {confidence:.2f}", (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, .48, col, 1)
            if capture_enabled:
                encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if encoded_ok:
                    clean_ok, clean_encoded = cv2.imencode(".jpg", clean_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    payload = encoded.tobytes()
                    clean_payload = clean_encoded.tobytes() if clean_ok else payload
                    clip_buffer.append((wall_t, payload, clean_payload))
                    clip_bytes += len(payload) + len(clean_payload)
                    while clip_buffer and (clip_buffer[0][0] < wall_t - clip_seconds or clip_bytes > clip_max_bytes):
                        _, displayed, clean = clip_buffer.popleft()
                        clip_bytes -= len(displayed) + len(clean)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug_enabled = not debug_enabled
            if key == ord("f") and capture_enabled and clip_buffer:
                output_dir = Path(cfg.get("false_alarm_dir", cfg["session_log_dir"]))
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"false_alarm_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                clean_path = output_path.with_name(output_path.stem + "_clean.mp4")
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (w, h))
                if not writer.isOpened():
                    writer.release()
                    print("Could not save review clip: this OpenCV installation cannot write mp4v video.")
                    continue
                clean_writer = cv2.VideoWriter(str(clean_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (w, h))
                if not clean_writer.isOpened():
                    writer.release()
                    clean_writer.release()
                    print("Could not save the clean review clip: this OpenCV installation cannot write mp4v video.")
                    continue
                saved = 0
                try:
                    for _, display_jpg, clean_jpg in clip_buffer:
                        displayed = cv2.imdecode(np.frombuffer(display_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        clean = cv2.imdecode(np.frombuffer(clean_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if displayed is not None and clean is not None:
                            writer.write(displayed)
                            clean_writer.write(clean)
                            saved += 1
                finally:
                    writer.release()
                    clean_writer.release()
                labels_path.parent.mkdir(parents=True, exist_ok=True)
                needs_header = not labels_path.exists() or labels_path.stat().st_size == 0
                with labels_path.open("a", newline="", encoding="utf-8") as label_file:
                    label_writer = csv.writer(label_file)
                    if needs_header:
                        label_writer.writerow(["filename", "label"])
                    label_writer.writerow([clean_path.name, ""])
                print(f"Review clip saved ({saved} frames): {output_path}")
                print(f"Clean clip: {clean_path}; add its label to {labels_path}")
    finally:
        alarm.close()
        face_extractor.close()
        cap.release()
        cv2.destroyAllWindows()
        metrics.close()
    print(f"Session metrics: {metrics.path}")


if __name__ == "__main__":
    main()
