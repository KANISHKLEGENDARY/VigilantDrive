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
from src.config import (ALERT_POLICIES, CAMERA_LOST_ALARM_SECONDS, CLIP_FRAME_INTERVAL,
                        CLIP_MAX_WIDTH, DEFAULT_INFERENCE_INTERVAL,
                        DEFAULT_RUNTIME_CONFIG,
                        EAR_MAX_YAW_FOR_DECISION, HEAD_DOWN_DIRECTION,
                        MINIMUM_BRIGHTNESS, MINIMUM_SHARPNESS,
                        OVERLAY_FONT_SCALE, OVERLAY_SMOOTHING_ALPHA, CALIBRATION_SECONDS)
from src.eye_logic import (EyeDebouncer, combine_eye_signals,
                           ear_allows_yolo_override, ear_is_surely_closed,
                           ear_is_usable_at_yaw, ear_says_closed)
from src.face_signals import FaceCalibration, FaceSignalExtractor, angle_delta


def put_text_fit(frame, text: str, y: int, color, font_scale: float = OVERLAY_FONT_SCALE) -> None:
    """Draw one overlay row with a font small enough to fit the video width."""
    available_width = frame.shape[1] - 36
    scale = font_scale
    while scale > 0.25 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > available_width:
        scale -= 0.025
    cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def smooth_display(previous: float | None, current: float) -> float:
    """Reduce visual jitter without changing the values used by alert logic."""
    if previous is None:
        return current
    alpha = OVERLAY_SMOOTHING_ALPHA
    return alpha * current + (1.0 - alpha) * previous


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
    calibration_reported = False
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
    display_pitch = display_yaw = display_fps = display_frame_ms = display_yolo_ms = None
    display_capture_ms = display_face_mesh_ms = display_yolo_stage_ms = None
    display_jpeg_ms = display_drawing_ms = display_display_ms = None
    last_stage_times: dict[str, float] = {}
    last_complete_frame_ms = 0.0
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
            capture_started = time.perf_counter()
            ok, frame = cap.read()
            capture_ms = (time.perf_counter() - capture_started) * 1000
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
            face_started = time.perf_counter()
            try:
                face_measurement = face_extractor.measure(frame)
            except (ValueError, cv2.error):
                face_measurement = None
            face_mesh_ms = (time.perf_counter() - face_started) * 1000
            face_box = face_measurement.box if face_measurement is not None else None
            face_visible = face_measurement is not None
            face_not_found = not face_visible
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            camera_clear = brightness >= MINIMUM_BRIGHTNESS and sharpness >= MINIMUM_SHARPNESS
            yolo_started = time.perf_counter()
            if face_visible and (frame_index - 1) % inference_interval == 0:
                last_detections = detector.detect(frame)
            yolo_stage_ms = ((time.perf_counter() - yolo_started) * 1000
                             if face_visible and (frame_index - 1) % inference_interval == 0 else 0.0)
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
            pitch_delta = 0.0
            yaw_delta = 0.0
            if calibrated and face_measurement is not None and not args.no_calibration:
                # This multiplier makes positive pitch mean chin down on this camera.
                pitch_delta = angle_delta(face_measurement.pitch_degrees, calibration.pitch_baseline) * HEAD_DOWN_DIRECTION
                yaw_delta = angle_delta(face_measurement.yaw_degrees, calibration.yaw_baseline)
            ear_usable = ear_is_usable_at_yaw(
                yaw_delta, float(cfg.get("ear_max_yaw_for_decision", EAR_MAX_YAW_FOR_DECISION)))
            ear_closed = (ear_says_closed(ear, calibration.ear_baseline)
                          if calibrated and not args.no_calibration and ear_usable else None)
            ear_sure_closed = (ear_is_surely_closed(ear, calibration.ear_baseline)
                               if calibrated and not args.no_calibration and ear_usable else False)
            ear_yolo_override = (ear_allows_yolo_override(ear, calibration.ear_baseline)
                                 if calibrated and not args.no_calibration and ear_usable else False)
            eye_candidate = combine_eye_signals(yolo_eye_state, ear_closed, ear_sure_closed,
                                                ear_yolo_override)
            eye_state = eye_debouncer.update(eye_candidate, t)
            if calibrated and not args.no_calibration and not calibration_reported:
                cutoff = calibration.ear_baseline * cfg.get("ear_closed_baseline_fraction", 0.60)
                print(f"Eye calibration ready: EAR baseline={calibration.ear_baseline:.3f}, closed cutoff={cutoff:.3f}")
                calibration_reported = True
            yawn = calibrated and any(name == "Yawn" for name, _, _ in detections)
            monitor_eye_state = (False if not calibrated and face_visible else
                                 None if eye_candidate is None else eye_state)
            monitor.update(t, monitor_eye_state, yawn,
                           face_visible=face_visible, head_pitch=pitch_delta,
                           head_yaw=yaw_delta, camera_clear=camera_clear)
            incoming_state = monitor.alert_level() if calibrated else "NORMAL"
            safe_to_clear = (calibrated and face_visible and camera_clear and eye_state is False
                             and pitch_delta < cfg["head_down_angle_degrees"]
                             and abs(yaw_delta) < cfg["head_turn_angle_degrees"])
            state = alarm.update(incoming_state, t, safe_to_clear=safe_to_clear)
            alarm_active = state != "NORMAL"
            display_reason = ("Calibrating - look forward with eyes open" if not calibrated else
                              monitor.reason() if state == incoming_state
                             else "Hold alert until eyes are open, head is up, and face is visible")
            if alarm_active and not last_alarm_active:
                alert_starts.append(wall_t)
            last_alarm_active = alarm_active
            while alert_starts and alert_starts[0] < wall_t - 1800:
                alert_starts.popleft()
            fps = metrics.estimate_fps(wall_t)
            display_pitch = smooth_display(display_pitch, pitch_delta)
            display_yaw = smooth_display(display_yaw, yaw_delta)
            display_fps = smooth_display(display_fps, fps)
            display_frame_ms = smooth_display(display_frame_ms, last_complete_frame_ms)
            display_yolo_ms = smooth_display(display_yolo_ms, detector.last_latency_ms)
            display_capture_ms = smooth_display(display_capture_ms, capture_ms)
            display_face_mesh_ms = smooth_display(display_face_mesh_ms, face_mesh_ms)
            display_yolo_stage_ms = smooth_display(display_yolo_stage_ms, yolo_stage_ms)
            display_jpeg_ms = smooth_display(display_jpeg_ms, last_stage_times.get("jpeg_encode_ms", 0.0))
            display_drawing_ms = smooth_display(display_drawing_ms, last_stage_times.get("drawing_ms", 0.0))
            display_display_ms = smooth_display(display_display_ms, last_stage_times.get("display_ms", 0.0))
            drawing_started = time.perf_counter()
            color = {"NORMAL": (50, 200, 50), "WARNING": (0, 180, 255), "DROWSY": (0, 0, 255),
                     "CRITICAL": (0, 0, 255), "DRIVER_NOT_FOUND": (0, 180, 255),
                     "CAMERA_UNCLEAR": (0, 180, 255)}[state]
            cv2.rectangle(frame, (0, 0), (frame.shape[1], min(210, frame.shape[0])), (18, 18, 18), -1)
            put_text_fit(frame, f"{state} | {display_reason}", 25, color, .58)
            put_text_fit(frame, ALERT_POLICIES[state]["message"], 49, color)
            if debug_enabled:
                ear_display = f"{ear:.2f}" if ear is not None else "n/a"
                ear_baseline = calibration.ear_baseline if calibrated and not args.no_calibration else None
                ear_cutoff = (ear_baseline * cfg.get("ear_closed_baseline_fraction", 0.60)
                              if ear_baseline is not None else None)
                cutoff_display = f"/{ear_cutoff:.2f}" if ear_cutoff is not None else ""
                put_text_fit(frame, f"Eyes: {monitor.eye_status()}  EAR: {ear_display}{cutoff_display}  PERCLOS: {monitor.perclos():.2f}", 73, (245, 245, 245))
                put_text_fit(frame, f"Head pitch (+ = down): {display_pitch:.1f} deg  Head turn: {display_yaw:.1f} deg  Yawns: {monitor.yawn_events()}", 97, (245, 245, 245))
                put_text_fit(frame, f"FPS: {display_fps:.1f}  Frame: {display_frame_ms:.1f} ms  YOLO: {display_yolo_ms:.1f} ms  Eye coverage: {monitor.eye_coverage():.0%}", 121, (245, 245, 245))
                camera_label = "CLEAR" if camera_clear else "DARK / BLURRED"
                if calibrated:
                    calibration_label = "READY"
                elif calibration.failed:
                    calibration_label = "FAILED - look forward with eyes open"
                else:
                    calibration_label = "CALIBRATING - eyes open, face forward"
                put_text_fit(frame, f"Camera: {camera_label}  {calibration_label}", 145, (245, 245, 245))
                put_text_fit(frame, f"Stage ms - capture: {display_capture_ms:.1f}  face mesh: {display_face_mesh_ms:.1f}  YOLO: {display_yolo_stage_ms:.1f}", 169, (245, 245, 245))
                put_text_fit(frame, f"Stage ms - JPEG: {display_jpeg_ms:.1f}  draw: {display_drawing_ms:.1f}  display: {display_display_ms:.1f}", 193, (245, 245, 245))
            if active_face_box is not None:
                cv2.rectangle(frame, active_face_box[:2], active_face_box[2:], (255, 180, 0), 2)
            if cfg.get("show_detection_boxes", True):
                colors = {"Eyeclosed": (40, 70, 240), "Eyeopen": (60, 190, 70), "Yawn": (0, 180, 255)}
                for name, confidence, (x1, y1, x2, y2) in detections:
                    col = colors.get(name, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(frame, f"{name} {confidence:.2f}", (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, .48, col, 1)
            drawing_ms = (time.perf_counter() - drawing_started) * 1000
            jpeg_started = time.perf_counter()
            jpeg_encode_ms = 0.0
            if capture_enabled:
                clip_interval = max(1, int(cfg.get("clip_frame_interval", CLIP_FRAME_INTERVAL)))
                if frame_index % clip_interval == 0:
                    height, width = frame.shape[:2]
                    max_width = int(cfg.get("clip_max_width", CLIP_MAX_WIDTH))
                    clip_scale = min(1.0, max_width / width)
                    clip_size = (max(1, int(width * clip_scale)), max(1, int(height * clip_scale)))
                    display_clip_frame = (frame if clip_scale == 1.0 else
                                          cv2.resize(frame, clip_size, interpolation=cv2.INTER_AREA))
                    clean_clip_frame = (clean_frame if clip_scale == 1.0 else
                                        cv2.resize(clean_frame, clip_size, interpolation=cv2.INTER_AREA))
                    encoded_ok, encoded = cv2.imencode(".jpg", display_clip_frame,
                                                       [cv2.IMWRITE_JPEG_QUALITY, 65])
                    clean_ok, clean_encoded = cv2.imencode(".jpg", clean_clip_frame,
                                                           [cv2.IMWRITE_JPEG_QUALITY, 65])
                    if encoded_ok and clean_ok:
                        payload = encoded.tobytes()
                        clean_payload = clean_encoded.tobytes()
                        clip_buffer.append((wall_t, payload, clean_payload))
                        clip_bytes += len(payload) + len(clean_payload)
                        while clip_buffer and (clip_buffer[0][0] < wall_t - clip_seconds or clip_bytes > clip_max_bytes):
                            _, displayed, clean = clip_buffer.popleft()
                            clip_bytes -= len(displayed) + len(clean)
                jpeg_encode_ms = (time.perf_counter() - jpeg_started) * 1000
            display_started = time.perf_counter()
            cv2.imshow(window_name, frame)
            display_ms = (time.perf_counter() - display_started) * 1000
            frame_ms = (time.perf_counter() - frame_started) * 1000
            stage_times = {"capture_ms": capture_ms, "face_mesh_ms": face_mesh_ms,
                           "yolo_stage_ms": yolo_stage_ms, "jpeg_encode_ms": jpeg_encode_ms,
                           "drawing_ms": drawing_ms, "display_ms": display_ms}
            metrics.record(wall_t, monitor.score(), monitor.perclos(), monitor.yawn_events(), state,
                           detector.last_latency_ms, monitor.eye_status(), display_reason, alarm_active,
                           monitor.eye_coverage(), monitor.head_drop_active, ear,
                           pitch_delta, yaw_delta, state, frame_ms, stage_times)
            last_stage_times = stage_times
            last_complete_frame_ms = frame_ms
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug_enabled = not debug_enabled
            if key == ord("c"):
                calibration.reset()
                monitor.reset()
                eye_debouncer.reset()
                alarm.force_normal()
                calibration_reported = False
            if key == ord("f") and capture_enabled and clip_buffer:
                output_dir = Path(cfg.get("false_alarm_dir", cfg["session_log_dir"]))
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"false_alarm_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                clean_path = output_path.with_name(output_path.stem + "_clean.mp4")
                h, w = frame.shape[:2]
                clip_interval = max(1, int(cfg.get("clip_frame_interval", CLIP_FRAME_INTERVAL)))
                max_width = int(cfg.get("clip_max_width", CLIP_MAX_WIDTH))
                clip_scale = min(1.0, max_width / w)
                clip_size = (max(1, int(w * clip_scale)), max(1, int(h * clip_scale)))
                clip_fps = max(1.0, source_fps / clip_interval)
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), clip_fps, clip_size)
                if not writer.isOpened():
                    writer.release()
                    print("Could not save review clip: this OpenCV installation cannot write mp4v video.")
                    continue
                clean_writer = cv2.VideoWriter(str(clean_path), cv2.VideoWriter_fourcc(*"mp4v"), clip_fps, clip_size)
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
