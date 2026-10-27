"""One readable place for drowsiness and alarm timing defaults."""

# An eye must remain closed for this long before the state reaches DROWSY.
CLOSED_EYE_WARNING_SECONDS = 2.0
# A longer eye closure raises the state to CRITICAL.
CLOSED_EYE_CRITICAL_SECONDS = 4.0
# Unknown eyes on a visible face become a conservative warning after this delay.
UNKNOWN_EYE_WARNING_SECONDS = 1.0
# Require stable evidence before changing the eye state.
EYE_CLOSED_FRAME_COUNT = 3
EYE_OPEN_FRAME_COUNT = 5
EYE_OPEN_STABLE_SECONDS = 0.3
# Alarm can clear only after all safe signals remain true for this long.
SAFE_CLEAR_SECONDS = 2.0
MINIMUM_ALARM_SECONDS = 3.0
# Face and head-pose alert thresholds.
FACE_MISSING_WARNING_SECONDS = 3.0
HEAD_DOWN_ANGLE_DEGREES = 25.0
HEAD_DOWN_WARNING_SECONDS = 2.0
HEAD_DOWN_CRITICAL_SECONDS = 4.0
HEAD_TURN_ANGLE_DEGREES = 35.0
HEAD_TURN_WARNING_SECONDS = 3.0
# Landmark calibration and camera quality thresholds.
CALIBRATION_SECONDS = 5.0
MIN_CALIBRATION_FRAMES = 10
MIN_CALIBRATION_EAR = 0.16
EAR_CLOSED_BASELINE_FRACTION = 0.72
MINIMUM_BRIGHTNESS = 30.0
MINIMUM_SHARPNESS = 15.0
# Run YOLO every second frame when configured for faster webcam processing.
DEFAULT_INFERENCE_INTERVAL = 2
UNKNOWN_STATE_RESET_SECONDS = 1.0
HEAD_DOWN_DIRECTION = -1.0
CAMERA_LOST_ALARM_SECONDS = 2.0
OVERLAY_FONT_SCALE = 0.5
AUDIO_SAMPLE_RATE = 22050
AUDIO_TONE_SECONDS = 0.22
AUDIO_FADE_SECONDS = 0.015

# Numeric project settings are merged under config.yaml values at startup.
DEFAULT_RUNTIME_CONFIG = {
    "camera_index": 0, "camera_fps_fallback": 20, "confidence": 0.35,
    "class_confidence": {"Eyeclosed": 0.25, "Eyeopen": 0.35, "Yawn": 0.35},
    "iou": 0.45, "imgsz": 416, "half": False,
    "class_names": ["Eyeclosed", "Eyeopen", "Yawn"], "single_eye_closed_confidence": 0.90,
    "window_seconds": 30.0, "perclos_warning": 0.15, "perclos_drowsy": 0.35,
    "yawn_window_seconds": 60.0, "yawn_min_duration_seconds": 2.0,
    "yawn_debounce_seconds": 2.0, "yawn_events_for_alarm": 2,
    "eye_weight": 0.75, "yawn_weight": 0.25, "score_warning": 0.45,
    "score_drowsy": 0.70, "hysteresis_margin": 0.10,
    "closed_eye_alarm_seconds": CLOSED_EYE_WARNING_SECONDS,
    "closed_eye_gap_tolerance_seconds": 0.35, "yawn_drowsy_hold_seconds": 5.0,
    "show_detection_boxes": True, "false_alarm_capture_enabled": True,
    "false_alarm_buffer_seconds": 10.0, "false_alarm_buffer_max_mb": 80,
    "rest_reminder_alert_count": 3, "yawn_gap_tolerance_seconds": 0.35,
    "driver_roi": [0.0, 0.0, 1.0, 1.0], "minimum_eye_coverage": 0.5,
    "face_box_padding_fraction": 0.25, "eye_pair_max_size_ratio": 2.0,
    "head_down_angle_degrees": HEAD_DOWN_ANGLE_DEGREES,
    "head_down_warning_seconds": HEAD_DOWN_WARNING_SECONDS,
    "head_down_critical_seconds": HEAD_DOWN_CRITICAL_SECONDS,
    "head_turn_angle_degrees": HEAD_TURN_ANGLE_DEGREES,
    "head_turn_warning_seconds": HEAD_TURN_WARNING_SECONDS,
    "unknown_eye_warning_seconds": UNKNOWN_EYE_WARNING_SECONDS,
    "closed_eye_warning_seconds": CLOSED_EYE_WARNING_SECONDS,
    "closed_eye_critical_seconds": CLOSED_EYE_CRITICAL_SECONDS,
    "face_missing_warning_seconds": FACE_MISSING_WARNING_SECONDS,
    "head_drop_weight": 0.1, "yawn_events_for_max": 3,
    "inference_interval_frames": DEFAULT_INFERENCE_INTERVAL,
}
# Alarm levels, their sound pattern, and the same message used on screen.
ALERT_POLICIES = {
    "NORMAL": {"sound": "silent", "frequency": 0, "volume": 0.0, "repeat_seconds": 0.0,
               "message": "No drowsiness signal"},
    "WARNING": {"sound": "beep", "frequency": 880, "volume": 0.18, "repeat_seconds": 3.0,
                "message": "Warning — please stay alert"},
    "DROWSY": {"sound": "repeating", "frequency": 1040, "volume": 0.32, "repeat_seconds": 0.55,
               "message": "Drowsy — take a break"},
    "CRITICAL": {"sound": "urgent", "frequency": 1320, "volume": 0.5, "repeat_seconds": 0.38,
                 "message": "Critical — stop safely and rest"},
    "DRIVER_NOT_FOUND": {"sound": "beep", "frequency": 880, "volume": 0.18, "repeat_seconds": 3.0,
                         "message": "Driver not visible — check camera"},
    "CAMERA_UNCLEAR": {"sound": "beep", "frequency": 880, "volume": 0.18, "repeat_seconds": 3.0,
                       "message": "Camera view unclear — check camera"},
}
ALERT_PRIORITY = {"NORMAL": 0, "WARNING": 1, "DRIVER_NOT_FOUND": 1,
                  "CAMERA_UNCLEAR": 1, "DROWSY": 2, "CRITICAL": 3}
