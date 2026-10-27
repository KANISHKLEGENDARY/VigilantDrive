"""Extract eye openness and head direction from ready-made face landmarks."""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from statistics import median

import cv2
import numpy as np
from src.config import CALIBRATION_SECONDS, MIN_CALIBRATION_EAR, MIN_CALIBRATION_FRAMES


@dataclass
class FaceSignals:
    """Measurements from the largest face found in one camera frame."""
    ear: float
    pitch_degrees: float
    yaw_degrees: float
    box: tuple[int, int, int, int]


def _distance(first, second) -> float:
    """Measure the straight-line pixel distance between two landmark points."""
    return hypot(first[0] - second[0], first[1] - second[1])


def angle_delta(current: float, baseline: float) -> float:
    """Return the shortest signed angle change across the -180/180 boundary."""
    return (current - baseline + 180.0) % 360.0 - 180.0


def _angle_median(values: list[float]) -> float:
    """Find a median angle after unwrapping values around the first sample."""
    reference = values[0]
    return median([reference + angle_delta(value, reference) for value in values])


def _eye_aspect_ratio(points: list[tuple[float, float]], indices: tuple[int, ...]) -> float:
    """Calculate one eye's opening ratio; a lower number usually means closed."""
    corner_a, upper_a, upper_b, corner_b, lower_b, lower_a = [points[i] for i in indices]
    vertical = _distance(upper_a, lower_a) + _distance(upper_b, lower_b)
    return vertical / max(2 * _distance(corner_a, corner_b), 1.0)


def _head_angles(points: list[tuple[float, float]], width: int, height: int) -> tuple[float, float]:
    """Estimate head pitch and yaw with six facial points and solvePnP."""
    landmark_ids = (1, 152, 33, 263, 61, 291)
    image_points = np.asarray([points[index] for index in landmark_ids], dtype=np.float64)
    model_points = np.asarray([(0, 0, 0), (0, -330, -65), (-225, 170, -135),
                               (225, 170, -135), (-150, -150, -125), (150, -150, -125)], dtype=np.float64)
    focal_length = float(width)
    camera = np.asarray([[focal_length, 0, width / 2], [0, focal_length, height / 2],
                         [0, 0, 1]], dtype=np.float64)
    solved, rotation, _ = cv2.solvePnP(model_points, image_points, camera,
                                       np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not solved:
        raise ValueError("Could not estimate head direction from landmarks")
    matrix, _ = cv2.Rodrigues(rotation)
    pitch = degrees(atan2(matrix[2, 1], matrix[2, 2]))
    yaw = degrees(atan2(-matrix[2, 0], max(1e-6, np.hypot(matrix[0, 0], matrix[1, 0]))))
    return pitch, yaw


class FaceSignalExtractor:
    """Run MediaPipe Face Mesh and return measurements for its largest face."""
    def __init__(self):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("Face landmarks need MediaPipe. Install it with: python -m pip install -r requirements.txt") from exc
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=2, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def measure(self, frame) -> FaceSignals | None:
        """Return EAR, pitch, and yaw for the largest visible face, if any."""
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)
        faces = results.multi_face_landmarks or []
        if not faces:
            return None
        face = max(faces, key=lambda item: ((max(point.x for point in item.landmark) - min(point.x for point in item.landmark))
                                           * (max(point.y for point in item.landmark) - min(point.y for point in item.landmark))))
        points = [(point.x * width, point.y * height) for point in face.landmark]
        left = _eye_aspect_ratio(points, (33, 160, 158, 133, 153, 144))
        right = _eye_aspect_ratio(points, (362, 385, 387, 263, 373, 380))
        pitch, yaw = _head_angles(points, width, height)
        box = (max(0, int(min(point[0] for point in points))), max(0, int(min(point[1] for point in points))),
               min(width, int(max(point[0] for point in points))),
               min(height, int(max(point[1] for point in points))))
        return FaceSignals((left + right) / 2, pitch, yaw, box)

    def close(self) -> None:
        """Release MediaPipe's face-landmark resources."""
        self.mesh.close()


class FaceCalibration:
    """Collect a short open-eye, forward-facing baseline at startup."""
    def __init__(self, seconds: float = CALIBRATION_SECONDS):
        self.seconds = seconds
        self.started_at: float | None = None
        self.ears: list[float] = []
        self.pitches: list[float] = []
        self.yaws: list[float] = []
        self.ear_baseline: float | None = None
        self.pitch_baseline: float | None = None
        self.yaw_baseline: float | None = None
        self.ready = False

    def update(self, now: float, signals: FaceSignals | None) -> bool:
        """Collect measurements until calibration time passes; return when ready."""
        if self.ready:
            return True
        if self.started_at is None:
            self.started_at = now
        if signals is not None and signals.ear >= MIN_CALIBRATION_EAR:
            self.ears.append(signals.ear)
            self.pitches.append(signals.pitch_degrees)
            self.yaws.append(signals.yaw_degrees)
        if now - self.started_at < self.seconds or len(self.ears) < MIN_CALIBRATION_FRAMES:
            return False
        self.ear_baseline = median(self.ears)
        self.pitch_baseline = _angle_median(self.pitches)
        self.yaw_baseline = _angle_median(self.yaws)
        self.ready = True
        return True
