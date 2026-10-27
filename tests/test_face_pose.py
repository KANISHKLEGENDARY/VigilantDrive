import cv2
import numpy as np
from src.face_signals import _head_angles, angle_delta


def projected_landmarks(rotation_axis: int, angle_degrees: float):
    """Build known synthetic face points to check the solvePnP angle axes."""
    ids = (1, 152, 33, 263, 61, 291)
    model = np.asarray([(0, 0, 0), (0, -330, -65), (-225, 170, -135),
                        (225, 170, -135), (-150, -150, -125), (150, -150, -125)], dtype=np.float64)
    camera = np.asarray([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float64)
    rotation = np.zeros(3, dtype=np.float64)
    rotation[rotation_axis] = np.deg2rad(angle_degrees)
    image, _ = cv2.projectPoints(model, rotation, np.asarray([0, 0, 1000.0]), camera, np.zeros((4, 1)))
    points = [(0.0, 0.0) for _ in range(300)]
    for landmark_id, point in zip(ids, image.reshape(-1, 2)):
        points[landmark_id] = tuple(point)
    return points


def test_solvepnp_reports_nod_as_pitch_and_side_turn_as_yaw():
    pitch, yaw = _head_angles(projected_landmarks(0, -15), 640, 480)
    assert abs(pitch + 15) < 0.1
    assert abs(yaw) < 0.1
    pitch, yaw = _head_angles(projected_landmarks(1, 15), 640, 480)
    assert abs(pitch) < 0.1
    assert abs(yaw - 15) < 0.1


def test_angle_difference_handles_wrapping_at_180_degrees():
    assert angle_delta(-179.0, 179.0) == 2.0
