"""Ultralytics adapter plus conservative interpretation of eye detections."""
from __future__ import annotations
import time


class Detector:
    def __init__(self, weights: str, device: str = "0", conf: float = 0.35, iou: float = 0.45,
                 class_confidence: dict[str, float] | None = None, imgsz: int = 640,
                 half: bool = False, expected_classes: list[str] | None = None):
        try:
            import ultralytics.nn.tasks as tasks
            from model.fcbam import FCBAM
            tasks.FCBAM = FCBAM
            from ultralytics import YOLO
        except ImportError as exc:
            detail = str(exc)
            if "Application Control policy" in detail or "blocked this file" in detail:
                message = (
                    "Windows Application Control blocked a native dependency while importing Ultralytics "
                    "(the traceback identifies Pillow's PIL._imaging module). This is an operating-system "
                    "policy block, not an FCBAM or checkpoint error. Check the CodeIntegrity/Operational log "
                    "or ask the device administrator to approve the package/environment. Original error: "
                    f"{detail}"
                )
            else:
                message = f"Could not import Ultralytics or one of its dependencies: {detail}"
            raise RuntimeError(message) from exc
        except AttributeError as exc:
            raise RuntimeError(f"Could not register the custom FCBAM layer with Ultralytics: {exc}") from exc
        self.model = YOLO(weights)
        if str(device).lower() == "auto":
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        self.device, self.conf, self.iou = device, conf, iou
        self.imgsz, self.half = int(imgsz), bool(half and str(device) != "cpu")
        self.class_confidence = class_confidence or {}
        self.inference_conf = min([conf, *self.class_confidence.values()])
        self.last_latency_ms = 0.0
        self.names = self.model.names
        names = set(self.names.values()) if isinstance(self.names, dict) else set(self.names)
        required = set(expected_classes or ("Eyeclosed", "Eyeopen", "Yawn"))
        missing = required - names
        if missing:
            raise ValueError(f"Checkpoint classes do not match runtime config; missing: {sorted(missing)}; found: {sorted(names)}")

    def detect(self, frame) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        start = time.perf_counter()
        result = self.model.predict(frame, device=self.device, conf=self.inference_conf,
                                    iou=self.iou, imgsz=self.imgsz, half=self.half, verbose=False)[0]
        self.last_latency_ms = (time.perf_counter() - start) * 1000
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            name = self.names[class_id]
            confidence = float(box.conf.item())
            if confidence < self.class_confidence.get(name, self.conf):
                continue
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            detections.append((name, confidence, (x1, y1, x2, y2)))
        return detections


def infer_eye_state(detections, single_eye_closed_confidence: float = 0.90,
                    max_eye_size_ratio: float = 2.0) -> bool | None:
    """Return closed/open/unknown using distinct eye boxes, not any closed box.

    Conflicting same-location class predictions are deduplicated. Two visible
    eyes must agree to declare closure; one visible open eye is treated as open,
    while one closed eye must be exceptionally confident. This is a conservative
    heuristic and does not replace driver-face tracking.
    """
    eyes = [d for d in detections if d[0] in ("Eyeclosed", "Eyeopen")]
    eyes.sort(key=lambda item: item[1], reverse=True)
    distinct = []
    for candidate in eyes:
        _, _, (x1, y1, x2, y2) = candidate
        area = max(1, (x2 - x1) * (y2 - y1))
        duplicate = False
        for selected in distinct:
            _, _, (a1, b1, a2, b2) = selected
            intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
            selected_area = max(1, (a2 - a1) * (b2 - b1))
            if intersection / min(area, selected_area) >= 0.5:
                duplicate = True
                break
        if not duplicate:
            distinct.append(candidate)
    # Pair eyes by image geometry so two detections from separate faces are
    # less likely to be treated as one closed pair.
    pairs = []
    for i, first in enumerate(distinct):
        _, first_conf, (x1, y1, x2, y2) = first
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        w1, h1 = max(1, x2 - x1), max(1, y2 - y1)
        for second in distinct[i + 1:]:
            _, second_conf, (a1, b1, a2, b2) = second
            cx2, cy2 = (a1 + a2) / 2, (b1 + b2) / 2
            w2, h2 = max(1, a2 - a1), max(1, b2 - b1)
            horizontal = abs(cx1 - cx2)
            mean_width = (w1 + w2) / 2
            mean_height = (h1 + h2) / 2
            size_ratio = max(w1 / w2, w2 / w1, h1 / h2, h2 / h1)
            if (0.5 * mean_width <= horizontal <= 5.0 * mean_width
                    and abs(cy1 - cy2) <= 0.8 * mean_height
                    and size_ratio <= max_eye_size_ratio):
                pairs.append((first_conf + second_conf, first, second))
    if pairs:
        _, first, second = max(pairs, key=lambda item: item[0])
        return first[0] == "Eyeclosed" and second[0] == "Eyeclosed"
    if len(distinct) == 1:
        if distinct[0][0] == "Eyeopen":
            return False
        if distinct[0][1] >= single_eye_closed_confidence:
            return True
    return None


def load_face_detector():
    """Load OpenCV's bundled frontal-face cascade without extra packages."""
    import cv2
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("OpenCV's bundled frontal-face cascade could not be loaded")
    return cascade


def find_largest_face(frame, cascade, scale_factor: float = 1.1, min_neighbors: int = 5,
                      min_face_pixels: int = 60):
    """Return the largest detected face box, or None when no face is found."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                     minSize=(min_face_pixels, min_face_pixels))
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    return int(x), int(y), int(x + width), int(y + height)


def expand_face_box(box, frame_shape, padding: float = 0.25):
    """Expand a face rectangle so the driver's eyes and mouth fit inside it."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    pad_x, pad_y = int((x2 - x1) * padding), int((y2 - y1) * padding)
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def keep_detections_in_box(detections, box):
    """Keep detections whose centers lie inside the selected driver's face region."""
    if box is None:
        return detections
    x1, y1, x2, y2 = box
    return [item for item in detections
            if x1 <= item[2][0] and y1 <= item[2][1]
            and item[2][2] <= x2 and item[2][3] <= y2]


def keep_driver_region(detections, frame_shape, roi):
    """Keep detections whose box centers fall inside a normalized ROI.

    ROI is ``[left, top, right, bottom]`` with coordinates from 0 to 1.
    This is a configurable camera-region filter, not automatic driver tracking.
    """
    if roi is None:
        return detections
    if len(roi) != 4 or any(not 0 <= float(v) <= 1 for v in roi):
        raise ValueError("driver_roi must contain four normalized values from 0 to 1")
    left, top, right, bottom = map(float, roi)
    if right <= left or bottom <= top:
        raise ValueError("driver_roi right/bottom must be greater than left/top")
    height, width = frame_shape[:2]
    x_min, x_max, y_min, y_max = left * width, right * width, top * height, bottom * height
    return [item for item in detections
            if x_min <= (item[2][0] + item[2][2]) / 2 <= x_max
            and y_min <= (item[2][1] + item[2][3]) / 2 <= y_max]

