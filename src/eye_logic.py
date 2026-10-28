"""Combine the two eye signals and reject one-frame detector mistakes."""
from __future__ import annotations

from src.config import (EYE_CLOSED_FRAME_COUNT, EYE_OPEN_FRAME_COUNT,
                        EYE_CLOSED_STABLE_SECONDS, EYE_OPEN_STABLE_SECONDS,
                        EAR_CLOSED_BASELINE_FRACTION, EAR_SURE_CLOSED_FRACTION,
                        EAR_MAX_YAW_FOR_DECISION)
from src.config import UNKNOWN_STATE_RESET_SECONDS


def combine_eye_signals(yolo_closed: bool | None, ear_closed: bool | None,
                        ear_sure_closed: bool = False) -> bool | None:
    """Use strong EAR as decisive, and require YOLO to confirm borderline EAR."""
    if ear_sure_closed:
        return True
    if ear_closed is False:
        return False
    if ear_closed is None:
        return yolo_closed
    if yolo_closed is False:
        return None
    return True


class EyeDebouncer:
    """Keep eye state steady across brief misses and single wrong detections."""
    def __init__(self, closed_frames: int = EYE_CLOSED_FRAME_COUNT,
                 open_frames: int = EYE_OPEN_FRAME_COUNT,
                 open_seconds: float = EYE_OPEN_STABLE_SECONDS,
                 closed_seconds: float = EYE_CLOSED_STABLE_SECONDS):
        self.closed_frames = closed_frames
        self.open_frames = open_frames
        self.open_seconds = open_seconds
        self.closed_seconds = closed_seconds
        self.state: bool | None = None
        self.candidate: bool | None = None
        self.candidate_frames = 0
        self.candidate_since: float | None = None
        self.unknown_since: float | None = None

    def update(self, candidate: bool | None, now: float) -> bool | None:
        """Return the stable state, requiring repeated evidence to change it."""
        if candidate is None:
            if self.unknown_since is None:
                self.unknown_since = now
            self.candidate = None
            self.candidate_frames = 0
            self.candidate_since = None
            if now - self.unknown_since >= UNKNOWN_STATE_RESET_SECONDS:
                self.state = None
            return self.state
        self.unknown_since = None
        if candidate != self.candidate:
            self.candidate = candidate
            self.candidate_frames = 1
            self.candidate_since = now
        else:
            self.candidate_frames += 1
        required_frames = self.closed_frames if candidate else self.open_frames
        elapsed = now - self.candidate_since if self.candidate_since is not None else 0.0
        stable_duration = elapsed >= (self.closed_seconds if candidate else self.open_seconds)
        if self.candidate_frames >= required_frames and stable_duration:
            self.state = candidate
        return self.state

    def reset(self) -> None:
        """Forget eye history after a camera interruption or a new clip."""
        self.state = self.candidate = None
        self.candidate_frames = 0
        self.candidate_since = None
        self.unknown_since = None


def ear_says_closed(ear: float | None, baseline: float | None) -> bool | None:
    """Compare current eye openness with the driver's calibrated open baseline."""
    if ear is None or baseline is None:
        return None
    return ear < baseline * EAR_CLOSED_BASELINE_FRACTION


def ear_is_surely_closed(ear: float | None, baseline: float | None) -> bool:
    """Identify a strongly closed eye even if YOLO disagrees."""
    return (ear is not None and baseline is not None
            and ear < baseline * EAR_SURE_CLOSED_FRACTION)


def ear_is_usable_at_yaw(yaw_delta: float, max_yaw: float = EAR_MAX_YAW_FOR_DECISION) -> bool:
    """Avoid using EAR when a sideways turn makes one eye measurement unreliable."""
    return abs(yaw_delta) <= max_yaw
