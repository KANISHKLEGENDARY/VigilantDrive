"""Time-based drowsiness signals independent of the camera and detector."""
from __future__ import annotations

from collections import deque
from src.config import DEFAULT_RUNTIME_CONFIG


class DrowsinessMonitor:
    """Calculate PERCLOS, sustained yawn events, score and alert state.

    eye_closed accepts True (closed), False (open), or None (not visible).
    Unknown observations are excluded from PERCLOS instead of being counted as
    open. Short gaps can bridge an ongoing closure but cannot start one.
    """

    def __init__(self, config: dict):
        self.cfg = dict(DEFAULT_RUNTIME_CONFIG)
        self.cfg.update(config)
        self.samples: deque[tuple[float, bool | None]] = deque()
        self.risk_samples: deque[tuple[float, bool | None]] = deque()
        self.events: deque[float] = deque()
        self.current_state = "NORMAL"
        self.last_t: float | None = None
        self.session_start: float | None = None
        self.eye_closed_since: float | None = None
        self.eye_gap_since: float | None = None
        self.last_eye_state: bool | None = None
        self.yawn_since: float | None = None
        self.yawn_gap_since: float | None = None
        self.yawn_counted = False
        self.last_yawn_event = float("-inf")
        self._reason = "No drowsiness signal"
        self.head_drop_since: float | None = None
        self.head_drop_active = False
        self.unknown_eye_since: float | None = None
        self.face_missing_since: float | None = None
        self.head_down_since: float | None = None
        self.head_turn_since: float | None = None
        self.head_pitch = 0.0
        self.head_yaw = 0.0
        self.face_visible = True
        self.camera_clear = True

    def reset(self) -> None:
        """Clear temporal state after a camera interruption or new clip."""
        self.samples.clear()
        self.risk_samples.clear()
        self.events.clear()
        self.current_state = "NORMAL"
        self.last_t = None
        self.session_start = None
        self.eye_closed_since = None
        self.eye_gap_since = None
        self.last_eye_state = None
        self.yawn_since = None
        self.yawn_gap_since = None
        self.yawn_counted = False
        self.last_yawn_event = float("-inf")
        self._reason = "No drowsiness signal"
        self.head_drop_since = None
        self.head_drop_active = False
        self.unknown_eye_since = None
        self.face_missing_since = None
        self.head_down_since = None
        self.head_turn_since = None
        self.head_pitch = self.head_yaw = 0.0
        self.face_visible = True
        self.camera_clear = True

    def update(self, t: float, eye_closed: bool | None, yawn: bool, *, face_visible: bool = True,
               head_pitch: float | None = None, head_yaw: float | None = None,
               camera_clear: bool = True) -> None:
        """Add an observation at monotonic (or source-video) time ``t``."""
        if self.last_t is not None and t < self.last_t:
            raise ValueError("timestamps must be non-decreasing")
        if self.session_start is None:
            self.session_start = t
        self.face_visible = face_visible
        self.camera_clear = camera_clear
        if not face_visible:
            if self.face_missing_since is None:
                self.face_missing_since = t
        else:
            self.face_missing_since = None
        if eye_closed is None and face_visible:
            if self.unknown_eye_since is None:
                self.unknown_eye_since = t
        else:
            self.unknown_eye_since = None
        self.head_pitch = float(head_pitch or 0.0)
        self.head_yaw = float(head_yaw or 0.0)
        if self.head_pitch >= self.cfg["head_down_angle_degrees"]:
            if self.head_down_since is None:
                self.head_down_since = t
        else:
            self.head_down_since = None
        self.head_drop_since = self.head_down_since
        self.head_drop_active = (self.head_down_since is not None and
                                 t - self.head_down_since >= self.cfg["head_down_warning_seconds"])
        if abs(self.head_yaw) >= self.cfg["head_turn_angle_degrees"]:
            if self.head_turn_since is None:
                self.head_turn_since = t
        else:
            self.head_turn_since = None
        eye_closed = None if eye_closed is None else bool(eye_closed)
        self.last_eye_state = eye_closed
        gap = self.cfg["closed_eye_gap_tolerance_seconds"]
        if eye_closed is True:
            if self.eye_closed_since is None:
                self.eye_closed_since = t
            self.eye_gap_since = None
        elif self.eye_closed_since is not None:
            if self.eye_gap_since is None:
                self.eye_gap_since = t
            # A visible open eye or a prolonged unknown interval ends the episode.
            if eye_closed is False or t - self.eye_gap_since > gap:
                self.eye_closed_since = None
                self.eye_gap_since = None

        if yawn:
            gap_tolerance = self.cfg["yawn_gap_tolerance_seconds"]
            if (self.yawn_since is None or
                    (self.yawn_gap_since is not None and t - self.yawn_gap_since > gap_tolerance)):
                self.yawn_since = t
                self.yawn_counted = False
            self.yawn_gap_since = None
            required = self.cfg["yawn_min_duration_seconds"]
            if (not self.yawn_counted and t - self.yawn_since >= required
                    and t - self.last_yawn_event >= self.cfg["yawn_debounce_seconds"]):
                self.events.append(t)
                self.last_yawn_event = t
                self.yawn_counted = True
        elif self.yawn_since is not None:
            if self.yawn_gap_since is None:
                self.yawn_gap_since = t
            if t - self.yawn_gap_since > self.cfg["yawn_gap_tolerance_seconds"]:
                self.yawn_since = None
                self.yawn_gap_since = None
                self.yawn_counted = False

        possible_closed = (eye_closed is True or
                           (eye_closed is None and face_visible and self.unknown_eye_since is not None
                            and t - self.unknown_eye_since >= self.cfg["unknown_eye_warning_seconds"]))
        self.samples.append((t, eye_closed))
        self.risk_samples.append((t, possible_closed if possible_closed else (False if eye_closed is False else None)))
        self.last_t = t
        self._prune(t)

    def _prune(self, now: float) -> None:
        cutoff = now - max(self.cfg["window_seconds"], self.cfg["yawn_window_seconds"])
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        while self.risk_samples and self.risk_samples[0][0] < cutoff:
            self.risk_samples.popleft()
        while self.events and self.events[0] < now - self.cfg["yawn_window_seconds"]:
            self.events.popleft()

    def eye_status(self) -> str:
        if self.last_eye_state is None:
            return "UNKNOWN - eyes not visible"
        return "CLOSED" if self.last_eye_state else "OPEN"

    def perclos(self) -> float:
        """Return closed seconds divided by all available time in the window."""
        if len(self.risk_samples) < 2:
            return 0.0
        now = self.risk_samples[-1][0]
        start = max(self.risk_samples[0][0], now - self.cfg["window_seconds"])
        closed_time = 0.0
        for i, (sample_t, closed) in enumerate(self.risk_samples):
            end = self.risk_samples[i + 1][0] if i + 1 < len(self.risk_samples) else now
            span = max(0.0, min(end, now) - max(sample_t, start))
            if closed is True:
                closed_time += span
        available_time = max(0.0, now - start)
        return closed_time / available_time if available_time else 0.0

    def eye_coverage(self) -> float:
        """Fraction of the available rolling window with a known eye state."""
        if len(self.samples) < 2 or self.last_t is None or self.session_start is None:
            return 1.0 if self.last_eye_state is not None else 0.0
        now = self.samples[-1][0]
        start = max(self.samples[0][0], now - self.cfg["window_seconds"])
        observed = 0.0
        for i, (sample_t, state) in enumerate(self.samples):
            end = self.samples[i + 1][0] if i + 1 < len(self.samples) else now
            if state is not None:
                observed += max(0.0, min(end, now) - max(sample_t, start))
        elapsed = min(self.cfg["window_seconds"], max(0.0, now - self.session_start))
        if elapsed <= 0:
            return 1.0 if self.last_eye_state is not None else 0.0
        return min(1.0, observed / elapsed)

    def _reliable_perclos(self) -> float:
        """Return time-based PERCLOS without disabling it for low coverage."""
        return self.perclos()

    def yawn_events(self) -> int:
        if self.last_t is None:
            return 0
        cutoff = self.last_t - self.cfg["yawn_window_seconds"]
        return sum(event >= cutoff for event in self.events)

    def yawn_alert_active(self) -> bool:
        return (self.last_t is not None and self.last_yawn_event != float("-inf")
                and self.yawn_events() >= self.cfg["yawn_events_for_alarm"]
                and self.last_t - self.last_yawn_event <= self.cfg["yawn_drowsy_hold_seconds"])

    def recent_yawn(self) -> bool:
        return (self.last_t is not None and self.last_yawn_event != float("-inf")
                and self.yawn_events() > 0
                and self.last_t - self.last_yawn_event <= self.cfg["yawn_drowsy_hold_seconds"])

    def eye_alarm_active(self) -> bool:
        return (self.last_t is not None and self.eye_closed_since is not None
                and self.last_eye_state is True
                and self.last_t - self.eye_closed_since >= self.cfg["closed_eye_alarm_seconds"])

    def alert_level(self) -> str:
        """Return the single state that drives both the screen and the alarm."""
        self.state()
        return self.current_state

    def score(self) -> float:
        perclos_part = self._reliable_perclos()
        event_rate = min(1.0, self.yawn_events() / max(1, self.cfg["yawn_events_for_max"]))
        head_drop_part = float(self.head_drop_active)
        weights = (self.cfg["eye_weight"] + self.cfg["yawn_weight"]
                   + self.cfg["head_drop_weight"])
        if weights <= 0:
            raise ValueError("At least one drowsiness score weight must be positive")
        return (self.cfg["eye_weight"] * perclos_part + self.cfg["yawn_weight"] * event_rate
                + self.cfg["head_drop_weight"] * head_drop_part) / weights

    def reason(self) -> str:
        return self._reason

    def state(self) -> str:
        now = self.last_t or 0.0
        closed_seconds = now - self.eye_closed_since if self.eye_closed_since is not None else 0.0
        head_down_seconds = now - self.head_down_since if self.head_down_since is not None else 0.0
        if closed_seconds >= self.cfg["closed_eye_critical_seconds"]:
            self._reason, self.current_state = "Eyes closed for critical duration", "CRITICAL"
            return self.current_state
        if head_down_seconds >= self.cfg["head_down_critical_seconds"]:
            self._reason, self.current_state = "Head-down posture for critical duration", "CRITICAL"
            return self.current_state
        if self.face_missing_since is not None and now - self.face_missing_since >= self.cfg["face_missing_warning_seconds"]:
            self._reason, self.current_state = "Driver not found", "DRIVER_NOT_FOUND"
            return self.current_state
        if not self.camera_clear:
            self._reason, self.current_state = "Camera view is dark or unclear", "CAMERA_UNCLEAR"
            return self.current_state
        if closed_seconds >= self.cfg["closed_eye_warning_seconds"]:
            self._reason, self.current_state = "Eyes closed too long", "DROWSY"
            return self.current_state
        if head_down_seconds >= self.cfg["head_down_warning_seconds"]:
            self._reason, self.current_state = "Head-down posture", "WARNING"
            return self.current_state
        if self.head_turn_since is not None and now - self.head_turn_since >= self.cfg["head_turn_warning_seconds"]:
            self._reason, self.current_state = "Distracted - head turned", "WARNING"
            return self.current_state
        if self.unknown_eye_since is not None and now - self.unknown_eye_since >= self.cfg["unknown_eye_warning_seconds"]:
            self._reason, self.current_state = "Eyes unclear while face is visible", "WARNING"
            return self.current_state
        if self.yawn_alert_active():
            self._reason = "Repeated sustained yawns"
            self.current_state = "DROWSY"
            return self.current_state
        if self.head_drop_active:
            self._reason = "Sustained downward face position"
            self.current_state = "DROWSY"
            return self.current_state
        if self.recent_yawn():
            self._reason = "Sustained yawn detected"
            self.current_state = "WARNING"
            return self.current_state

        score = self.score()
        perclos = self._reliable_perclos()
        if perclos >= self.cfg["perclos_drowsy"]:
            self._reason, self.current_state = "PERCLOS above drowsy threshold", "DROWSY"
            return self.current_state
        if perclos >= self.cfg["perclos_warning"]:
            self._reason, self.current_state = "PERCLOS above warning threshold", "WARNING"
            return self.current_state
        margin = self.cfg["hysteresis_margin"]
        if self.current_state in {"DROWSY", "CRITICAL"}:
            if score >= self.cfg["score_drowsy"] - margin or perclos >= self.cfg["perclos_drowsy"] - margin:
                self.current_state = "DROWSY"
                self._reason = "Elevated recent eye-closure rate"
                return self.current_state
            self.current_state = ("WARNING" if score >= self.cfg["score_warning"] - margin
                                  or perclos >= self.cfg["perclos_warning"] - margin else "NORMAL")
        elif self.current_state == "WARNING":
            if score >= self.cfg["score_drowsy"] or perclos >= self.cfg["perclos_drowsy"]:
                self.current_state = "DROWSY"
            elif score < self.cfg["score_warning"] - margin and perclos < self.cfg["perclos_warning"] - margin:
                self.current_state = "NORMAL"
        elif score >= self.cfg["score_drowsy"] or perclos >= self.cfg["perclos_drowsy"]:
            self.current_state = "DROWSY"
        elif score >= self.cfg["score_warning"] or perclos >= self.cfg["perclos_warning"]:
            self.current_state = "WARNING"
        self._reason = ("Eyes not visible reliably" if self.eye_coverage() < self.cfg["minimum_eye_coverage"]
                        else "Elevated eye-closure rate" if perclos >= self.cfg["perclos_warning"]
                        else "Elevated drowsiness score" if score >= self.cfg["score_warning"]
                        else "No drowsiness signal")
        return self.current_state
