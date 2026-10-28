"""Time-based drowsiness signals independent of the camera and detector."""
from __future__ import annotations

from collections import deque
from src.config import DEFAULT_RUNTIME_CONFIG
from src.config import (CAMERA_CLEAR_SECONDS, FACE_RETURN_CLEAR_SECONDS,
                        PERCLOS_MIN_HISTORY_SECONDS, RECOVERY_CLEAR_SECONDS,
                        PERCLOS_SAFE_CLEAR_SECONDS, TIMER_GRACE_SECONDS)


class GraceTimer:
    """Keep a condition active through short false readings."""

    def __init__(self, grace_seconds: float = TIMER_GRACE_SECONDS):
        self.grace_seconds = grace_seconds
        self.active_since: float | None = None
        self.false_since: float | None = None

    def update(self, condition: bool | None, now: float) -> float | None:
        """Return the start time until the condition stays false beyond grace."""
        if condition is None:
            return self.active_since
        if condition:
            self.false_since = None
            if self.active_since is None:
                self.active_since = now
        elif self.active_since is not None:
            if self.false_since is None:
                self.false_since = now
            elif now - self.false_since > self.grace_seconds:
                self.active_since = None
                self.false_since = None
        return self.active_since


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
        grace = float(self.cfg.get("timer_grace_seconds", TIMER_GRACE_SECONDS))
        self.head_down_timer = GraceTimer(grace)
        self.head_turn_timer = GraceTimer(grace)
        self.face_missing_timer = GraceTimer(grace)
        self.unknown_eye_timer = GraceTimer(grace)
        self.ok_timer = GraceTimer(grace)
        self.face_visible_since: float | None = None
        self.camera_clear_since: float | None = None
        self.perclos_history_start: float | None = None
        self._recovery_cleared = False
        self.perclos_only_alert = False
        self._perclos_recovery_suppressed = False

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
        grace = float(self.cfg.get("timer_grace_seconds", TIMER_GRACE_SECONDS))
        self.head_down_timer = GraceTimer(grace)
        self.head_turn_timer = GraceTimer(grace)
        self.face_missing_timer = GraceTimer(grace)
        self.unknown_eye_timer = GraceTimer(grace)
        self.ok_timer = GraceTimer(grace)
        self.face_visible_since = None
        self.camera_clear_since = None
        self.perclos_history_start = None
        self._recovery_cleared = False
        self.perclos_only_alert = False
        self._perclos_recovery_suppressed = False

    def update(self, t: float, eye_closed: bool | None, yawn: bool, *, face_visible: bool = True,
               head_pitch: float | None = None, head_yaw: float | None = None,
               camera_clear: bool = True) -> None:
        """Add an observation at monotonic (or source-video) time ``t``."""
        if self.last_t is not None and t < self.last_t:
            raise ValueError("timestamps must be non-decreasing")
        if self.session_start is None:
            self.session_start = t
            self.perclos_history_start = t
        self.face_visible = face_visible
        self.camera_clear = camera_clear
        if face_visible:
            self.face_visible_since = self.face_visible_since if self.face_visible_since is not None else t
        else:
            self.face_visible_since = None
        if camera_clear:
            self.camera_clear_since = self.camera_clear_since if self.camera_clear_since is not None else t
        else:
            self.camera_clear_since = None
        self.face_missing_since = self.face_missing_timer.update(not face_visible, t)
        unknown_condition = None if not face_visible else eye_closed is None
        self.unknown_eye_since = self.unknown_eye_timer.update(unknown_condition, t)
        self.head_pitch = float(head_pitch or 0.0)
        self.head_yaw = float(head_yaw or 0.0)
        head_down_condition = (None if not face_visible or head_pitch is None
                               else self.head_pitch >= self.cfg["head_down_angle_degrees"])
        self.head_down_since = self.head_down_timer.update(head_down_condition, t)
        self.head_drop_since = self.head_down_since
        self.head_drop_active = (self.head_down_since is not None and
                                 t - self.head_down_since >= self.cfg["head_down_warning_seconds"])
        head_turn_condition = (None if not face_visible or head_yaw is None
                               else abs(self.head_yaw) >= self.cfg["head_turn_angle_degrees"])
        self.head_turn_since = self.head_turn_timer.update(head_turn_condition, t)
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
        recovered = (eye_closed is False and face_visible and camera_clear
                     and (head_pitch is None or self.head_pitch < self.cfg["head_down_angle_degrees"])
                     and (head_yaw is None or abs(self.head_yaw) < self.cfg["head_turn_angle_degrees"]))
        if not recovered:
            # New unsafe evidence permits a later PERCLOS alert to start again.
            self._perclos_recovery_suppressed = False
        recovered_since = self.ok_timer.update(recovered, t)
        if recovered_since is None:
            self._recovery_cleared = False
        elif (not self._recovery_cleared
              and t - recovered_since >= self.cfg.get("recovery_clear_seconds", RECOVERY_CLEAR_SECONDS)):
            self.samples.clear()
            self.risk_samples.clear()
            self.events.clear()
            self.perclos_history_start = t
            self.session_start = t
            self.samples.append((t, eye_closed))
            self.risk_samples.append((t, possible_closed if possible_closed else
                                      (False if eye_closed is False else None)))
            self.yawn_since = self.yawn_gap_since = None
            self.yawn_counted = False
            self.last_yawn_event = float("-inf")
            self.current_state = "NORMAL"
            self._reason = "No drowsiness signal"
            self.perclos_only_alert = False
            self._perclos_recovery_suppressed = False
            self._recovery_cleared = True

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
        if self.last_t is None or self.perclos_history_start is None:
            return 0.0
        if self.last_t - self.perclos_history_start < self.cfg.get(
                "perclos_min_history_seconds", PERCLOS_MIN_HISTORY_SECONDS):
            return 0.0
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

    @property
    def perclos_recovery_suppressed(self) -> bool:
        """Tell the alarm that a history-only alert has completed recovery."""
        return self._perclos_recovery_suppressed

    def _set_result(self, state: str, reason: str, *, perclos_only: bool = False) -> str:
        """Set state and reason together so the overlay cannot contradict itself."""
        assert reason != "No drowsiness signal" or state == "NORMAL"
        self.current_state, self._reason = state, reason
        self.perclos_only_alert = perclos_only and state != "NORMAL"
        return state

    def state(self) -> str:
        now = self.last_t or 0.0
        closed_seconds = now - self.eye_closed_since if self.eye_closed_since is not None else 0.0
        head_down_seconds = now - self.head_down_since if self.head_down_since is not None else 0.0
        if closed_seconds >= self.cfg["closed_eye_critical_seconds"]:
            return self._set_result("CRITICAL", "Eyes closed for critical duration")
        if head_down_seconds >= self.cfg["head_down_critical_seconds"]:
            return self._set_result("CRITICAL", "Head-down posture for critical duration")
        face_returned = (self.face_visible_since is not None and
                         now - self.face_visible_since >= self.cfg.get(
                             "face_return_clear_seconds", FACE_RETURN_CLEAR_SECONDS))
        if self.current_state == "DRIVER_NOT_FOUND" and not face_returned:
            return self._set_result("DRIVER_NOT_FOUND", "Driver not found")
        if (self.face_missing_since is not None
                and now - self.face_missing_since >= self.cfg["face_missing_warning_seconds"]
                and not face_returned):
            return self._set_result("DRIVER_NOT_FOUND", "Driver not found")
        if not self.camera_clear:
            return self._set_result("CAMERA_UNCLEAR", "Camera view is dark or unclear")
        camera_recovered = (self.camera_clear_since is not None and
                            now - self.camera_clear_since >= self.cfg.get(
                                "camera_clear_seconds", CAMERA_CLEAR_SECONDS))
        if self.current_state == "CAMERA_UNCLEAR" and not camera_recovered:
            return self._set_result("CAMERA_UNCLEAR", "Camera view is recovering")
        if closed_seconds >= self.cfg["closed_eye_warning_seconds"]:
            return self._set_result("DROWSY", "Eyes closed too long")
        if head_down_seconds >= self.cfg["head_down_warning_seconds"]:
            return self._set_result("WARNING", "Head-down posture")
        if self.head_turn_since is not None and now - self.head_turn_since >= self.cfg["head_turn_warning_seconds"]:
            return self._set_result("WARNING", "Distracted - head turned")
        if self.unknown_eye_since is not None and now - self.unknown_eye_since >= self.cfg["unknown_eye_warning_seconds"]:
            return self._set_result("WARNING", "Eyes unclear while face is visible")
        if self.yawn_alert_active():
            return self._set_result("DROWSY", "Repeated sustained yawns")
        if self.head_drop_active:
            return self._set_result("DROWSY", "Sustained downward face position")
        if self.recent_yawn():
            return self._set_result("WARNING", "Sustained yawn detected")

        safe_since = self.ok_timer.active_since
        safe_for = now - safe_since if safe_since is not None else 0.0
        if self._perclos_recovery_suppressed:
            return self._set_result("NORMAL", "Recent PERCLOS alert cleared after recovery")
        if (self.perclos_only_alert and safe_for >= self.cfg.get(
                "perclos_safe_clear_seconds", PERCLOS_SAFE_CLEAR_SECONDS)):
            self._perclos_recovery_suppressed = True
            return self._set_result("NORMAL", "Recent PERCLOS alert cleared after recovery")

        score = self.score()
        perclos = self._reliable_perclos()
        history_only = self.yawn_events() == 0 and not self.head_drop_active
        previous = self.current_state
        warning_enter = self.cfg.get("perclos_warning_enter", 0.25)
        warning_exit = self.cfg.get("perclos_warning_exit", 0.15)
        drowsy_enter = self.cfg.get("perclos_drowsy_enter", 0.35)
        drowsy_exit = self.cfg.get("perclos_drowsy_exit", 0.25)
        if previous in {"DROWSY", "CRITICAL"}:
            if perclos >= drowsy_exit or score >= self.cfg["score_drowsy"] - self.cfg["hysteresis_margin"]:
                return self._set_result("DROWSY", "Elevated recent eye-closure rate",
                                        perclos_only=history_only)
            if perclos >= warning_exit or score >= self.cfg["score_warning"] - self.cfg["hysteresis_margin"]:
                return self._set_result("WARNING", "Elevated recent drowsiness signal",
                                        perclos_only=history_only)
        elif previous == "WARNING":
            if perclos >= drowsy_enter or score >= self.cfg["score_drowsy"]:
                return self._set_result("DROWSY", "Drowsiness signal above threshold",
                                        perclos_only=history_only)
            if perclos >= warning_exit or score >= self.cfg["score_warning"] - self.cfg["hysteresis_margin"]:
                return self._set_result("WARNING", "Elevated recent drowsiness signal",
                                        perclos_only=history_only)
        else:
            if perclos >= drowsy_enter or score >= self.cfg["score_drowsy"]:
                return self._set_result("DROWSY", "PERCLOS or score above drowsy threshold",
                                        perclos_only=history_only)
            if perclos >= warning_enter or score >= self.cfg["score_warning"]:
                return self._set_result("WARNING", "PERCLOS or score above warning threshold",
                                        perclos_only=history_only)
        return self._set_result("NORMAL", "No drowsiness signal")
