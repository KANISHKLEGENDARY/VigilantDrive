"""Time-based drowsiness signals independent of the camera and detector."""
from __future__ import annotations

from collections import deque
from statistics import median


class DrowsinessMonitor:
    """Calculate PERCLOS, sustained yawn events, score and alert state.

    eye_closed accepts True (closed), False (open), or None (not visible).
    Unknown observations are excluded from PERCLOS instead of being counted as
    open. Short gaps can bridge an ongoing closure but cannot start one.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.samples: deque[tuple[float, bool | None]] = deque()
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
        self.head_baseline_values: list[float] = []
        self.head_baseline: float | None = None
        self.head_drop_since: float | None = None
        self.head_drop_active = False

    def reset(self) -> None:
        """Clear temporal state after a camera interruption or new clip."""
        self.samples.clear()
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
        self.head_baseline_values.clear()
        self.head_baseline = None
        self.head_drop_since = None
        self.head_drop_active = False

    def update(self, t: float, eye_closed: bool | None, yawn: bool,
               face_center_y: float | None = None) -> None:
        """Add an observation at monotonic (or source-video) time ``t``."""
        if self.last_t is not None and t < self.last_t:
            raise ValueError("timestamps must be non-decreasing")
        if self.session_start is None:
            self.session_start = t
        baseline_seconds = self.cfg.get("head_drop_baseline_seconds", 10.0)
        if face_center_y is not None:
            if self.head_baseline is None and t - self.session_start <= baseline_seconds:
                self.head_baseline_values.append(float(face_center_y))
            elif self.head_baseline is None and self.head_baseline_values:
                self.head_baseline = median(self.head_baseline_values)
            if self.head_baseline is not None and face_center_y - self.head_baseline >= self.cfg.get("head_drop_offset_fraction", 0.12):
                if self.head_drop_since is None:
                    self.head_drop_since = t
                self.head_drop_active = t - self.head_drop_since >= self.cfg.get("head_drop_duration_seconds", 2.0)
            else:
                self.head_drop_since = None
                self.head_drop_active = False
        else:
            self.head_drop_since = None
            self.head_drop_active = False
        eye_closed = None if eye_closed is None else bool(eye_closed)
        self.last_eye_state = eye_closed
        gap = self.cfg.get("closed_eye_gap_tolerance_seconds", 0.35)
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
            gap_tolerance = self.cfg.get("yawn_gap_tolerance_seconds", 0.35)
            if (self.yawn_since is None or
                    (self.yawn_gap_since is not None and t - self.yawn_gap_since > gap_tolerance)):
                self.yawn_since = t
                self.yawn_counted = False
            self.yawn_gap_since = None
            required = self.cfg.get("yawn_min_duration_seconds", 2.0)
            if (not self.yawn_counted and t - self.yawn_since >= required
                    and t - self.last_yawn_event >= self.cfg["yawn_debounce_seconds"]):
                self.events.append(t)
                self.last_yawn_event = t
                self.yawn_counted = True
        elif self.yawn_since is not None:
            if self.yawn_gap_since is None:
                self.yawn_gap_since = t
            if t - self.yawn_gap_since > self.cfg.get("yawn_gap_tolerance_seconds", 0.35):
                self.yawn_since = None
                self.yawn_gap_since = None
                self.yawn_counted = False

        self.samples.append((t, eye_closed))
        self.last_t = t
        self._prune(t)

    def _prune(self, now: float) -> None:
        cutoff = now - max(self.cfg["window_seconds"], self.cfg["yawn_window_seconds"])
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        while self.events and self.events[0] < now - self.cfg["yawn_window_seconds"]:
            self.events.popleft()

    def eye_status(self) -> str:
        if self.last_eye_state is None:
            return "UNKNOWN — eyes not visible"
        return "CLOSED" if self.last_eye_state else "OPEN"

    def perclos(self) -> float:
        """Closed-time / observed eye-time; unknown time is excluded."""
        if len(self.samples) < 2:
            return 0.0
        now = self.samples[-1][0]
        start = max(self.samples[0][0], now - self.cfg["window_seconds"])
        closed_time = observed_time = 0.0
        for i, (sample_t, closed) in enumerate(self.samples):
            end = self.samples[i + 1][0] if i + 1 < len(self.samples) else now
            span = max(0.0, min(end, now) - max(sample_t, start))
            if closed is not None:
                observed_time += span
                if closed:
                    closed_time += span
        return closed_time / observed_time if observed_time else 0.0

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
        if self.eye_coverage() < self.cfg.get("minimum_eye_coverage", 0.5):
            return 0.0
        return self.perclos()

    def yawn_events(self) -> int:
        if self.last_t is None:
            return 0
        cutoff = self.last_t - self.cfg["yawn_window_seconds"]
        return sum(event >= cutoff for event in self.events)

    def yawn_alert_active(self) -> bool:
        return (self.last_t is not None and self.last_yawn_event != float("-inf")
                and self.yawn_events() >= self.cfg.get("yawn_events_for_alarm", 2)
                and self.last_t - self.last_yawn_event <= self.cfg.get("yawn_drowsy_hold_seconds", 5.0))

    def recent_yawn(self) -> bool:
        return (self.last_t is not None and self.last_yawn_event != float("-inf")
                and self.yawn_events() > 0
                and self.last_t - self.last_yawn_event <= self.cfg.get("yawn_drowsy_hold_seconds", 5.0))

    def eye_alarm_active(self) -> bool:
        return (self.last_t is not None and self.eye_closed_since is not None
                and self.last_eye_state is True
                and self.last_t - self.eye_closed_since >= self.cfg.get("closed_eye_alarm_seconds", 1.5))

    def score(self) -> float:
        perclos_part = self._reliable_perclos()
        event_rate = min(1.0, self.yawn_events() / max(1, self.cfg.get("yawn_events_for_max", 3)))
        head_drop_part = float(self.head_drop_active)
        weights = (self.cfg["eye_weight"] + self.cfg["yawn_weight"]
                   + self.cfg.get("head_drop_weight", 0.0))
        if weights <= 0:
            raise ValueError("At least one drowsiness score weight must be positive")
        return (self.cfg["eye_weight"] * perclos_part + self.cfg["yawn_weight"] * event_rate
                + self.cfg.get("head_drop_weight", 0.0) * head_drop_part) / weights

    def reason(self) -> str:
        return self._reason

    def state(self) -> str:
        if self.eye_alarm_active():
            self._reason = "Eyes closed too long"
            self.current_state = "DROWSY"
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
        margin = self.cfg["hysteresis_margin"]
        if self.current_state == "DROWSY":
            if score >= self.cfg["score_drowsy"] - margin or perclos >= self.cfg["perclos_drowsy"] - margin:
                self._reason = "Elevated recent eye-closure rate"
                return self.current_state
            self.current_state = "WARNING" if score >= self.cfg["score_warning"] - margin or perclos >= self.cfg["perclos_warning"] - margin else "NORMAL"
        elif self.current_state == "WARNING":
            if score >= self.cfg["score_drowsy"] or perclos >= self.cfg["perclos_drowsy"]:
                self.current_state = "DROWSY"
            elif score < self.cfg["score_warning"] - margin and perclos < self.cfg["perclos_warning"] - margin:
                self.current_state = "NORMAL"
        elif score >= self.cfg["score_drowsy"] or perclos >= self.cfg["perclos_drowsy"]:
            self.current_state = "DROWSY"
        elif score >= self.cfg["score_warning"] or perclos >= self.cfg["perclos_warning"]:
            self.current_state = "WARNING"
        self._reason = ("Eyes not visible reliably" if self.eye_coverage() < self.cfg.get("minimum_eye_coverage", 0.5)
                        else "Elevated eye-closure rate" if perclos >= self.cfg["perclos_warning"]
                        else "Elevated drowsiness score" if score >= self.cfg["score_warning"]
                        else "No drowsiness signal")
        return self.current_state
