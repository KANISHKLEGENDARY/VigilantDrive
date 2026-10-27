"""Per-session live telemetry and CSV output."""
import csv
from datetime import datetime
from pathlib import Path


class Metrics:
    def __init__(self, output_dir: str):
        self.path = Path(output_dir) / f"session_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["monotonic_seconds", "score", "perclos", "eye_coverage", "yawn_events", "eye_status",
                              "ear", "head_pitch_degrees", "head_yaw_degrees", "head_down", "state", "reason",
                              "alarm_level", "fps", "frame_ms", "detector_latency_ms"])
        self.frame_times: list[float] = []

    def record(self, t: float, score: float, perclos: float, yawns: int, state: str,
               latency_ms: float, eye_status: str = "UNKNOWN", reason: str = "", alarm_active: bool = False,
               eye_coverage: float = 0.0, head_drop_active: bool = False,
               ear: float | None = None, head_pitch: float | None = None,
               head_yaw: float | None = None, alarm_level: str = "NORMAL",
               frame_ms: float = 0.0) -> float:
        self.frame_times.append(t)
        while self.frame_times and self.frame_times[0] < t - 5:
            self.frame_times.pop(0)
        fps = ((len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])
               if len(self.frame_times) > 1 and self.frame_times[-1] > self.frame_times[0] else 0.0)
        self.writer.writerow([f"{t:.3f}", f"{score:.4f}", f"{perclos:.4f}", f"{eye_coverage:.4f}", yawns,
                              eye_status, "" if ear is None else f"{ear:.4f}",
                              "" if head_pitch is None else f"{head_pitch:.2f}",
                              "" if head_yaw is None else f"{head_yaw:.2f}", int(head_drop_active),
                              state, reason, alarm_level, f"{fps:.2f}", f"{frame_ms:.2f}", f"{latency_ms:.2f}"])
        self.file.flush()
        return fps

    def close(self) -> None:
        self.file.close()
