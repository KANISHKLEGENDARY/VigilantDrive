import csv
from src.metrics import Metrics


def test_metrics_csv_includes_per_stage_timings(tmp_path):
    metrics = Metrics(str(tmp_path))
    metrics.record(1.0, 0.0, 0.0, 0, "NORMAL", 0.0,
                   stage_times={"capture_ms": 2.5, "face_mesh_ms": 4.0,
                                "yolo_stage_ms": 0.0, "jpeg_encode_ms": 1.0,
                                "drawing_ms": 2.0, "display_ms": 0.5})
    metrics.close()
    with metrics.path.open(newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))
    assert float(rows[0]["face_mesh_ms"]) == 4.0
    assert float(rows[0]["display_ms"]) == 0.5
