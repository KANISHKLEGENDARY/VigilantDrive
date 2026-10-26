"""Evaluate an explicit checkpoint on a selected dataset split."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ultralytics.nn.tasks as tasks
from model.fcbam import FCBAM
from ultralytics import YOLO
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--config", default="config.yaml", help="Runtime config supplies the webcam image size")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tasks.FCBAM = FCBAM
    model = YOLO(args.weights)
    result = model.val(data=args.data, split=args.split, device=args.device,
                       imgsz=int(cfg["imgsz"]), plots=False)
    metrics = result.box
    per_class = {}
    for index, class_id in enumerate(metrics.ap_class_index.tolist()):
        name = result.names[int(class_id)]
        per_class[name] = {"precision": float(metrics.p[index]), "recall": float(metrics.r[index]),
                           "mAP50": float(metrics.ap50[index]), "mAP50_95": float(metrics.ap[index])}
    payload = {"weights": str(Path(args.weights).resolve()), "data": str(Path(args.data).resolve()),
               "split": args.split, "precision": float(metrics.mp), "recall": float(metrics.mr),
               "mAP50": float(metrics.map50), "mAP50_95": float(metrics.map), "per_class": per_class}
    output = Path(args.output_dir) / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
