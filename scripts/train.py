"""Reproducible YOLO baseline or FCBAM training entry point."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ultralytics.nn.tasks as tasks
from model.fcbam import FCBAM
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11s.pt", help="Baseline checkpoint/YAML or custom FCBAM YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pretrained", nargs="?", const="yolo11s.pt", help="Optional source checkpoint for transfer, e.g. yolo11s.pt")
    parser.add_argument("--resume", help="Resume an interrupted run from its weights/last.pt checkpoint")
    args = parser.parse_args()
    tasks.FCBAM = FCBAM
    if args.resume:
        model = YOLO(args.resume)
        # Resume restores optimizer, epoch, and run settings stored in last.pt.
        model.train(resume=True, device=args.device)
    else:
        model = YOLO(args.model)
        if args.pretrained:
            model.load(args.pretrained)
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    seed=args.seed, project=args.project, name=args.name, device=args.device,
                    workers=args.workers, pretrained=False)
    output = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best checkpoint: {output}")


if __name__ == "__main__":
    main()
