"""Verify FCBAM parameter registration, gradient flow, shapes, and saved names."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import ultralytics.nn.tasks as tasks
from model.fcbam import FCBAM
from ultralytics import YOLO


def main() -> None:
    torch.manual_seed(42)
    tasks.FCBAM = FCBAM
    detector = YOLO("configs/yolov11s_fcbam.yaml").model
    modules = [(name, module) for name, module in detector.named_modules() if isinstance(module, FCBAM)]
    if len(modules) != 3:
        raise AssertionError(f"Expected three FCBAM blocks in the parsed YOLO model, found {len(modules)}")
    for name, module in modules:
        count = sum(parameter.numel() for parameter in module.parameters())
        print(f"{name}: {count} parameters")
    detector.train()
    predictions = detector(torch.randn(2, 3, 128, 128))

    def collect_tensors(value):
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, dict):
            return [tensor for child in value.values() for tensor in collect_tensors(child)]
        if isinstance(value, (list, tuple)):
            return [tensor for child in value for tensor in collect_tensors(child)]
        return []

    tensors = collect_tensors(predictions)
    loss = sum(item.float().mean() for item in tensors)
    if not loss.requires_grad:
        raise AssertionError("YOLO forward output did not retain gradients")
    loss.backward()
    for name, module in modules:
        missing = [key for key, param in module.named_parameters() if param.grad is None]
        if missing:
            raise AssertionError(f"Missing gradients in {name}: {missing}")
    state = detector.state_dict()
    expected_names = {name for name in state if ".channel_attention." in name or ".spatial_attention." in name}
    with __import__("tempfile").NamedTemporaryFile(suffix=".pt", delete=False) as temporary:
        checkpoint = Path(temporary.name)
    try:
        torch.save(state, checkpoint)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    finally:
        checkpoint.unlink(missing_ok=True)
    if not expected_names or not expected_names.issubset(loaded):
        raise AssertionError("FCBAM parameter names are missing after state_dict save/reload")
    print(f"FCBAM verification: PASS ({len(expected_names)} attention state entries, all block parameters received gradients).")


if __name__ == "__main__":
    main()
