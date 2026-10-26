# Measured results

Fill each placeholder only from the named command output. No measurements have been run for this new project yet.

| System | mAP50 | mAP50-95 | Eyeclosed P/R | Eyeopen P/R | Yawn P/R | Parameters | How measured |
|---|---|---|---|---|---|---:|---|
| YOLOv11s baseline | [FILL FROM MEASUREMENT] | [FILL FROM MEASUREMENT] | [FILL FROM MEASUREMENT] | [FILL FROM MEASUREMENT] | [FILL FROM MEASUREMENT] | [FILL FROM MEASUREMENT] | `python scripts/evaluate.py --weights ... --data ... --split test`; parameters from the trained model summary. |
| YOLOv11s + FCBAM | [FILL FROM MEASUREMENT] | [FILL FROM MEASUREMENT] | [FILL FROM MEASUREMENT] | [FILL FROM MEASUREMENT] | [FILL FROM MEASUREMENT] | [FILL FROM MEASUREMENT] | Same held-out test set and evaluation script; keep image size and validation settings equal to baseline. |

| Verification/demo measure | Result | How measured |
|---|---|---|
| FCBAM verification | [FILL FROM MEASUREMENT: PASS/FAIL] | `python scripts/verify_fcbam.py`; verify on the complete Ultralytics model before calling the branch verified. |
| Live average FPS | [FILL FROM MEASUREMENT] | Webcam session CSV, measured on the named machine and selected checkpoint. |
| Mean / p95 detector latency (ms) | [FILL FROM MEASUREMENT] | Timed inference on labeled demo clips; report resolution, device, and warm-up policy. |
| Drowsiness event recall | [FILL FROM MEASUREMENT] | `python scripts/evaluate_demo.py` against manually labeled event intervals. |
| False alarms/hour on normal video | [FILL FROM MEASUREMENT] | `scripts/evaluate_demo.py` counts predicted DROWSY intervals on clips marked `normal: true`, divided by clip duration in hours; verify the ground-truth protocol before quoting. |

## Limitations

- Dataset label quality has not been independently reviewed; the preparation report only confirms format/range checks.
- Subject/driver-level split isolation is unverified.
- A frame-based object detector is not a medical diagnosis or a validated safety device.
- FCBAM is experimental until gradient and full-model checkpoint checks pass in the target Ultralytics/PyTorch environment.
- No test metrics, real-time performance results, or drowsiness-event results are available yet.
