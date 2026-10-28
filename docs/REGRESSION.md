# Regression fixes: head-down and closed-eye misses

- `angle_delta(current, baseline)` is the shortest signed raw angle difference; it does not choose what “down” means.
- The last B1–B13 round did not change pitch direction; `src/config.py` still sets `HEAD_DOWN_DIRECTION = -1.0` from the earlier pose integration.
- `src/main.py` multiplies pitch delta by that value and passes the resulting signed angle to the monitor; the monitor treats a positive value as head-down.
- The prior eye-fusion change made clearly closed EAR decisive, but immediately treated open EAR as open, so a borderline closed EAR could discard YOLO evidence.
- `EAR_CLOSED_BASELINE_FRACTION` was `0.60`; the displayed cutoff is the calibrated open-eye baseline multiplied by that fraction.

## Fixes in this round

1. **Head-down sign:** `HEAD_DOWN_DIRECTION` is now `+1.0` for the reported camera direction. `head_pitch_delta` names the normalization step, the monitor receives positive values for chin-down, the overlay says `Head pitch (+ = down)`, and head-back stays negative. No separate head-back alert was added.
2. **Closed-eye fusion:** the closed EAR fraction is now `0.70` (the prompt's suggested starting value), and the strong-open boundary is `0.85`. A borderline EAR can be overridden only by the detector's strong paired-closed result; that detector requires both eye boxes at confidence `>= 0.40`. Clearly open EAR remains open; clearly closed EAR conflicting with a both-open pair becomes unknown.
3. **History-only alert recovery:** after eyes-open, face-visible, camera-clear and head-up signals have remained safe for 2 seconds, a PERCLOS-only alert drops to NORMAL with a recovery message. The alarm therefore becomes silent; later history-only PERCLOS does not restart it until new unsafe evidence arrives. Direct eye-closure, head-pose, yawn, face and camera rules still run before this recovery rule.

## Replay acceptance

The requested `Screen_Recording_2026-10-28_121131.mp4` was not found in the repository or provided as an attachment in this task. Therefore this recording could not be replayed, no replay CSV exists, and no observed times, beep gaps, or demo FPS are claimed. The synthetic regression tests exercise the corresponding rule paths but are not substitutes for the recording.

| Window (s) | What the recording should show | Replay result |
|---|---|---|
| 2–23 | Head down: WARNING by 2.5 s, CRITICAL by about 4.5 s, remains until head comes up | Not replayed; MP4 unavailable. Synthetic test verifies normalized positive pitch reaches CRITICAL while negative head-back pitch stays NORMAL. |
| 26–35 | Eyes closed with head tilted back | Not replayed; regression not checked on the recording. |
| 36–44 | Eyes open: silent within 3 s and NORMAL by 5 s | Not replayed; synthetic state test verifies history-only PERCLOS clears after 2 s of safe recovery. No physical alarm timing measured. |
| 45–57 | Eyes closed with head slightly forward: alarm by 2.5 s, CRITICAL by 4.5 s | Not replayed; synthetic eye-fusion test verifies strong paired YOLO can close a borderline EAR result. |
| 1:08–1:20 | Yawning | Not replayed; no yawn regression result available. |
| Longest silent gap while drowsy | At most 1 s | Not measured without the recording and speaker output. |
| YOLO checkpoint SHA-256 | Unchanged | Verified: `6D64CDE43C226A1163BB3268CBFB99C4EB0FED50B8F69DA4E066AFBE2CED5597`. |

The `0.70` EAR fraction is an initial value, not a proven maximum that avoids false closures in all three requested alert stretches. Selecting the largest safe fraction requires replaying this specific video and inspecting the EAR/YOLO/fused signal values. The test suite cannot establish that calibration value for a person or camera.

## Synthetic checks added

| Test file | Coverage | Result after fixes |
|---|---|---|
| `tests/test_head_pitch_sign.py` | Positive-down normalization; head-down reaches CRITICAL and head-back does not | Pass |
| `tests/test_eye_override.py` | Strong paired YOLO override bounds; conflicts remain unknown | Pass |
| `tests/test_perclos_recovery.py` | PERCLOS-only alert clears after safe recovery and shows silent recovery message | Pass |
