# Alarm and drowsiness fixes, explained

This document describes the current rules in plain language. The YOLO weights were not retrained or changed. These changes adjust how the application combines detector and face-landmark signals, waits for evidence, selects a screen state, and plays sound.

## What the states mean

| State | How it starts | How it clears | Sound |
|---|---|---|---|
| `NORMAL` | No current rule calls for an alert. PERCLOS cannot trigger until there are 10 seconds of history. | Remains until an alert rule is met. Five seconds of recovered signals clear old eye/yawn history. | Silent. |
| `WARNING` | Examples: head down for 2 seconds, head turned for 3 seconds, unknown eyes for 1 second, a sustained yawn, or warning-level PERCLOS/score. | The relevant signal must recover; alarm release also requires the configured safe interval. | At most two beeps during one warning episode. |
| `DROWSY` | Eyes closed for 2 seconds, repeated sustained yawns, sustained downward posture, or drowsy-level PERCLOS/score. | Safe signals must persist; a brief noisy reading is tolerated. | Repeating pulse. |
| `CRITICAL` | Eyes closed for 4 seconds or head down for 4 seconds. | Safe signals must persist; a brief noisy reading is tolerated. | Faster, louder repeating pulse. |
| `DRIVER_NOT_FOUND` | Face is missing for 3 seconds. | Face must be visible for 0.5 seconds. | Repeating visibility warning. |
| `CAMERA_UNCLEAR` | Camera image quality or feed-loss rule fails. | Image/feed must be clear for 1 second. | Repeating camera warning. |

The display state and alarm state come from the same monitor state. This is a laptop prototype, not a certified driver-safety system.

## Important thresholds

| Setting | Current default | Meaning |
|---|---:|---|
| Closed-eye warning / critical | 2 / 4 seconds | Continuous closed-eye evidence needed for DROWSY / CRITICAL. |
| Head-down warning / critical | 25 degrees for 2 / 4 seconds | Downward pitch relative to the startup calibration. |
| Head-turn warning | 35 degrees for 3 seconds | Absolute yaw relative to calibration. |
| Eye debounce | 3 frames and 0.15 seconds, for both closing and opening | Filters short flicker and avoids keeping normal blinks closed too long. |
| Timer grace | 0.5 seconds | Short false/noisy reads do not immediately erase ongoing pose, missing-face, or unknown-eye evidence. |
| Alarm-safe grace | 0.3 seconds | A single short unsafe read does not restart safe-to-clear timing. |
| Alarm clear | 2 seconds safe; alert minimum 3 seconds | Keeps an alert briefly visible and avoids stopping on one apparently safe frame. |
| Recovered-history clear | 5 seconds | Clears old PERCLOS and yawn evidence after eyes-open, face-visible, camera-clear, head-up recovery. |
| PERCLOS minimum history | 10 seconds | Prevents a short early-session sample from dominating the percentage. Direct duration rules still work immediately. |
| PERCLOS warning enter / exit | 0.25 / 0.15 | Separate values reduce rapid state switching around one boundary. |
| PERCLOS drowsy enter / exit | 0.35 / 0.25 | Separate values reduce rapid state switching around one boundary. |
| Calibration | At least 40 valid samples, about 5 seconds | Starts once a valid face signal arrives; EAR outside 0.18–0.45 or pitch spread over 10 degrees is rejected. Press **C** to retry. |
| EAR yaw limit | 25 degrees | EAR alone is ignored during larger head turns because turning can distort the eye measurement. |
| Closed-eye pair confidence | 0.40 per eye | Both closed-eye boxes must meet this confidence before the detector calls a pair closed. |
| Clip sampling | Every 3rd frame, maximum width 640 pixels | Reduces JPEG work in the camera loop; output timing is adjusted to keep clip duration. |
| WARNING beeps | Maximum 2 per episode | Warning remains visible after the second beep; higher states continue their repeating alarm. |

All defaults live in `src/config.py`; operational choices such as model path, camera, and output directories live in `config.yaml`.

## Bug-by-bug explanation

| Bug | What was wrong | Why it happened | What the current code does |
|---|---|---|---|
| B1 | The display could say “No drowsiness signal” while remaining WARNING. | State and reason were chosen by separate threshold paths, and a shared score margin was inappropriate for PERCLOS. | `_set_result` sets state and reason together. PERCLOS has explicit warning/drowsy enter and exit levels. |
| B2 | `DRIVER_NOT_FOUND` or `CAMERA_UNCLEAR` could stick after recovery. | The state logic lacked explicit recovery paths for those states. | Driver visibility must return for 0.5 seconds; camera clarity must return for 1 second. |
| B3 | One noisy unsafe frame could keep the alarm from clearing. | The safe timer restarted on every unsafe frame; main also used a raw eye candidate. | Alarm recovery has a 0.3-second unsafe grace, and `main.py` uses the debounced eye result. |
| B4 | A brief event could affect PERCLOS/yawn state after the driver recovered. | Old rolling samples stayed in the history. | Five seconds of continuous safe recovery clears old samples, risk samples, and yawn events. |
| B5 | WARNING could beep every 3 seconds indefinitely. | WARNING had no beep limit. | It beeps at most twice per episode; NORMAL immediately stops sound and drains old queued sound. DROWSY/CRITICAL keep repeating. |
| B6 | A loose OR between YOLO and EAR could label a normal eye closed. | Either source could independently win, and the EAR cutoff could be close to relaxed open-eye values. | EAR leads: open EAR means open; moderate closure needs YOLO support; very low EAR can mean closed by itself; conflicting evidence becomes unknown. EAR is ignored beyond the yaw limit. The calibration cutoff is shown on screen. |
| B7 | Configured alarm pitch could differ from the generated sound pitch. | Sound buffers assumed a sample rate/channel format rather than using the active mixer format. | The app reads the mixer’s actual sample rate and channel count and generates buffers in that format. Physical recording verification was not available. |
| B8 | Camera processing could be slowed by repeated full-size JPEG encoding, and the bottleneck was unclear. | Several stages shared the frame loop without per-stage timing. | The app records capture, face-mesh, YOLO, JPEG, draw, display, and total frame timing. Review clips encode every third, resized frame. MediaPipe does not calculate unused iris landmarks. No real webcam FPS claim is made. |
| B9 | Normal blinks could be counted closed too long. | Reopening required more frames/time than closing. | Open and closed debounce are symmetric: 3 frames and 0.15 seconds. Synthetic tests cover repeated blinks; the webcam behavior still needs observation. |
| B10 | PERCLOS could become extreme from only a few seconds of startup data. | It divided by the short amount of data currently available. | PERCLOS-based state changes wait for 10 seconds of history; direct closure and pose rules remain active. |
| B11 | A brief landmark/pose miss could reset timers and hide sustained head-down evidence. | A single false frame cleared condition timers; missing-face frames supplied a fake neutral pose. | `GraceTimer` tolerates short gaps. Missing pose is passed as unknown rather than neutral. Failed pose estimation reuses the last good pose while the face remains visible. |
| B12 | Calibration could finish with too few/unstable samples or before the driver appeared. | Its clock began at application launch and acceptance checks were weak. | It starts with the first valid face signal, requires 40 samples, rejects implausible EAR or unstable pitch, and supports **C** to reset. Alerts are silenced while calibration runs. |
| B13 | Conflicting or weak eye boxes could be treated as safe/closed incorrectly; boxes near face edges could be dropped. | Pair logic accepted disagreement as open and did not require adequate closed confidence; containment checked all box edges. | Conflicting or weak pairs are unknown, both closed boxes must meet 0.40 confidence, and face filtering uses the detection center. Device auto-selection is printed. Haar helpers remain because other scripts use them. |

## Verification results

These are code-level checks with synthetic inputs, not measurements from a person or camera.

| Check | Result |
|---|---|
| Full automated suite | **66 passed** |
| Python compilation | Passed |
| Dependency consistency (`pip check`) | Passed |
| `realtime.py --help` and offline video runner help | Passed |
| Repository whitespace check | Passed; Git only reported expected LF-to-CRLF working-copy notices. |
| Trained checkpoint hash | Unchanged: `6D64CDE43C226A1163BB3268CBFB99C4EB0FED50B8F69DA4E066AFBE2CED5597` |
| Webcam/demo video before-and-after table | Not available: no demo videos were present to run. No event latency, false-alarm count, silent-gap, or median webcam FPS is claimed. |

The test suite checks timers, state recovery, eye signal combinations, alarm state/policy, calibration acceptance, pose fallback, detector pair interpretation, and metrics columns. Passing these tests does not establish real-driver reliability, landmark accuracy, physical speaker pitch, or road safety. The attached task requested writing failing tests first; that order was not preserved for every item in this completed patch, so do not present this as strict test-first development.

## Still needs real-world checking

- Glasses and sunglasses, low light, backlight, camera placement, face angle, and different people can change landmarks and detection quality.
- The largest face is only a driver proxy; it does not establish identity or distinguish the driver from a passenger reliably.
- The head-pose thresholds and EAR calibration need checking on the intended laptop camera; camera orientation can change the sign and scale of pitch/yaw.
- No consented, labeled driving/demo clips were available. Alert recall, false alarms per hour, delay to alert, time to silence, and real webcam FPS remain unmeasured.
- The audible output and pitch have not been verified from a microphone recording after the mixer-format fix.
- YOLO mAP is object-detection performance, not proof that the drowsiness alarm is reliable.
- This system is a portfolio prototype. Do not rely on it as a real driving safety device.
