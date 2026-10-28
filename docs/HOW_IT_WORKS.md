# How VigilantDrive works

VigilantDrive is a webcam demonstration. It combines a trained YOLO checkpoint with face landmarks and time-based rules. The rules are a prototype; they are not a medical diagnosis or a certified vehicle-safety system.

## From camera frame to alert

1. `realtime.py` starts `src/main.py`. The app reads `config.yaml`, then adds the numeric defaults from `src/config.py`. `config.yaml` identifies the checkpoint, device, and output folders. `src/config.py` contains the timing, score, camera-quality, and sound defaults.
2. OpenCV reads a camera or video frame. MediaPipe Face Mesh finds face landmarks. The largest detected face is used as the driver proxy. If MediaPipe cannot see a face, YOLO eye/yawn detections are not used for that frame.
3. During startup, `FaceCalibration` waits for a visible face, then collects at least 40 valid samples over about five seconds. It stores median eye openness, head pitch, and head turn. It rejects an EAR baseline outside 0.18–0.45 or pitch variation above 10 degrees. The person should look forward with eyes open. Press **C** to reset and repeat calibration; alerts stay silent while calibration is in progress.
4. `src/detector.py` runs the existing YOLO checkpoint for `Eyeopen`, `Eyeclosed`, and `Yawn`. `src/face_signals.py` calculates eye aspect ratio (EAR) from face landmarks and estimates head pitch/yaw with `solvePnP`. EAR is the vertical opening of the eye divided by its width; it usually decreases when the eye closes.
5. `src/eye_logic.py` gives EAR priority: open EAR means open; moderate closed EAR needs YOLO confirmation; very low EAR can independently mean closed; when EAR is unavailable, YOLO is used. EAR is ignored during a head turn above 25 degrees. Eye states need three frames and 0.15 seconds to change in either direction, filtering brief blinks without adding extra reopening time.
6. `src/drowsiness.py` turns observations into a state. It counts elapsed seconds rather than assuming a fixed frame rate. PERCLOS is ignored until 10 seconds of history; it has separate enter/exit thresholds. A short noisy measurement does not erase head, face, or unknown-eye timers. Five seconds of recovered signals clear old eye/yawn history.
7. `src/alarm.py` maps that one state to a sound pattern. It has a separate sound thread, so tones do not block video processing. The function returns the state the alarm actually holds, and that same state is shown in the overlay and session log.
8. The overlay shows state, message, eye state, EAR and its cutoff, PERCLOS, head angles, FPS, frame time, camera quality, calibration, and per-stage timings. Display values are smoothed; decisions and CSV values use raw measurements. Press **D** to hide/show diagnostics, **C** to recalibrate, and **Q** to quit. A CSV is written under `results/`.

## Decision and alarm rules

The defaults are in `src/config.py`. They are starting values for a laptop demo, not industry-certified thresholds.

| Signal | Starting rule | Result |
|---|---|---|
| Eye closure | 2 seconds | DROWSY |
| Eye closure | 4 seconds | CRITICAL |
| Unknown eyes, face visible | 1 second | WARNING; later unknown time is treated conservatively in PERCLOS |
| PERCLOS | At or above configured warning/drowsy values | At least WARNING; it is not disabled when coverage is low |
| PERCLOS history | 10 seconds | Do not use PERCLOS to set an alert before enough history is present |
| PERCLOS state recovery | WARNING 0.25 enter / 0.15 exit; DROWSY 0.35 enter / 0.25 exit | Different enter/exit values reduce state chatter |
| Head pitch down from calibration | 25 degrees for 2 seconds | WARNING |
| Head pitch down from calibration | 25 degrees for 4 seconds | CRITICAL |
| Head turned from calibration | 35 degrees for 3 seconds | WARNING (“Distracted”) |
| Face missing while frames are arriving | 3 seconds | DRIVER_NOT_FOUND warning |
| Camera dark/blurred or feed lost | Configured image-quality rule / 2 seconds lost | CAMERA_UNCLEAR warning |
| Alert clearing | Eyes open, face visible, camera clear, head up for 2 seconds; alert has lasted at least 3 seconds | Alarm may return to NORMAL; brief unsafe readings get 0.3 seconds of grace |
| Timer noise | 0.5 seconds | One bad face/head/unknown reading does not erase that condition's timer |
| Recovered history | 5 seconds of eyes open, face visible, camera clear, head up | Clear stale PERCLOS/yawn evidence |
| WARNING audio | At most two beeps per warning episode | Warning stays visible; audio stops until the state changes |
| Eye debounce | 3 frames and 0.15 seconds to close or open | Avoids normal blink inflation from a slow reopen rule |

The policy table for each state also contains its display message, tone frequency, volume, and repeat interval. WARNING beeps twice per episode. DROWSY repeats a pulse about every half-second, and CRITICAL repeats faster and louder. The audio buffer uses the sample rate and channel count reported by the active mixer. The audio loop checks at short intervals on a worker thread. If pygame or the system audio device cannot start, the app prints a warning; visual alert state still works.

PERCLOS here means estimated closed-eye seconds divided by time in the rolling window. For a face-visible unknown-eye period, the first second is unknown; after that it is conservatively treated as closed for the risk calculation and raises at least WARNING. If the face is absent, the system does not claim that its eyes are closed; it raises DRIVER_NOT_FOUND after the configured delay.

## What it can and cannot establish

- MediaPipe's EAR/head-angle signals require the added `mediapipe` package. They do not retrain or replace the YOLO model. `requirements.txt` pins the dependency.
- The largest face is only a proxy for the driver. This is not identity tracking and can select the wrong person if another face is larger.
- Head pose depends on camera angle, face landmarks, and calibration. The pitch sign (`HEAD_DOWN_DIRECTION`) may need adjustment for the camera orientation. Check the on-screen angle while deliberately looking down before relying on it.
- Brightness and sharpness are simple image-quality checks. They can warn on a dim or naturally soft image; test and tune them for the camera.
- Running MediaPipe on every frame adds CPU work. YOLO is run every second frame by default and its latest boxes are reused between runs. Actual FPS depends on the PC and must be measured on that PC.
- JPEG review clips now store one resized frame every three camera frames, with the output frame rate adjusted to preserve their duration. The CSV and debug overlay include capture, face-mesh, YOLO, JPEG, drawing, and display timings. The webcam FPS target has not been measured in this environment.
- The app's state tests use made-up inputs. They confirm the rule logic, not face-landmark accuracy, real-driver event recall, false-alarm rate, or safe use on a road.
- `scripts/test_video.py` simulates tone start times from the configured policy. Its silent-gap number is a prediction from the tone schedule, not a recording of actual speaker output.
- The demo video mentioned in the request was not present in the repository when this note was written. No before/after video result is claimed.
