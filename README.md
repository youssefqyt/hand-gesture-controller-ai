# Hand Gesture Controller

A privacy-first desktop application that turns live webcam hand poses into mouse and keyboard controls. It uses MediaPipe's pretrained hand-landmark model locally; camera frames are processed in memory and are not uploaded or recorded. The app retains the original project's OpenCV preview and gesture-control purpose, with configurable tracking, safer input gating, and local session summaries.

## Who it is for

The controller is intended for desktop users who want hands-free pointer movement and a small set of gesture shortcuts, including accessibility and presentation use. It is a prototype interaction tool, not a replacement for an assistive device, and gesture reliability depends on camera placement, lighting, and hand visibility.

## Features

- Live, mirrored camera preview with hand landmarks, current pose/action, FPS, and control state.
- Local MediaPipe hand tracking with handedness-aware thumb rules and normalized pinch distance.
- Stable-pose filtering to reduce flicker and edge-triggered discrete actions to prevent held gestures from repeating.
- Index-finger pointer movement with smoothing and screen-boundary clamping.
- Explicit input arming: computer control is off by default and can be armed with Space.
- Independent mouse/keyboard toggles, emergency `Q` exit, and PyAutoGUI corner fail-safe.
- `--dry-run` recognition mode that never arms or sends input.
- Local JSON session analytics with duration, frame counts, recognized gesture counts, and executed action counts (last 100 sessions).
- Tunable settings through `.env`; useful runtime events and failures are logged.

### Gesture map

| Pose | Action |
| --- | --- |
| Index finger only | Move pointer |
| Thumb and index pinch | Left click |
| Index and middle fingers | Left click |
| Thumb up | Right click |
| Closed fist | Escape |
| Three fingers (middle, ring, pinky) | Right arrow / next |
| Open hand | Neutral / pause |

The pointer pose does not issue discrete actions. A pose must remain stable for `GESTURE_STABILITY_FRAMES` before it is recognized. Actions execute once on entry to a pose, subject to the cooldown.

## Technology and architecture

This is a local Python desktop application; it has no web server, remote API, user account system, or database. OpenCV captures and displays frames, MediaPipe Hands estimates 21 landmarks and handedness, and PyAutoGUI performs local computer input. `app.py` contains environment configuration, pose classification, temporal stabilization, analytics persistence, and the camera/action loop. `tests/test_core.py` covers the pure logic and persistence behavior without camera hardware.

```text
Webcam -> OpenCV frame capture -> MediaPipe landmarks
       -> handedness-aware pose classifier -> temporal gesture filter
       -> OpenCV preview                       -> PyAutoGUI (only when armed)
       -> local session analytics JSON
```

All processing happens on the user's machine. The analytics file can contain interaction counts and timestamps, but no images, landmarks, or personally identifying profile data. There are no user roles or remote permissions; operating-system camera and input permissions govern access.

## Project layout

```text
app.py                 Application, gesture logic, camera loop, analytics
tests/test_core.py     Hardware-independent unit tests
requirements.txt       Runtime dependencies
.env.example           Supported configuration reference
README.md              Setup, operation, and limitations
```

## Requirements and installation

- Windows, macOS, or Linux desktop with a webcam and GUI session.
- Python 3.10–3.12 recommended. MediaPipe wheels may not yet support the newest Python release.
- Camera permissions and a functioning desktop display. PyAutoGUI may require OS accessibility/input permissions (notably macOS).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS/Linux, create and activate the virtual environment with `python3.12 -m venv .venv` and `source .venv/bin/activate`; copy the sample with `cp .env.example .env`.

## Configuration

The application reads `.env` from the working directory (or process environment variables). Defaults are listed in `.env.example`. Important options:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CAMERA_INDEX` | `0` | OpenCV camera device index |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | `960`, `540` | Requested camera capture size |
| `CAMERA_MIRROR` | `true` | Mirror preview and pointer mapping |
| `CONTROLS_ENABLED` | `false` | Startup arming preference; Space toggles at runtime |
| `MOUSE_CONTROL_ENABLED`, `KEYBOARD_CONTROL_ENABLED` | `true` | Initial per-device controls |
| `DETECTION_CONFIDENCE`, `TRACKING_CONFIDENCE` | `0.65` | MediaPipe confidence thresholds (0–1) |
| `MOUSE_SMOOTHING` | `0.28` | Pointer interpolation (0–1; higher follows faster) |
| `ACTION_COOLDOWN` | `0.8` | Minimum seconds between different discrete actions |
| `GESTURE_STABILITY_FRAMES` | `4` | Consecutive frames required to accept a pose |
| `PINCH_THRESHOLD` | `0.055` | Normalized landmark distance for pinch detection |
| `STATS_FILE` | `data/session_stats.json` | Local JSON analytics destination |
| `LOG_LEVEL` | `INFO` | Python logging level |

## Run and use

```powershell
python app.py
python app.py --dry-run
python app.py --version
```

The window starts disarmed. Position your hand in view and use `--dry-run` first to check recognition without computer input. In the camera window press Space to arm/disarm; `M` toggles pointer control; `K` toggles keyboard shortcuts; `C` centers the pointer; `Q` exits. Moving the pointer into a screen corner also triggers PyAutoGUI's fail-safe. Session summaries append to the configured JSON file on exit.

## Data and integrations

There is no database. `SessionAnalytics` appends compact summaries to the JSON `sessions` array, keeps the latest 100 entries, and writes atomically via a temporary sibling file. Failure to save a summary is logged and does not prevent shutdown. There are no HTTP endpoints or third-party cloud integrations. MediaPipe's model is the only AI component: it estimates hand landmarks locally; a deterministic rule classifier and temporal filter map those landmarks to actions. The project does not call a generative AI service or train a custom model.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests exercise hand-side thumb logic, pose stabilization/reset, default disarmed state, and analytics persistence. Camera, OS input injection, and actual model accuracy require manual verification on a desktop with a webcam.

## Deployment

This app is deployed by installing its Python dependencies on the target desktop and launching `python app.py`; it is not a hosted service. For redistribution, package it with a tool such as PyInstaller on each target OS, include the MediaPipe model assets required by the installed package, and test camera access and OS input permissions on the target system. No container or server deployment configuration is included because webcam capture and desktop input require an interactive user session.

## Troubleshooting

- **Camera cannot open:** close other camera applications, check OS camera permissions, and try another `CAMERA_INDEX`.
- **Missing dependency / model import error:** use Python 3.10–3.12 in a clean environment and reinstall `requirements.txt`.
- **Poor recognition:** improve front lighting, keep the whole hand in frame, reduce background clutter, and adjust confidence or stability settings.
- **Pointer feels slow or jittery:** tune `MOUSE_SMOOTHING` (larger values react faster); verify `CAMERA_MIRROR` matches your expected orientation.
- **No computer input:** press Space to arm, ensure its specific M/K toggle is on, and grant OS accessibility permissions. `--dry-run` always disables computer input.
- **Fail-safe interrupted control:** move the pointer away from the screen corner and restart if needed.
- **Analytics did not save:** verify the configured directory is writable; the app logs persistence errors.

## Limitations and future work

Recognition uses a single hand and fixed pose rules; it does not learn user-specific gestures. The thumb heuristic can vary with rotation and camera perspective. Open-hand pause is neutral but does not suspend tracking. The app has no settings dashboard, profiles, calibration workflow, or installer. Useful next steps include configurable gesture bindings with an on-screen calibration flow, optional second-hand/chord gestures, adaptive pointer filtering, accessibility evaluation, and signed platform-specific installers. These are not represented as implemented features.
