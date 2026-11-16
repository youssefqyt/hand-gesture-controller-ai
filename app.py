"""Hand Gesture Controller: local, privacy-first webcam interaction."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LOG = logging.getLogger("gesture-controller")


@dataclass(frozen=True)
class Settings:
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    detection_confidence: float = 0.65
    tracking_confidence: float = 0.65
    smoothing: float = 0.28
    action_cooldown: float = 0.8
    gesture_frames: int = 4
    pinch_threshold: float = 0.055
    enabled: bool = False
    mouse_enabled: bool = True
    keyboard_enabled: bool = True
    mirror: bool = True
    stats_file: str = "data/session_stats.json"

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            camera_index=int(os.getenv("CAMERA_INDEX", defaults.camera_index)),
            frame_width=int(os.getenv("FRAME_WIDTH", defaults.frame_width)),
            frame_height=int(os.getenv("FRAME_HEIGHT", defaults.frame_height)),
            detection_confidence=float(os.getenv("DETECTION_CONFIDENCE", defaults.detection_confidence)),
            tracking_confidence=float(os.getenv("TRACKING_CONFIDENCE", defaults.tracking_confidence)),
            smoothing=float(os.getenv("MOUSE_SMOOTHING", defaults.smoothing)),
            action_cooldown=float(os.getenv("ACTION_COOLDOWN", defaults.action_cooldown)),
            gesture_frames=int(os.getenv("GESTURE_STABILITY_FRAMES", defaults.gesture_frames)),
            pinch_threshold=float(os.getenv("PINCH_THRESHOLD", defaults.pinch_threshold)),
            enabled=os.getenv("CONTROLS_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
            mouse_enabled=os.getenv("MOUSE_CONTROL_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
            keyboard_enabled=os.getenv("KEYBOARD_CONTROL_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
            mirror=os.getenv("CAMERA_MIRROR", "true").lower() in {"1", "true", "yes", "on"},
            stats_file=os.getenv("STATS_FILE", defaults.stats_file),
        )


def finger_states(landmarks: Any, handedness: str = "Right") -> list[int]:
    """Return thumb/index/middle/ring/pinky raised flags from MediaPipe landmarks."""
    thumb_is_right = handedness.lower() != "left"
    thumb_up = landmarks[4].x < landmarks[3].x if thumb_is_right else landmarks[4].x > landmarks[3].x
    return [int(thumb_up), *(int(landmarks[tip].y < landmarks[pip].y)
                              for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))]


def recognize_gesture(landmarks: Any, handedness: str = "Right", pinch_threshold: float = 0.055) -> tuple[str, str]:
    """Classify a static pose; temporal filtering is applied by GestureFilter."""
    thumb, index, middle, ring, pinky = finger_states(landmarks, handedness)
    pinch = math.hypot(landmarks[4].x - landmarks[8].x, landmarks[4].y - landmarks[8].y)
    if pinch < pinch_threshold:
        return "PINCH", "LEFT CLICK"
    pose = [thumb, index, middle, ring, pinky]
    mapping = {
        (1, 1, 1, 1, 1): ("OPEN HAND", "PAUSE"),
        (0, 0, 0, 0, 0): ("FIST", "ESC"),
        (0, 1, 0, 0, 0): ("ONE FINGER", "MOUSE MOVE"),
        (0, 1, 1, 0, 0): ("TWO FINGERS", "LEFT CLICK"),
        (1, 0, 0, 0, 0): ("THUMBS UP", "RIGHT CLICK"),
        (0, 0, 1, 1, 1): ("THREE FINGERS", "NEXT"),
    }
    return mapping.get(tuple(pose), ("UNKNOWN", "NO ACTION"))


class GestureFilter:
    """Require a stable pose before changing output; avoid flicker-triggered actions."""
    def __init__(self, required_frames: int = 4):
        self.required_frames = max(1, required_frames)
        self.history: deque[tuple[str, str]] = deque(maxlen=self.required_frames)
        self.current = ("NO HAND", "NO ACTION")

    def update(self, candidate: tuple[str, str] | None) -> tuple[str, str]:
        if candidate is None:
            self.history.clear()
            self.current = ("NO HAND", "NO ACTION")
            return self.current
        self.history.append(candidate)
        if len(self.history) == self.required_frames and len(set(self.history)) == 1:
            self.current = candidate
        elif candidate[0] != self.current[0] and self.history and self.history[-1][0] != self.current[0]:
            # Keep the prior stable gesture until a new one is confirmed.
            pass
        return self.current


class SessionAnalytics:
    """Aggregate local session counts and safely persist a compact summary."""
    def __init__(self, path: str):
        self.path = Path(path)
        self.started = time.time()
        self.frames = 0
        self.hands_seen = 0
        self.gestures: Counter[str] = Counter()
        self.actions: Counter[str] = Counter()

    def frame(self, gesture: str) -> None:
        self.frames += 1
        if gesture != "NO HAND":
            self.hands_seen += 1
            if gesture not in {"UNKNOWN"}:
                self.gestures[gesture] += 1

    def action(self, action: str) -> None:
        self.actions[action] += 1

    def save(self) -> dict[str, Any]:
        summary = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            "duration_seconds": round(max(0, time.time() - self.started), 1),
            "frames": self.frames,
            "hand_frames": self.hands_seen,
            "gestures": dict(self.gestures),
            "actions": dict(self.actions),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existing = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"sessions": []}
            if not isinstance(existing, dict) or not isinstance(existing.get("sessions", []), list):
                existing = {"sessions": []}
            existing.setdefault("sessions", []).append(summary)
            existing["sessions"] = existing["sessions"][-100:]
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            temporary.replace(self.path)
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("Could not save session analytics to %s: %s", self.path, exc)
        return summary


def run(settings: Settings, dry_run: bool = False) -> int:
    try:
        import cv2
        import mediapipe as mp
        import pyautogui
    except ImportError as exc:
        LOG.error("Missing runtime dependency: %s. Install requirements.txt first.", exc.name)
        return 2

    pyautogui.PAUSE = 0.03
    pyautogui.FAILSAFE = True
    cap = cv2.VideoCapture(settings.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
    if not cap.isOpened():
        LOG.error("Cannot open camera %s. Check permissions or set CAMERA_INDEX.", settings.camera_index)
        return 2

    screen_w, screen_h = pyautogui.size()
    hands_api = mp.solutions.hands
    tracker = hands_api.Hands(max_num_hands=1, min_detection_confidence=settings.detection_confidence,
                              min_tracking_confidence=settings.tracking_confidence)
    drawer = mp.solutions.drawing_utils
    stable = GestureFilter(settings.gesture_frames)
    analytics = SessionAnalytics(settings.stats_file)
    mouse_x, mouse_y = screen_w // 2, screen_h // 2
    last_gesture, last_action_at = "", 0.0
    controls_on = settings.enabled and not dry_run
    mouse_on, keyboard_on = settings.mouse_enabled, settings.keyboard_enabled
    previous = time.perf_counter()
    fps = 0.0
    LOG.info("Camera ready. Controls %s. Q quit, Space arm/disarm, M mouse, K keyboard, C center.", "ARMED" if controls_on else "DISARMED")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                LOG.error("Camera stopped returning frames.")
                break
            if settings.mirror:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            result = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            candidate = None
            hand = None
            handedness = "Right"
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                if result.multi_handedness:
                    handedness = result.multi_handedness[0].classification[0].label
                candidate = recognize_gesture(hand.landmark, handedness, settings.pinch_threshold)
                drawer.draw_landmarks(frame, hand, hands_api.HAND_CONNECTIONS)
            gesture, action = stable.update(candidate)
            analytics.frame(gesture)

            if gesture == "ONE FINGER" and hand and mouse_on and controls_on:
                tip = hand.landmark[8]
                target_x = int(max(0, min(1, tip.x)) * (screen_w - 1))
                target_y = int(max(0, min(1, tip.y)) * (screen_h - 1))
                mouse_x += (target_x - mouse_x) * settings.smoothing
                mouse_y += (target_y - mouse_y) * settings.smoothing
                pyautogui.moveTo(int(mouse_x), int(mouse_y))

            # Discrete actions trigger only on entering a gesture. Held poses never repeat.
            now = time.monotonic()
            if gesture not in {"ONE FINGER", "OPEN HAND", "UNKNOWN", "NO HAND"} and gesture != last_gesture and now - last_action_at >= settings.action_cooldown:
                dispatched = False
                if controls_on:
                    if action == "LEFT CLICK" and mouse_on:
                        pyautogui.click()
                        dispatched = True
                    elif action == "RIGHT CLICK" and mouse_on:
                        pyautogui.rightClick()
                        dispatched = True
                    elif action == "ESC" and keyboard_on:
                        pyautogui.press("esc")
                        dispatched = True
                    elif action == "NEXT" and keyboard_on:
                        pyautogui.press("right")
                        dispatched = True
                    LOG.info("Gesture %s -> %s", gesture, action)
                if dispatched:
                    analytics.action(action)
                last_gesture, last_action_at = gesture, now
            if gesture in {"ONE FINGER", "OPEN HAND", "UNKNOWN", "NO HAND"}:
                last_gesture = ""

            current = time.perf_counter()
            instant = 1 / max(current - previous, 1e-6)
            fps = instant if fps == 0 else fps * .9 + instant * .1
            previous = current
            panel = frame.copy()
            cv2.rectangle(panel, (12, 12), (370, 183), (15, 22, 34), -1)
            cv2.addWeighted(panel, .78, frame, .22, 0, frame)
            color = (90, 225, 145) if controls_on else (75, 175, 245)
            lines = [(f"GESTURE CONTROL  |  {int(fps)} FPS", (235, 245, 255)),
                     (f"Gesture: {gesture}", (235, 245, 255)),
                     (f"Action: {action}", (165, 205, 255)),
                     (f"Controls: {'ARMED' if controls_on else 'DISARMED'}", color),
                     (f"Mouse {'ON' if mouse_on else 'OFF'}  |  Keyboard {'ON' if keyboard_on else 'OFF'}", (190, 200, 215))]
            for i, (label, ink) in enumerate(lines):
                cv2.putText(frame, label, (24, 40 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, .56, ink, 1, cv2.LINE_AA)
            cv2.putText(frame, "SPACE arm  M mouse  K keyboard  C center  Q quit", (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (230, 235, 245), 1, cv2.LINE_AA)
            cv2.imshow("Hand Gesture Controller", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and not dry_run:
                controls_on = not controls_on
                LOG.info("Controls %s", "armed" if controls_on else "disarmed")
            elif key == ord("m"):
                mouse_on = not mouse_on
            elif key == ord("k"):
                keyboard_on = not keyboard_on
            elif key == ord("c"):
                mouse_x, mouse_y = screen_w // 2, screen_h // 2
                if controls_on and mouse_on:
                    pyautogui.moveTo(mouse_x, mouse_y)
    except pyautogui.FailSafeException:
        LOG.warning("PyAutoGUI fail-safe activated; controls disarmed.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        summary = analytics.save()
        LOG.info("Session saved: %s gestures across %s frames", sum(summary["gestures"].values()), summary["frames"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Control mouse and keyboard with local hand gestures.")
    parser.add_argument("--dry-run", action="store_true", help="Recognize gestures without sending computer input")
    parser.add_argument("--version", action="version", version="Hand Gesture Controller 2.0.0")
    args = parser.parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(message)s")
    return run(Settings.from_env(), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
