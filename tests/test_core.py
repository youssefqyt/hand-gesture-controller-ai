import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

from app import GestureFilter, SessionAnalytics, Settings, finger_states


def landmarks(x=.5, thumb=.3, raised=(False, False, False, False)):
    points = [SimpleNamespace(x=x, y=.8) for _ in range(21)]
    points[3] = SimpleNamespace(x=.4 if thumb else .6, y=.5)
    points[4] = SimpleNamespace(x=.3 if thumb else .7, y=.4)
    for tip, pip, up in zip((8, 12, 16, 20), (6, 10, 14, 18), raised):
        points[tip] = SimpleNamespace(x=.5, y=.2 if up else .8)
        points[pip] = SimpleNamespace(x=.5, y=.5)
    return points


class CoreTests(unittest.TestCase):
    def test_finger_states_follow_hand_side(self):
        self.assertEqual(finger_states(landmarks(thumb=True), "Right"), [1, 0, 0, 0, 0])
        self.assertEqual(finger_states(landmarks(thumb=False), "Left"), [1, 0, 0, 0, 0])

    def test_filter_requires_stable_pose_and_resets_on_hand_loss(self):
        f = GestureFilter(3)
        pose = ("FIST", "ESC")
        self.assertEqual(f.update(pose)[0], "NO HAND")
        f.update(pose)
        self.assertEqual(f.update(pose), pose)
        self.assertEqual(f.update(None), ("NO HAND", "NO ACTION"))

    def test_analytics_persists_compact_session(self):
        temp = Path(".test-session-data")
        try:
            temp.mkdir(exist_ok=True)
            path = temp / "stats.json"
            a = SessionAnalytics(str(path))
            a.frame("ONE FINGER")
            a.frame("NO HAND")
            a.action("LEFT CLICK")
            a.save()
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["sessions"][0]["frames"], 2)
            self.assertEqual(saved["sessions"][0]["actions"], {"LEFT CLICK": 1})
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    def test_controls_default_disarmed(self):
        self.assertFalse(Settings().enabled)


if __name__ == "__main__":
    unittest.main()
