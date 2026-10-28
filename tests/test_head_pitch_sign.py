from src.config import HEAD_DOWN_DIRECTION
from src.drowsiness import DrowsinessMonitor
from src.face_signals import angle_delta


def test_pitch_sign_convention_is_positive_for_observed_head_down_direction():
    # The current overlay applied -1 and showed a negative value while this camera looked down.
    # Therefore the raw delta is positive and must no longer be flipped.
    assert HEAD_DOWN_DIRECTION == 1.0
    assert angle_delta(35.0, 0.0) * HEAD_DOWN_DIRECTION > 0
    assert angle_delta(-35.0, 0.0) * HEAD_DOWN_DIRECTION < 0


def test_head_down_reaches_critical_but_head_back_does_not():
    down = DrowsinessMonitor({})
    back = DrowsinessMonitor({})
    for index in range(13 * 5):
        now = index / 13
        down.update(now, False, False, head_pitch=35.0)
        back.update(now, False, False, head_pitch=-35.0)
    assert down.state() == "CRITICAL"
    assert back.state() == "NORMAL"
