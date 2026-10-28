from src.eye_logic import (EyeDebouncer, combine_eye_signals, ear_is_surely_closed,
                           ear_is_usable_at_yaw, ear_says_closed)
from src.drowsiness import DrowsinessMonitor


def test_ear_led_eye_decision_table():
    assert combine_eye_signals(True, False) is False
    assert combine_eye_signals(True, True) is True
    assert combine_eye_signals(False, True) is None
    assert combine_eye_signals(None, True) is True
    assert combine_eye_signals(True, None) is True
    assert combine_eye_signals(False, None) is False
    assert combine_eye_signals(None, None) is None
    assert combine_eye_signals(False, True, ear_sure_closed=True) is True


def test_three_closed_frames_are_needed_to_close():
    eyes = EyeDebouncer()
    assert eyes.update(True, 0.0) is None
    assert eyes.update(True, 0.1) is None
    assert eyes.update(True, 0.2) is True


def test_brief_normal_blink_does_not_start_closed_eye_timer():
    eyes = EyeDebouncer()
    for now in (0.0, 0.1, 0.2, 0.3, 0.4):
        eyes.update(False, now)
    assert eyes.state is False
    for now in (0.5, 0.55, 0.6):
        assert eyes.update(True, now) is False
    assert eyes.update(False, 0.65) is False


def test_sustained_close_still_becomes_closed_after_minimum_duration():
    eyes = EyeDebouncer()
    for now in (0.0, 0.05, 0.10):
        assert eyes.update(True, now) is None
    assert eyes.update(True, 0.15) is True


def test_one_wrong_open_frame_does_not_reopen_closed_eyes():
    eyes = EyeDebouncer()
    for now in (0.0, 0.1, 0.2):
        eyes.update(True, now)
    assert eyes.update(False, 0.21) is True
    assert eyes.update(True, 0.22) is True


def test_three_open_frames_and_stable_time_are_needed_to_reopen():
    eyes = EyeDebouncer()
    for now in (0.0, 0.1, 0.2):
        eyes.update(True, now)
    for now in (0.21, 0.28, 0.35):
        assert eyes.update(False, now) is True
    assert eyes.update(False, 0.36) is False


def test_unknown_state_resets_after_one_second_without_claiming_open():
    eyes = EyeDebouncer()
    for now in (0.0, 0.1, 0.2):
        eyes.update(True, now)
    assert eyes.update(None, 0.3) is True
    assert eyes.update(None, 1.31) is None


def test_ear_closed_threshold_is_relative_to_calibration():
    assert ear_says_closed(0.14, 0.25) is True
    assert ear_says_closed(0.16, 0.25) is False
    assert ear_says_closed(None, 0.25) is None
    assert ear_is_surely_closed(0.12, 0.25)
    assert not ear_is_surely_closed(0.14, 0.25)
    assert ear_is_usable_at_yaw(25.0)
    assert not ear_is_usable_at_yaw(25.1)


def test_normal_30_second_blinks_do_not_raise_a_warning():
    monitor = DrowsinessMonitor({})
    eyes = EyeDebouncer()
    for index in range(13 * 60):
        now = index / 13
        blink_closed = now % 3.0 < 0.3
        stable_eye_state = eyes.update(blink_closed, now)
        monitor.update(now, stable_eye_state, False)
    assert monitor.state() == "NORMAL"
    assert 0.05 <= monitor.perclos() <= 0.15
