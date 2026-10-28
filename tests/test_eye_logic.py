from src.eye_logic import EyeDebouncer, combine_eye_signals, ear_says_closed


def test_yolo_or_ear_closed_is_a_closed_candidate():
    assert combine_eye_signals(True, False) is True
    assert combine_eye_signals(False, True) is True
    assert combine_eye_signals(False, None) is False
    assert combine_eye_signals(None, None) is None


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


def test_five_open_frames_and_stable_time_are_needed_to_reopen():
    eyes = EyeDebouncer()
    for now in (0.0, 0.1, 0.2):
        eyes.update(True, now)
    for now in (0.21, 0.28, 0.35, 0.42):
        assert eyes.update(False, now) is True
    assert eyes.update(False, 0.51) is False


def test_unknown_state_resets_after_one_second_without_claiming_open():
    eyes = EyeDebouncer()
    for now in (0.0, 0.1, 0.2):
        eyes.update(True, now)
    assert eyes.update(None, 0.3) is True
    assert eyes.update(None, 1.31) is None


def test_ear_closed_threshold_is_relative_to_calibration():
    assert ear_says_closed(0.15, 0.25) is True
    assert ear_says_closed(0.22, 0.25) is False
    assert ear_says_closed(None, 0.25) is None
