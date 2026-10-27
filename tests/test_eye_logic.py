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


def test_one_wrong_open_frame_does_not_reopen_closed_eyes():
    eyes = EyeDebouncer()
    for index in range(3):
        eyes.update(True, index / 30)
    assert eyes.update(False, 0.1) is True
    assert eyes.update(True, 0.2) is True


def test_five_open_frames_and_stable_time_are_needed_to_reopen():
    eyes = EyeDebouncer()
    for index in range(3):
        eyes.update(True, index / 30)
    for now in (0.2, 0.27, 0.34, 0.41):
        assert eyes.update(False, now) is True
    assert eyes.update(False, 0.51) is False


def test_unknown_state_resets_after_one_second_without_claiming_open():
    eyes = EyeDebouncer()
    for index in range(3):
        eyes.update(True, index / 30)
    assert eyes.update(None, 0.2) is True
    assert eyes.update(None, 1.21) is None


def test_ear_closed_threshold_is_relative_to_calibration():
    assert ear_says_closed(0.15, 0.25) is True
    assert ear_says_closed(0.22, 0.25) is False
    assert ear_says_closed(None, 0.25) is None
