from src.config import EAR_CLOSED_BASELINE_FRACTION
from src.eye_logic import combine_eye_signals, ear_allows_yolo_override


def test_borderline_closed_ear_can_be_overridden_by_strong_two_eye_yolo():
    assert EAR_CLOSED_BASELINE_FRACTION == 0.70
    # EAR is above its closed cutoff but below 0.85 of baseline.
    assert combine_eye_signals(True, False, ear_yolo_override=True) is True
    # Clearly open EAR cannot be overridden by YOLO.
    assert combine_eye_signals(True, False, ear_yolo_override=False) is False
    # Clearly closed EAR conflicts with a both-open YOLO result: do not guess.
    assert combine_eye_signals(False, True) is None


def test_yolo_override_zone_is_bounded_by_closed_cutoff_and_clear_open_limit():
    assert not ear_allows_yolo_override(0.69, 1.0)
    assert ear_allows_yolo_override(0.70, 1.0)
    assert ear_allows_yolo_override(0.849, 1.0)
    assert not ear_allows_yolo_override(0.85, 1.0)
