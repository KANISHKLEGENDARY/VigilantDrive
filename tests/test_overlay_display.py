from src.config import OVERLAY_SMOOTHING_ALPHA
from src.main import smooth_display


def test_display_smoothing_reduces_jumps_without_changing_first_value():
    assert smooth_display(None, 20.0) == 20.0
    assert smooth_display(20.0, 0.0) == (1.0 - OVERLAY_SMOOTHING_ALPHA) * 20.0
