import pytest
from src.alarm import Alarm
from src.config import ALERT_POLICIES


@pytest.mark.parametrize("level", [name for name in ALERT_POLICIES if name != "NORMAL"])
def test_each_alert_policy_level_is_returned_as_the_actual_state(level):
    alarm = Alarm(audio_enabled=False)
    assert alarm.update(level, 0.0) == level
    alarm.close()


def test_state_cannot_clear_before_minimum_duration_and_safe_window():
    alarm = Alarm(open_clear_seconds=2.0, minimum_alarm_seconds=3.0, audio_enabled=False)
    assert alarm.update("DROWSY", 0.0) == "DROWSY"
    assert alarm.update("NORMAL", 1.0, safe_to_clear=True) == "DROWSY"
    assert alarm.update("NORMAL", 2.9, safe_to_clear=True) == "DROWSY"
    assert alarm.update("NORMAL", 3.1, safe_to_clear=True) == "NORMAL"
    alarm.close()


def test_a_drowsy_alarm_does_not_drop_to_warning_on_one_safe_frame():
    alarm = Alarm(audio_enabled=False)
    assert alarm.update("DROWSY", 0.0) == "DROWSY"
    assert alarm.update("WARNING", 0.5, safe_to_clear=False) == "DROWSY"
    assert alarm.update("WARNING", 4.0, safe_to_clear=True) == "DROWSY"
    assert alarm.update("WARNING", 6.1, safe_to_clear=True) == "WARNING"
    alarm.close()
