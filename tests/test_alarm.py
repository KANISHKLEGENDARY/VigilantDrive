import pytest
import numpy as np
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


def test_perclos_recovery_can_clear_after_two_safe_seconds_before_minimum_hold():
    alarm = Alarm(open_clear_seconds=2.0, minimum_alarm_seconds=3.0, audio_enabled=False)
    assert alarm.update("DROWSY", 0.0) == "DROWSY"
    assert alarm.update("NORMAL", 0.5, safe_to_clear=True) == "DROWSY"
    assert alarm.update("NORMAL", 2.5, safe_to_clear=True,
                        allow_early_clear=True) == "NORMAL"
    alarm.close()


def test_a_drowsy_alarm_does_not_drop_to_warning_on_one_safe_frame():
    alarm = Alarm(audio_enabled=False)
    assert alarm.update("DROWSY", 0.0) == "DROWSY"
    assert alarm.update("WARNING", 0.5, safe_to_clear=False) == "DROWSY"
    assert alarm.update("WARNING", 4.0, safe_to_clear=True) == "DROWSY"
    assert alarm.update("WARNING", 6.1, safe_to_clear=True) == "WARNING"
    alarm.close()


def test_alarm_clears_despite_a_brief_unsafe_frame_during_recovery():
    alarm = Alarm(open_clear_seconds=2.0, minimum_alarm_seconds=3.0, audio_enabled=False)
    assert alarm.update("CRITICAL", 0.0) == "CRITICAL"
    for index in range(1, 70):
        now = index / 13
        safe = index % 7 != 0
        state = alarm.update("NORMAL", now, safe_to_clear=safe)
    assert state == "NORMAL"
    alarm.close()


def test_warning_beeps_stop_after_two_but_drowsy_has_no_beep_limit():
    warning = ALERT_POLICIES["WARNING"]
    drowsy = ALERT_POLICIES["DROWSY"]
    assert Alarm._tone_allowed(warning, 0)
    assert Alarm._tone_allowed(warning, 1)
    assert not Alarm._tone_allowed(warning, 2)
    assert Alarm._tone_allowed(drowsy, 100)


def test_normal_replaces_queued_sound_and_stops_current_channel():
    alarm = Alarm(audio_enabled=False)

    class FakeChannel:
        stopped = False
        def stop(self):
            self.stopped = True

    alarm._channel = FakeChannel()
    alarm._enabled = True
    alarm._actual_state = "DROWSY"
    alarm._commands.put("WARNING")
    alarm._commands.put("CRITICAL")
    alarm._set_state("NORMAL")
    assert alarm._commands.get_nowait() == "NORMAL"
    assert alarm._commands.empty()
    assert alarm._channel.stopped
    alarm._enabled = False
    alarm.close()


def test_tone_buffer_uses_active_mixer_format(monkeypatch):
    captured = {}

    class FakeSound:
        def __init__(self, buffer):
            captured["length"] = len(buffer)

    class FakeChannel:
        def play(self, sound):
            pass
        def stop(self):
            pass

    class FakeMixer:
        def init(self, **kwargs):
            pass
        def get_init(self):
            return 44100, -16, 2
        def Channel(self, number):
            return FakeChannel()
        def Sound(self, buffer):
            return FakeSound(buffer)
        def quit(self):
            pass

    class FakePygame:
        mixer = FakeMixer()
        error = RuntimeError

    monkeypatch.setattr("src.alarm.pygame", FakePygame)
    alarm = Alarm(audio_enabled=True)
    assert alarm.sample_rate == 44100
    assert alarm.audio_channels == 2
    assert captured["length"] == int(44100 * 0.22) * 2 * np.dtype(np.int16).itemsize
    alarm.close()
