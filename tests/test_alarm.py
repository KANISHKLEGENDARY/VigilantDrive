from src.alarm import Alarm


def test_alarm_starts_only_for_qualifying_drowsy_signal_and_clears_after_open_delay():
    alarm = Alarm(open_clear_seconds=0.6, escalation_seconds=(5.0, 10.0), audio_enabled=False)
    assert not alarm.update("WARNING", 0.0)
    assert not alarm.update("DROWSY", 1.0, eye_closed=False)
    assert alarm.update("DROWSY", 2.0, eye_closed=True)
    assert not alarm.update("DROWSY", 4.0, eye_closed=True)
    assert not alarm.update("DROWSY", 7.1, eye_closed=True)  # Tone level may escalate, alarm does not restart.
    assert not alarm.update("NORMAL", 7.2, eye_closed=False)
    assert not alarm.update("NORMAL", 7.9, eye_closed=False)
    assert alarm.update("DROWSY", 8.0, yawn_active=True)
    assert not alarm.update("DROWSY", 9.0, head_drop_active=True)
    assert not alarm.update("NORMAL", 9.1)
    assert not alarm.update("NORMAL", 10.0)
    assert alarm.update("DROWSY", 10.1, head_drop_active=True)
    alarm.close()
