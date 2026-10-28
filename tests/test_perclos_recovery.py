from src.drowsiness import DrowsinessMonitor


def test_perclos_only_alert_clears_after_two_seconds_of_recovery():
    monitor = DrowsinessMonitor({})
    # Frequent short closures build high PERCLOS without triggering the direct 2 s rule.
    for index in range(13 * 20):
        now = index / 13
        eye_closed = index % 13 < 5
        monitor.update(now, eye_closed, False)
    assert monitor.state() == "DROWSY"

    # Open eyes and keep the face visible, camera clear, and head up.
    for index in range(1, 13 * 3 + 1):
        now = 20.0 + index / 13
        monitor.update(now, False, False, head_pitch=0.0)
        monitor.state()
    assert monitor.state() == "NORMAL"
    assert "recovery" in monitor.reason().lower()
