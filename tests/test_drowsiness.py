from src.drowsiness import DrowsinessMonitor


def cfg():
    return {"window_seconds": 10.0, "perclos_warning": 0.15, "perclos_drowsy": 0.35,
            "yawn_window_seconds": 60.0, "yawn_debounce_seconds": 2.0, "yawn_min_duration_seconds": 2.0,
            "yawn_gap_tolerance_seconds": 0.35, "eye_weight": 0.75, "yawn_weight": 0.25,
            "score_warning": 0.45, "score_drowsy": 0.70, "hysteresis_margin": 0.10,
            "closed_eye_alarm_seconds": 1.5, "minimum_eye_coverage": 0.5,
            "yawn_events_for_alarm": 2}


def test_open_signal_is_normal():
    monitor = DrowsinessMonitor(cfg())
    for t in range(11):
        monitor.update(float(t), False, False)
    assert monitor.state() == "NORMAL"
    assert monitor.eye_coverage() == 1.0


def test_sustained_closure_becomes_drowsy():
    monitor = DrowsinessMonitor(cfg())
    for t in range(12):
        monitor.update(float(t), True, False)
    assert monitor.perclos() == 1.0
    assert monitor.state() == "DROWSY"


def test_short_blink_warns_but_does_not_start_direct_alarm():
    monitor = DrowsinessMonitor(cfg())
    for t in range(10):
        monitor.update(float(t), False, False)
    for t in (10, 11, 12):
        monitor.update(float(t), t != 12, False)
    assert not monitor.eye_alarm_active()
    assert monitor.state() == "WARNING"
    for t in range(13, 24):
        monitor.update(float(t), False, False)
    assert monitor.state() == "NORMAL"


def test_unknown_eye_time_is_not_open_and_stale_perclos_loses_authority():
    monitor = DrowsinessMonitor(cfg())
    for t in range(5):
        monitor.update(float(t), False, False)
    for t in (5, 6, 7):
        monitor.update(float(t), True, False)
    assert monitor.eye_alarm_active()
    assert monitor.state() == "DROWSY"
    for t in range(8, 26):
        monitor.update(float(t), None, False)
    assert monitor.eye_status().startswith("UNKNOWN")
    assert monitor.eye_coverage() < cfg()["minimum_eye_coverage"]
    assert monitor.state() == "NORMAL"


def test_low_eye_coverage_makes_closed_observations_not_change_perclos_score():
    sparse = DrowsinessMonitor(cfg())
    closed = DrowsinessMonitor(cfg())
    for t in range(10):
        sparse.update(float(t), None, False)
        closed.update(float(t), True if t == 0 else None, False)
    assert sparse.eye_coverage() < cfg()["minimum_eye_coverage"]
    assert closed.eye_coverage() < cfg()["minimum_eye_coverage"]
    assert sparse.score() == closed.score()


def test_only_sustained_yawn_episodes_count():
    monitor = DrowsinessMonitor(cfg())
    for t in (0, 1):
        monitor.update(float(t), False, True)
    monitor.update(2.0, False, False)
    assert monitor.yawn_events() == 0
    for t in range(3, 7):
        monitor.update(float(t), False, False)
    for t in range(7, 11):
        monitor.update(float(t), False, True)
    assert monitor.yawn_events() == 1
    for t in range(11, 14):
        monitor.update(float(t), False, False)
    for t in range(14, 18):
        monitor.update(float(t), False, True)
    assert monitor.yawn_events() == 2
    assert monitor.yawn_alert_active()


def test_sustained_downward_face_movement_adds_to_score():
    settings = cfg()
    settings.update({"head_drop_baseline_seconds": 2.0, "head_drop_offset_fraction": 0.1,
                     "head_drop_duration_seconds": 1.0, "head_drop_weight": 0.1})
    monitor = DrowsinessMonitor(settings)
    for t in range(3):
        monitor.update(float(t), False, False, face_center_y=0.4)
    baseline_score = monitor.score()
    for t in (3.0, 4.0, 5.0):
        monitor.update(t, False, False, face_center_y=0.6)
    assert monitor.head_drop_active
    assert monitor.score() > baseline_score
    assert monitor.state() == "DROWSY"
    assert monitor.reason() == "Sustained downward face position"


def test_reset_clears_temporal_history():
    monitor = DrowsinessMonitor(cfg())
    monitor.update(0.0, True, True)
    monitor.reset()
    assert monitor.eye_status().startswith("UNKNOWN")
    assert monitor.yawn_events() == 0
    assert monitor.state() == "NORMAL"
