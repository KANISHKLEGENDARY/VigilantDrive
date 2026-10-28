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
    for t in range(3):
        monitor.update(float(t), True, False)
    assert monitor.perclos() == 1.0
    assert monitor.state() == "DROWSY"


def test_short_eye_closure_does_not_start_direct_alarm():
    monitor = DrowsinessMonitor(cfg())
    for t in range(10):
        monitor.update(float(t), False, False)
    for t in (10, 11, 12):
        monitor.update(float(t), t != 12, False)
    assert not monitor.eye_alarm_active()
    assert monitor.state() == "NORMAL"
    for t in range(13, 24):
        monitor.update(float(t), False, False)
    assert monitor.state() == "NORMAL"


def test_unknown_eye_time_becomes_a_warning_and_never_forces_normal():
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
    assert monitor.state() in {"WARNING", "DROWSY"}


def test_low_coverage_does_not_switch_off_conservative_perclos():
    sparse = DrowsinessMonitor(cfg())
    closed = DrowsinessMonitor(cfg())
    for t in range(10):
        sparse.update(float(t), None, False)
        closed.update(float(t), True if t == 0 else None, False)
    assert sparse.eye_coverage() < cfg()["minimum_eye_coverage"]
    assert closed.eye_coverage() < cfg()["minimum_eye_coverage"]
    assert sparse.perclos() > 0.0
    assert sparse._reliable_perclos() == 0.0
    assert sparse.score() == 0.0


def test_perclos_over_warning_threshold_never_reports_normal():
    settings = cfg()
    settings["recovery_clear_seconds"] = 100.0
    monitor = DrowsinessMonitor(settings)
    for index in range(13 * 11):
        monitor.update(index / 13, index % 13 < 4, False)
    assert monitor.perclos() >= cfg()["perclos_warning"]
    assert monitor.state() in {"WARNING", "DROWSY"}


def test_unknown_eyes_on_visible_face_raise_warning_after_one_second():
    monitor = DrowsinessMonitor(cfg())
    monitor.update(0.0, None, False, face_visible=True)
    monitor.update(1.1, None, False, face_visible=True)
    assert monitor.state() == "WARNING"


def test_face_missing_for_three_seconds_raises_driver_not_found():
    monitor = DrowsinessMonitor(cfg())
    monitor.update(0.0, None, False, face_visible=False)
    monitor.update(3.1, None, False, face_visible=False)
    assert monitor.state() == "DRIVER_NOT_FOUND"


def test_head_pose_escalates_by_duration():
    monitor = DrowsinessMonitor(cfg())
    monitor.update(0.0, False, False, head_pitch=26.0)
    monitor.update(2.1, False, False, head_pitch=26.0)
    assert monitor.state() == "WARNING"
    monitor.update(4.1, False, False, head_pitch=26.0)
    assert monitor.state() == "CRITICAL"


def test_sustained_head_turn_raises_distracted_warning():
    monitor = DrowsinessMonitor(cfg())
    monitor.update(0.0, False, False, head_yaw=36.0)
    monitor.update(3.1, False, False, head_yaw=36.0)
    assert monitor.state() == "WARNING"
    assert "Distracted" in monitor.reason()


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


def test_sustained_head_pitch_adds_to_score_and_warns():
    settings = cfg()
    settings.update({"head_down_angle_degrees": 25.0, "head_down_warning_seconds": 1.0,
                     "head_down_critical_seconds": 5.0, "head_drop_weight": 0.1})
    monitor = DrowsinessMonitor(settings)
    for t in range(3):
        monitor.update(float(t), False, False, head_pitch=0.0)
    baseline_score = monitor.score()
    for t in (3.0, 4.0, 5.0):
        monitor.update(t, False, False, head_pitch=26.0)
    assert monitor.head_drop_active
    assert monitor.score() > baseline_score
    assert monitor.state() == "WARNING"
    assert monitor.reason() == "Head-down posture"


def test_reset_clears_temporal_history():
    monitor = DrowsinessMonitor(cfg())
    monitor.update(0.0, True, True)
    monitor.reset()
    assert monitor.eye_status().startswith("UNKNOWN")
    assert monitor.yawn_events() == 0
    assert monitor.state() == "NORMAL"


def test_warning_recovers_after_closure_with_normal_eye_flicker():
    monitor = DrowsinessMonitor(cfg())
    for index in range(40):
        monitor.update(index / 13, True, False)
    assert monitor.state() in {"WARNING", "DROWSY"}
    for index in range(1, 781):
        now = 3.0 + index / 13
        monitor.update(now, index % 10 == 0, False)
        state = monitor.state()
        assert (monitor.reason() != "No drowsiness signal") or state == "NORMAL"
    assert monitor.state() == "NORMAL"


def test_driver_not_found_clears_after_face_returns():
    monitor = DrowsinessMonitor(cfg())
    for index in range(53):
        monitor.update(index / 13, None, False, face_visible=False)
    assert monitor.state() == "DRIVER_NOT_FOUND"
    for index in range(1, 15):
        monitor.update(4.0 + index / 13, False, False, face_visible=True)
    assert monitor.state() == "NORMAL"


def test_camera_unclear_clears_after_camera_is_clear():
    monitor = DrowsinessMonitor(cfg())
    monitor.update(0.0, False, False, camera_clear=False)
    assert monitor.state() == "CAMERA_UNCLEAR"
    for index in range(1, 15):
        monitor.update(index / 13, False, False, camera_clear=True)
    assert monitor.state() == "NORMAL"


def test_five_seconds_of_recovered_driving_clears_old_eye_history():
    monitor = DrowsinessMonitor(cfg())
    for index in range(39):
        monitor.update(index / 13, True, False)
    assert monitor.state() in {"DROWSY", "CRITICAL"}
    for index in range(1, 80):
        monitor.update(3.0 + index / 13, False, False)
    assert monitor.state() == "NORMAL"
    assert monitor.risk_samples[0][0] >= 7.5
    assert len(monitor.events) == 0


def test_perclos_does_not_trigger_before_minimum_session_history():
    monitor = DrowsinessMonitor(cfg())
    for index in range(66):
        now = index / 13
        monitor.update(now, now >= 0.5 and now < 2.0, False)
    assert monitor.perclos() > 0.25
    assert monitor.state() == "NORMAL"


def test_head_down_survives_one_noisy_frame_each_second():
    monitor = DrowsinessMonitor(cfg())
    for index in range(13 * 12):
        now = index / 13
        pitch = 20.0 if index % 13 == 0 else 35.0
        monitor.update(now, False, False, head_pitch=pitch)
    assert monitor.state() in {"WARNING", "DROWSY", "CRITICAL"}


def test_intermittent_face_detection_does_not_reset_head_timer():
    monitor = DrowsinessMonitor(cfg())
    for index in range(13 * 12):
        now = index / 13
        face_visible = index % 6 >= 2
        monitor.update(now, False, False, face_visible=face_visible,
                       head_pitch=35.0 if face_visible else None)
    assert monitor.state() == "CRITICAL"


def test_sparse_face_detection_still_raises_driver_not_found():
    monitor = DrowsinessMonitor(cfg())
    for index in range(13 * 12):
        monitor.update(index / 13, False, False, face_visible=index % 20 == 0)
    assert monitor.state() == "DRIVER_NOT_FOUND"
