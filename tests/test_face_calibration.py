from src.face_signals import FaceCalibration, FaceSignals, angle_delta


def test_calibration_uses_median_of_valid_open_eye_samples():
    calibration = FaceCalibration(seconds=5.0)
    for index in range(50):
        now = index / 10
        ready = calibration.update(now, FaceSignals(0.30, 2.0, -1.0, (0, 0, 20, 20)))
    ready = calibration.update(5.0, FaceSignals(0.30, 2.0, -1.0, (0, 0, 20, 20)))
    assert ready
    assert calibration.ear_baseline == 0.30
    assert calibration.pitch_baseline == 2.0
    assert calibration.yaw_baseline == -1.0


def test_calibration_waits_for_open_eye_samples_instead_of_guessing():
    calibration = FaceCalibration(seconds=5.0)
    for index in range(60):
        assert not calibration.update(index / 10, FaceSignals(0.10, 0.0, 0.0, (0, 0, 20, 20)))
    assert calibration.ear_baseline is None


def test_calibration_does_not_average_angles_across_the_wrap_boundary():
    calibration = FaceCalibration(seconds=1.0)
    for index in range(40):
        pitch = 179.0 if index % 2 == 0 else -179.0
        calibration.update(index / 40, FaceSignals(0.30, pitch, pitch, (0, 0, 20, 20)))
    calibration.update(1.0, FaceSignals(0.30, -179.0, -179.0, (0, 0, 20, 20)))
    assert abs(angle_delta(calibration.pitch_baseline, 179.0)) <= 2.1


def test_calibration_clock_starts_when_face_first_becomes_valid():
    calibration = FaceCalibration(seconds=5.0)
    for index in range(45):
        assert not calibration.update(index / 10, None)
    assert calibration.started_at is None
    for index in range(40):
        now = 4.5 + index / 10
        ready = calibration.update(now, FaceSignals(0.30, 2.0, 0.0, (0, 0, 20, 20)))
    assert not ready
    assert calibration.update(9.5, FaceSignals(0.30, 2.0, 0.0, (0, 0, 20, 20)))


def test_invalid_ear_or_unsteady_head_pose_rejects_calibration():
    calibration = FaceCalibration(seconds=1.0)
    for index in range(40):
        pitch = 0.0 if index % 2 == 0 else 15.0
        calibration.update(index / 40, FaceSignals(0.17, pitch, 0.0, (0, 0, 20, 20)))
    assert not calibration.update(1.0, FaceSignals(0.17, 0.0, 0.0, (0, 0, 20, 20)))
    assert calibration.failed
    assert not calibration.ready
    assert calibration.ear_baseline is None


def test_reset_discards_old_calibration_samples():
    calibration = FaceCalibration(seconds=1.0)
    for index in range(40):
        calibration.update(index / 40, FaceSignals(0.30, 1.0, 1.0, (0, 0, 20, 20)))
    assert calibration.update(1.0, FaceSignals(0.30, 1.0, 1.0, (0, 0, 20, 20)))
    calibration.reset()
    assert calibration.started_at is None
    assert calibration.ear_baseline is None
    assert not calibration.ready
