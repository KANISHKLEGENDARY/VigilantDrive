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
    for index in range(10):
        pitch = 179.0 if index % 2 == 0 else -179.0
        calibration.update(index / 10, FaceSignals(0.30, pitch, pitch, (0, 0, 20, 20)))
    calibration.update(1.0, FaceSignals(0.30, -179.0, -179.0, (0, 0, 20, 20)))
    assert abs(angle_delta(calibration.pitch_baseline, 179.0)) <= 2.1
