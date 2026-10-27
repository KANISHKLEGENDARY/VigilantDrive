from scripts.test_video import intervals, longest_silent_gap, tone_windows


def test_alert_intervals_capture_each_state_transition():
    assert intervals([0.0, 1.0, 2.0, 3.0], [False, True, True, False]) == [(1.0, 3.0)]


def test_longest_silent_gap_is_calculated_only_inside_labeled_drowsy_time():
    truth = [(0.0, 10.0), (20.0, 24.0)]
    alerts = [(2.0, 3.0), (6.0, 8.0), (21.0, 23.0)]
    assert longest_silent_gap(truth, alerts) == 3.0


def test_drowsy_tone_schedule_has_less_than_one_second_between_tones():
    times = [index / 10 for index in range(21)]
    states = ["DROWSY"] * len(times)
    tones = tone_windows(times, states)
    assert abs(longest_silent_gap([(0.0, 2.0)], tones) - 0.33) < 1e-6
