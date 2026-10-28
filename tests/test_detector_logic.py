from src.detector import infer_eye_state, keep_detections_in_box, keep_driver_region
from scripts.evaluate_demo import match_events


def detection(name, confidence, box):
    return name, confidence, box


def test_eye_pair_must_be_geometrically_plausible_and_both_closed():
    pair = [detection("Eyeclosed", .8, (10, 10, 20, 20)),
            detection("Eyeclosed", .85, (31, 11, 41, 21))]
    assert infer_eye_state(pair) is True
    mixed = [pair[0], detection("Eyeopen", .9, (31, 11, 41, 21))]
    assert infer_eye_state(mixed) is None
    different_rows = [pair[0], detection("Eyeclosed", .99, (31, 80, 41, 90))]
    assert infer_eye_state(different_rows) is None


def test_overlapping_conflicting_predictions_are_deduplicated():
    items = [detection("Eyeclosed", .94, (10, 10, 20, 20)),
             detection("Eyeopen", .75, (11, 10, 21, 20))]
    assert infer_eye_state(items) is True


def test_single_open_eye_is_open_but_low_confidence_single_closed_is_unknown():
    assert infer_eye_state([detection("Eyeopen", .6, (1, 1, 10, 10))]) is False
    assert infer_eye_state([detection("Eyeclosed", .7, (1, 1, 10, 10))]) is None


def test_roi_filters_boxes_using_normalized_center():
    items = [detection("Eyeopen", .9, (10, 10, 20, 20)),
             detection("Yawn", .9, (80, 10, 90, 20))]
    assert keep_driver_region(items, (100, 100, 3), [0.0, 0.0, 0.5, 1.0]) == [items[0]]


def test_eye_pair_with_large_size_mismatch_is_unknown():
    pair = [detection("Eyeclosed", .9, (10, 10, 20, 20)),
            detection("Eyeclosed", .9, (31, 10, 56, 35))]
    assert infer_eye_state(pair, max_eye_size_ratio=2.0) is None


def test_driver_face_filter_requires_whole_detection_inside_face_region():
    items = [detection("Eyeopen", .9, (10, 10, 20, 20)),
             detection("Yawn", .9, (90, 10, 110, 30))]
    assert keep_detections_in_box(items, (0, 0, 100, 100)) == items


def test_weak_closed_eye_pair_is_unknown():
    pair = [detection("Eyeclosed", .26, (10, 10, 20, 20)),
            detection("Eyeclosed", .27, (31, 11, 41, 21))]
    assert infer_eye_state(pair) is None


def test_conflicting_eye_pair_is_unknown():
    pair = [detection("Eyeclosed", .74, (10, 10, 20, 20)),
            detection("Eyeopen", .44, (31, 11, 41, 21))]
    assert infer_eye_state(pair) is None


def test_event_match_rejects_tiny_overlap_and_accepts_half_overlap():
    truth = [(10.0, 20.0)]
    assert match_events([(19.5, 22.0)], truth, 0.5) == []
    matches = match_events([(15.0, 21.0)], truth, 0.5)
    assert len(matches) == 1
    assert matches[0][2] == 0.5
