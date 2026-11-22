from medsyringe.field_status import describe_fields, report_fields
from scripts.evaluate_field_abstention import score


def test_a_counted_object_can_abstain_per_field_and_a_measured_zero_is_valid():
    obj = {"id": "item", "scale": None, "fill_level": {"status": "visual estimate", "ml": 0,
                                                       "interval_ml": [0, 1]}}
    result = describe_fields(obj)
    assert result["syringe_type"]["status"] == "indeterminate"
    assert result["liquid_volume"]["status"] == "estimated"
    assert result["liquid_volume"]["ml"] == 0
    reviewed = dict(obj, scale="20-scale", fill_level={"status": "insufficient evidence", "reason": "Piston obscured"})
    report = report_fields(dict(objects=[reviewed], original=dict(objects=[obj], accepted_count=1)))
    assert len(report["objects"]) == report["original"]["accepted_count"] == 1
    assert report["field_status"]["item"]["syringe_type"]["status"] == "supported"
    assert report["field_status"]["item"]["liquid_volume"] == {
        "status": "indeterminate", "ml": None, "interval_ml": None, "reason": "Piston obscured"}
    assert report["original_field_status"]["item"]["liquid_volume"]["ml"] == 0


def test_abstention_scores_ignore_unreviewed_rows_and_track_intervals_separately():
    field = "liquid_volume"
    examples = [
        ({"status": "observable", "ml": 0}, {"status": "estimated", "ml": 0}),
        ({"status": "observable", "ml": 5}, {"status": "interval only", "ml": None, "interval_ml": [4, 6]}),
        ({"status": "indeterminate"}, {"status": "estimated", "ml": 0}),
        ({"status": "unreviewed"}, {"status": "estimated", "ml": 15}),
    ]
    measured = score([dict(reference={field: r}, prediction={field: p}) for r, p in examples], field)
    assert measured["coverage"] == 1
    assert measured["point_coverage"] == measured["interval_only_coverage"] == .5
    assert measured["emission_on_indeterminate"] == 1
    assert measured["point_mae_ml"] == 0
    assert measured["interval_contains_exact"] == 1


def test_capacity_and_liquid_presence_are_scored_independently():
    observation = {"inscriptions": {"capacity": {"value": 5., "unit": "mL"}},
                   "fill_level": {"status": "visual estimate", "ml": 0., "interval_ml": [0., 1.]}}
    result = describe_fields(observation)
    assert result["nominal_capacity"]["ml"] == 5.
    assert result["liquid_presence"]["status"] == "indeterminate"
    assert result["liquid_volume"]["ml"] == 0.
    observation["liquid_presence_override"] = "empty"
    assert describe_fields(observation)["liquid_presence"]["value"] is False
    examples = [
        {"reference": {"liquid_presence": {"status": "observable", "value": False}},
         "prediction": {"liquid_presence": {"status": "empty", "value": False}}},
        {"reference": {"liquid_presence": {"status": "observable", "value": True}},
         "prediction": {"liquid_presence": {"status": "indeterminate"}}},
    ]
    outcomes = score(examples, "liquid_presence")
    assert outcomes["coverage"] == .5
    assert outcomes["by_reference"]["present"]["unresolved"] == 1
    assert outcomes["by_reference"]["empty"]["correct"] == 1
