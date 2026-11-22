from PIL import Image

from medsyringe.chamber_appearance import inspect_chamber
from scripts import low_count_review
from scripts.small_field_study import render


def test_recovery_reserves_one_peer_and_preserves_the_original(monkeypatch):
    existing = [{"bbox": [.1, .22, .3, .9], "score": .80}]
    duplicate = {"bbox": [.14, .14, .36, .92], "score": .95}
    separate = {"bbox": [.48, .15, .63, .9], "score": .93}
    monkeypatch.setattr(low_count_review, "disjoint", lambda raw: [duplicate, separate])
    monkeypatch.setattr(low_count_review, "verification_score", lambda head, p: 1.)
    head = {"floor": .8}
    suggestions = low_count_review.propose(existing, {}, head)
    assert [p["bbox"] for p in suggestions] == [separate["bbox"]]
    assert len(existing) == 1
    assert low_count_review.propose(existing * 2, {}, head) == []


def test_color_analysis_with_covered_markings_does_not_claim_numeric_fields():
    pixels, _, _ = render(2.7, (49, 49, 51), ((25, 181, 230), .43), unreadable=True)
    result = inspect_chamber(Image.fromarray(pixels))
    assert result["presence"] == "present"
    assert result["appearance"]["apparent_color"]["family"] == "cyan"
    assert result["fill_ml"] is None and result["opacity"] is None


def test_blue_stopper_at_zero_does_not_become_a_colored_column():
    pixels, _, _ = render(0., (42, 86, 155), None)
    result = inspect_chamber(Image.fromarray(pixels))
    assert result["presence"] == "unknown"
    assert result["appearance"] is None
