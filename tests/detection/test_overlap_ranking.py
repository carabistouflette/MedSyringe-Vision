from scripts.overlap_ranking import conflict, select
from scripts.overlap_ranking_review import pair_label, tags


def fixture(box, score):
    return dict(bbox=box, score=score, point=score, rank=score)


def rule(kind="learned", joint=False):
    return dict(rank_score="point", ranker="point", conflicts=kind, distinct=.8, joint=joint, floor=.50)


def test_pair_training_distinguishes_fragments_and_neighbors():
    refs = [{"bbox": [.05, .10, .44, .45]}, {"bbox": [.32, .20, .82, .66]}]
    first = tags(refs[0]["bbox"], refs)
    second = tags(refs[1]["bbox"], refs)
    fragment = tags([.12, .11, .32, .31], refs)
    background = tags([.8,.75,.95,.88], refs)
    assert pair_label(first, second) == "different"
    assert pair_label(first, fragment) == "same"
    assert pair_label(first, background) is None


def test_unknown_pair_falls_back_to_overlap_and_near_identical_pairs_stay_blocked():
    a = [.1,.1,.7,.7]
    b = [.28,.1,.88,.7]
    assert conflict(a, b, None, rule())
    assert not conflict(a, b, .95, rule())
    assert conflict(a, a, .999, rule())


def test_learned_pair_can_block_same_instance_across_lower_overlap():
    a = [.10,.10,.35,.6]
    b = [.25,.1,.60,.75]
    assert not conflict(a, b, .03, rule("overlap"))
    assert conflict(a, b, .03, rule())


def test_joint_selection_recovers_two_compatible_candidates_under_a_broad_box():
    first = fixture([.1,.1,.42,.6], .80)
    second = fixture([.58,.1,.90,.6], .80)
    broad = fixture([.08,.09,.93,.65], .99)
    candidates = [broad, first, second]
    greedy = select([], candidates, {}, rule("overlap"))
    joint = select([], candidates, {}, rule("overlap", joint=True))
    assert [r["bbox"] for r in greedy] == [broad["bbox"]]
    assert [r["bbox"] for r in joint] == [first["bbox"],second["bbox"]]


def test_a_new_candidate_cannot_override_an_incumbent():
    current = fixture([.1,.1,.42,.6], .80)
    better_scored_duplicate = fixture([.1,.1,.42,.6], .99)
    assert not select([current], [better_scored_duplicate], {(0,1):1.}, rule())
