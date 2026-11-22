from scripts.experiments.public_data.public_catalog_exposure_audit import project_links, universe_project


def test_project_alias_blocks_related_exports_across_public_projects():
    protected = {"same-team/first-export"}
    assert project_links({
        "source_url": "https://universe.roboflow.com/same-team/second-export",
        "group": "roboflow:unrelated-example",
        "candidate_group": "roboflow:same-team/first-export",
    }, protected) == ["same-team/first-export"]
    assert project_links({
        "source_url": "https://universe.roboflow.com/other-team/independent-export",
        "group": "roboflow:other-team/independent-export",
    }, protected) == []


def test_project_id_is_independent_of_version_path_and_query():
    assert universe_project("https://universe.roboflow.com/Team/Project/dataset/2?id=picture") == "team/project"
    assert universe_project("https://example.org/Team/Project") is None
