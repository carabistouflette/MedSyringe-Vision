from scripts.runtime_bundle import environment_fingerprint


def test_package_fingerprint_detects_changed_weights_and_ignores_bytecode(tmp_path):
    packages = tmp_path/"lib/python3.12/site-packages"
    cached = packages/"example/__pycache__"
    cached.mkdir(parents=True)
    module = cached.parent/"model.py"
    module.write_text("version = 1\n")
    expected = environment_fingerprint(tmp_path)
    assert expected["files"] == 1
    (cached/"model.cpython-312.pyc").write_bytes(b"interpreter cache")
    assert environment_fingerprint(tmp_path) == expected
    module.write_text("version = 2\n")
    assert environment_fingerprint(tmp_path)["sha256"] != expected["sha256"]
