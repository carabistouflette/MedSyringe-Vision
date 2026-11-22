"""Run real local inference and round-trip the API without touching the workspace database."""
import argparse
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/app_smoke.json"))
    args = parser.parse_args()
    with TemporaryDirectory(prefix="medsyringe-smoke-") as temporary:
        root = Path(temporary)
        os.environ["MEDSYRINGE_ROOT"] = temporary
        from medsyringe.app import create_app
        from medsyringe.data import digest, valid_box, write_json

        (root / "data").mkdir(exist_ok=True)
        (root / "data/models").symlink_to(Path("data/models").resolve(), target_is_directory=True)
        (root / "artifacts").symlink_to(Path("artifacts").resolve(), target_is_directory=True)
        if Path("artifacts/rfdetr_verified_release.json").exists():
            (root / "data/rfdetr").symlink_to(Path("data/rfdetr").resolve(), target_is_directory=True)
        for feature in ("sam3","ocr-v6"):
            if Path("data",feature).exists():
                (root/"data"/feature).symlink_to(Path("data",feature).resolve(),target_is_directory=True)
        client = TestClient(create_app(root))
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        with args.image.open("rb") as source:
            response = client.post("/api/observations", files={"file": (args.image.name, source)},
                                   data={"ocr": "true"})
        response.raise_for_status()
        report = response.json()
        assert report["objects"]
        assert report["original"]["source_sha256"] == digest(args.image)
        assert all(valid_box(o["bbox"]) and o["origin"] == "model" for o in report["objects"])
        route = f"/api/observations/{report['id']}"
        evidence = client.get(route + "/image")
        assert evidence.headers["content-type"] == "image/jpeg"
        with Image.open(args.image) as source, Image.open(io.BytesIO(evidence.content)) as converted:
            assert source.info.get("icc_profile") == converted.info.get("icc_profile")
        objects = report["objects"]
        review = client.post(route + "/review", json={
            "revision": 0, "reviewer": "Smoke test", "note": "Exercised persisted review flow",
            "status": "needs review", "objects": objects,
        })
        review.raise_for_status()
        assert review.json()["original"]["objects"] == objects
        check = client.post(route + "/inventory", json={"field": "count", "counts": {"syringe": 0}})
        assert check.json()["extra"]["syringe"] == len(objects)
        draft = client.post("/api/library/IMG_3198")
        # A detached inference-only workspace can run without private prepared photos.
        assert draft.status_code == 409
        comparison = client.get(f"{route}/compare/{report['id']}")
        assert comparison.json()["identity_status"] == "unresolved"
        write_json(args.output, {
            "image": args.image.name, "model": report["original"]["model"],
            "source_sha256": report["original"]["source_sha256"],
            "latency_ms_including_ocr": report["original"]["elapsed_ms"],
            "working_routes": ["home", "assets", "upload/inference", "evidence image",
                                "persisted review", "inventory", "observation comparison"],
            "predicted_objects": len(objects), "objects": objects,
            "embedded_color_profile_preserved": True,
            "inventory": check.json(), "review_revision": review.json()["revision"],
        })
        print("Real inference, review, inventory and comparison passed.")


if __name__ == "__main__":
    main()
