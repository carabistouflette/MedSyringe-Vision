"""Exercise the visible local interface with a real browser and a real image."""
import argparse
from contextlib import suppress
from hashlib import sha256
from io import BytesIO
import json
from math import ceil, floor
from pathlib import Path
import os
import signal
import socket
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import urllib.request
from zipfile import ZipFile

from PIL import Image
from playwright.sync_api import expect, sync_playwright

from medsyringe.data import write_json


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/browser_smoke.json"))
    parser.add_argument("--followup-control", action="store_true",
                        help="Check the known 20 mL and liquid evidence on IMG_3229.HEIC.")
    parser.add_argument("--expanded-markings-control", action="store_true",
                        help="Check the expanded reader and its export on IMG_3185.jpeg.")
    parser.add_argument("--graduation-control", action="store_true",
                        help="Check the stroke-guided capacity recovery on IMG_3294.jpeg.")
    parser.add_argument("--review-candidate", action="store_true",
                        help="Include a pending detection in a review and verify its provenance.")
    parser.add_argument("--native-evidence", action="store_true",
                        help="Verify the exact uploaded bytes and the native object crop.")
    parser.add_argument("--draft-native-crop", action="store_true",
                        help="Inspect the same native pixels before and after adopting an unsaved candidate.")
    parser.add_argument("--crowded-candidates", action="store_true",
                        help="Check that all five suggestions are immediately available on a multi-object scene.")
    parser.add_argument("--fill-control", action="store_true",
                        help="Check the piston overlay and export on a validated fill control image.")
    parser.add_argument("--manual-fill-control", action="store_true",
                        help="Check image landmarks and reviewed fill on IMG_3316.jpeg.")
    parser.add_argument("--server-python", default=sys.executable, help="Python interpreter for the local API.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                        help="Use the source and model files from a particular restored checkout.")
    parser.add_argument("--analysis-timeout", type=float, default=120,
                        help="Seconds to wait for real inference in this test.")
    parser.add_argument("--without-ocr", action="store_true",
                        help="Turn off visible marking analysis for a detection-only browser check.")
    args = parser.parse_args()
    if args.without_ocr and any((args.followup_control, args.expanded_markings_control,
                                 args.graduation_control, args.fill_control, args.manual_fill_control)):
        parser.error("Marking and fill controls require OCR.")
    if (args.draft_native_crop or args.crowded_candidates) and not args.review_candidate:
        parser.error("Candidate crop and crowded checks require the candidate review flow.")
    signal.signal(signal.SIGTERM, interrupted)
    project = args.project_root.resolve()
    screenshot_root = project/"data/workspace"
    screenshot_root.mkdir(parents=True,exist_ok=True)
    flows = ["image upload", "real detection overlay", "inventory", "persisted review",
             "ZIP report with evidence", "mobile viewport without horizontal overflow"]
    with TemporaryDirectory(prefix="medsyringe-browser-") as temporary:
        root = Path(temporary)
        (root / "data").mkdir()
        if (project/".venv").is_dir():
            (root / ".venv").symlink_to(project/".venv", target_is_directory=True)
        (root / "data/models").symlink_to(project/"data/models", target_is_directory=True)
        (root / "artifacts").symlink_to(project/"artifacts", target_is_directory=True)
        if (project/"artifacts/rfdetr_verified_release.json").exists():
            (root / "data/rfdetr").symlink_to(project/"data/rfdetr", target_is_directory=True)
        for feature in ("sam3","ocr-v6"):
            if (project/"data"/feature).exists():
                (root/"data"/feature).symlink_to(project/"data"/feature,target_is_directory=True)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        address = f"http://127.0.0.1:{port}/"
        server = subprocess.Popen(
            [args.server_python, "-m", "uvicorn", "medsyringe.app:app", "--port", str(port)],
            cwd=project,
            env={**os.environ, "MEDSYRINGE_ROOT": str(root), "PYTHONPATH": str(project)},
            stdout=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            deadline = monotonic()+25
            while True:
                try:
                    with urllib.request.urlopen(address, timeout=0.5):
                        break
                except OSError:
                    if monotonic() > deadline or server.poll() is not None:
                        raise
                    sleep(0.2)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    executable_path=os.environ.get("MEDSYRINGE_CHROMIUM") or shutil.which("chromium"),
                    args=["--no-sandbox"],
                )
                page = browser.new_page(viewport={"width": 1440, "height": 920}, accept_downloads=True)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(address)
                expect(page.get_by_role("heading", name="Scene inspection")).to_be_visible()
                if args.without_ocr:
                    page.get_by_role("checkbox", name="Read visible markings").uncheck()
                    flows.append("detection-only analysis selected in the upload form")
                page.locator("#file").set_input_files(args.image)
                page.get_by_role("button", name="Analyze", exact=True).click()
                expect(page.locator("#boxes button").first).to_be_visible(timeout=args.analysis_timeout*1000)
                count = page.locator("#boxes button").count()
                pixels = None
                if args.native_evidence:
                    link = page.get_by_role("link", name="Inspect native-resolution crop", exact=True).first
                    expect(link).to_be_visible()
                    native = page.request.get(address.rstrip("/") + link.get_attribute("href"))
                    assert native.ok and "image/png" in native.headers["content-type"]
                    with Image.open(BytesIO(native.body())) as crop_image:
                        pixels = crop_image.size
                revision_number = 1
                if args.expanded_markings_control:
                    expect(page.get_by_text("Nominal capacity · 20 mL", exact=True)).to_be_visible()
                if args.graduation_control:
                    expect(page.get_by_text("Nominal capacity · 20 mL", exact=True)).to_be_visible()
                if args.fill_control:
                    summary = page.get_by_text("Estimated fill · ≈", exact=False)
                    expect(summary).to_be_visible()
                    summary.click()
                    expect(page.get_by_text("Visual range ·", exact=False)).to_be_visible()
                    page.get_by_role("button", name="Show piston and graduations").click()
                    expect(page.locator('#boxes [data-label="Piston edge"]')).to_be_visible()
                    expect(page.locator('#boxes [data-label$="mL mark"]').first).to_be_visible()
                    page.locator("#reviewer").fill("Browser check")
                    page.locator("#note").fill("Reviewed the estimate without changing the scale or box")
                    page.get_by_role("button", name="Save review").click()
                    expect(page.locator("#revision")).to_contain_text("Revision 1")
                    expect(page.get_by_text("Estimated fill · ≈", exact=False)).to_be_visible()
                    revision_number = 2
                if args.manual_fill_control:
                    base = page.locator("#source").get_attribute("src").removesuffix("/image")
                    endpoint = address.rstrip("/") + base
                    first = page.request.get(endpoint).json()
                    original = first["original"]["objects"][0]
                    fill = original["fill_level"]
                    assert fill["ml"] == 16
                    low = max((t for t in fill["graduations"] if t["ml"] < 16), key=lambda t: t["ml"])
                    high = min((t for t in fill["graduations"] if t["ml"] > 16), key=lambda t: t["ml"])
                    page.get_by_role("button", name="Mark fill on image").first.click()
                    page.locator("#source").scroll_into_view_if_needed()
                    overlay = page.locator("#boxes").bounding_box()
                    left, top, right, bottom = original["bbox"]
                    for x, y in ([(m["crop_bbox"][0]+m["crop_bbox"][2])/2,
                                  (m["crop_bbox"][1]+m["crop_bbox"][3])/2] for m in (low, high)):
                        page.mouse.click(overlay["x"]+(left+(right-left)*x)*overlay["width"],
                                         overlay["y"]+(top+(bottom-top)*y)*overlay["height"])
                    px, py = fill["piston_line"][0]
                    page.mouse.click(overlay["x"]+(left+(right-left)*px)*overlay["width"],
                                     overlay["y"]+(top+(bottom-top)*py)*overlay["height"])
                    page.get_by_label("Lower (mL)").fill(str(low["ml"]))
                    page.get_by_label("Upper (mL)").fill(str(high["ml"]))
                    page.get_by_role("button", name="Apply marks").click()
                    expect(page.get_by_text("Fill landmarks ready to save.")).to_be_visible()
                    page.locator("#reviewer").fill("Browser check")
                    page.locator("#note").fill("Selected image landmarks for fill")
                    page.get_by_role("button", name="Save review").click()
                    revision_number += 1
                    expect(page.locator("#revision")).to_contain_text(f"Revision {revision_number-1}")
                    summary = page.get_by_text("Reviewed fill · ≈", exact=False)
                    expect(summary).to_be_visible()
                    summary.click()
                    page.get_by_role("button", name="Show reviewed landmarks").click()
                    expect(page.locator(".fill-point")).to_have_count(3)
                    with page.expect_download() as reviewed_download:
                        page.get_by_role("button", name="Export report").click()
                    with ZipFile(reviewed_download.value.path()) as archive:
                        reviewed = json.loads(archive.read("report.json"))
                    measured = reviewed["objects"][0]["fill_level"]
                    assert abs(measured["ml"]-16) <= 1.5
                    assert measured["source"] == "human" and len(measured["landmarks"]["piston"]) == 2
                    assert reviewed["original"]["objects"][0]["fill_level"] == fill
                    flows.extend(["three image landmarks", "server-side reviewed fill",
                                  "manual evidence and original model estimate in ZIP"])
                    if Path("artifacts/expanded/crop_verifier_release.json").exists():
                        assert "crop_verification_sha256" in reviewed["original"]["model"]
                        flows.append("released crop verifier on real inference")
                    if os.environ.get("MEDSYRINGE_PRIMARY_DEVICE") == "xpu":
                        assert reviewed["original"]["model"]["primary_backend"] == "xpu"
                        flows.append("real Intel XPU primary grounder")
                if args.followup_control:
                    expect(page.get_by_text("Nominal capacity · 20 mL", exact=True)).to_be_visible()
                    page.get_by_text("Liquid lightness · 48.2 L* · medium", exact=True).click()
                    expect(page.get_by_text("A registered clear-barrel reference", exact=False)).to_be_visible()
                    page.get_by_role("button", name="Show measured region").click()
                    expect(page.locator('#boxes [data-label="Measured liquid region"]')).to_be_visible()
                page.get_by_role("button", name="Compare inventory").click()
                expect(page.locator("#results")).to_contain_text(f"Extra {count}")
                field = page.locator(".object select").first
                corrected = "60-scale" if field.input_value() == "20-scale" else "20-scale"
                field.select_option(corrected)
                page.locator("#reviewer").fill("Browser check")
                page.locator("#note").fill("Verified the scale marking in the source image")
                page.get_by_role("button", name="Save review").click()
                expect(page.locator("#revision")).to_contain_text(f"Revision {revision_number}")
                page.get_by_role("button", name="Compare inventory").click()
                expect(page.locator("#results")).to_contain_text(f"Extra {count}")
                with page.expect_download() as download:
                    page.get_by_role("button", name="Export report").click()
                assert download.value.suggested_filename.endswith(".zip")
                with ZipFile(download.value.path()) as report_zip:
                    report = json.loads(report_zip.read("report.json"))
                    if args.native_evidence:
                        original = report_zip.read(report["original_image"])
                        source = report["original"]
                        assert original == args.image.read_bytes()
                        assert sha256(original).hexdigest() == source["source_sha256"]
                        w, h = source["dimensions"]
                        l, t, r, b = source["objects"][0]["bbox"]
                        assert pixels == (ceil(r*w)-floor(l*w), ceil(b*h)-floor(t*h))
                        assert source["detection_checks"]["status"] == "available"
                        flows.append("verified native-resolution crop and exact original bytes in export")
                    assert report["revision"] == revision_number
                    assert report["original"]["objects"][0]["origin"] == "model"
                    assert report["objects"][0]["scale"] == corrected
                    assert report["objects"][0]["origin"] == "human"
                    assert report["inventory_check"]["extra"]["syringe"] == count
                    assert report_zip.read("observation.jpg").startswith(b"\xff\xd8")
                    if args.expanded_markings_control:
                        assert "expanded_sha256" in report["original"]["model"]
                        assert report["objects"][0]["inscriptions"]["capacity"]["value"] == 20
                        flows.append("expanded marking release and preserved 20 mL evidence")
                    if args.graduation_control:
                        source = report["original"]
                        observed = source["objects"][0]
                        assert "graduation_capacity_sha256" in source["model"]
                        assert observed["inscriptions"]["capacity"]["value"] == 20
                        assert observed["inscriptions"]["graduation_crop_evidence"]
                        assert observed["fill_level"]["status"] == "insufficient evidence"
                        flows.append("graduation OCR recovered 20 mL and preserved apparent-fill abstention")
                    if args.fill_control:
                        original = report["original"]["objects"][0]
                        assert original["fill_level"]["piston_line"] and original["fill_level"]["graduations"]
                        assert "fill_level" not in report["objects"][0]
                        flows.extend(["fill percentage and visual range", "piston and graduation overlays",
                                      "fill retained after an unchanged box and scale review",
                                      "original fill retained in export after scale correction"])
                    if args.followup_control:
                        reviewed = report["objects"][0]
                        assert reviewed["inscriptions"]["capacity"]["value"] == 20
                        appearance = reviewed["liquid_appearance"]
                        assert abs(appearance["lightness"]["lstar"] - 48.2) <= 0.2
                        assert appearance["region"] and appearance["opacity"] is None
                        flows.extend(["nominal capacity", "liquid region overlay", "opacity abstention",
                                      "preserved markings and liquid evidence in reviewed export"])
                screenshot = args.output.stem.replace("_smoke", "")
                if args.review_candidate:
                    page.locator("#uncertain > summary").click()
                    more = page.locator("#candidates > details > summary")
                    if args.crowded_candidates:
                        assert count >= 2 and len(report["original"]["below_threshold_candidates"]) == 5
                        expect(page.locator("#candidates > .candidate-review")).to_have_count(5)
                        expect(page.get_by_role("button", name="Inspect candidate 5 ·", exact=False)).to_be_visible()
                        flows.append("all five proposals shown immediately when there are several automatic detections")
                    if more.count():
                        expect(page.locator("#candidates > .candidate-review")).to_have_count(2)
                        more.click()
                        expect(page.get_by_role("button",name="Inspect candidate 3 ·",exact=False).first).to_be_visible()
                        flows.append("two initial proposals and access to all remaining candidates")
                    page.get_by_role("button", name="Inspect candidate 1 ·", exact=False).first.click()
                    expect(page.locator('#boxes [data-label="Review candidate"]')).to_be_visible()
                    if args.draft_native_crop:
                        link = page.get_by_role("link", name="Inspect candidate at native resolution", exact=True)
                        expect(link).to_be_visible()
                        native = page.request.get(address.rstrip("/") + link.get_attribute("href"))
                        assert native.ok and "image/png" in native.headers["content-type"]
                    page.get_by_role("button", name="Add candidate 1 to review", exact=True).click()
                    expect(page.locator("#boxes button")).to_have_count(count+1)
                    if args.crowded_candidates:
                        expect(page.locator("#total")).to_have_text(str(count+1))
                        expect(page.locator("#summary")).to_contain_text(f"{count} automatically detected")
                        flows.append("automatic and draft counts stay distinguishable during review")
                    if args.draft_native_crop:
                        unsaved = page.locator(".object.active").get_by_role(
                            "link", name="Inspect native-resolution crop", exact=True)
                        expect(unsaved).to_be_visible()
                        crop = page.request.get(address.rstrip("/") + unsaved.get_attribute("href"))
                        assert crop.ok and crop.body() == native.body()
                        flows.append("matching native crop before candidate adoption and before saving")
                    page.locator("#reviewer").fill("Browser check")
                    page.locator("#note").fill("Checked candidate review provenance in a temporary workspace")
                    page.get_by_role("button", name="Save review").click()
                    expect(page.locator("#revision")).to_contain_text(f"Revision {revision_number+1}")
                    with page.expect_download() as candidate_download:
                        page.get_by_role("button", name="Export report").click()
                    with ZipFile(candidate_download.value.path()) as archive:
                        adopted_report = json.loads(archive.read("report.json"))
                    original = adopted_report["original"]
                    adopted = next(o for o in adopted_report["objects"] if o.get("candidate_index")==0)
                    assert adopted["bbox"] == original["below_threshold_candidates"][0]["bbox"]
                    assert adopted["origin"] == "human" and adopted["score"] is None
                    assert len(original["objects"]) == count
                    flows.append("candidate adoption with unchanged model output and linked review evidence")
                page.screenshot(path=screenshot_root/f"{screenshot}-desktop.png", full_page=True)
                page.set_viewport_size({"width": 390, "height": 850})
                assert page.evaluate("document.documentElement.scrollWidth") <= 390
                if args.manual_fill_control:
                    page.get_by_role("button", name="Mark fill on image").first.click()
                    assert page.evaluate("document.documentElement.scrollWidth") <= 390
                    page.get_by_role("button", name="Cancel", exact=True).click()
                    flows.append("fill editor at a 390 pixel viewport")
                page.screenshot(path=screenshot_root/f"{screenshot}-mobile.png", full_page=True)
                assert not errors, errors
                browser.close()
            write_json(args.output, dict(
                image=args.image.name, detected=count, browser="Chromium", no_page_errors=True,
                revision=subprocess.check_output(["git","-C",str(project),"rev-parse","HEAD"],text=True).strip(),
                installation_backend=report["original"]["model"].get("primary_backend","cpu"),
                appearance_release_sha256=report["original"]["model"].get("appearance_detection_sha256"),
                detection_checks=report["original"].get("detection_checks"),
                flows=flows))
        finally:
            if server.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(server.pid,signal.SIGKILL)
                server.wait(timeout=5)


if __name__ == "__main__":
    main()
