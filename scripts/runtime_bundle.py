"""Package and restore the released local models and worker environments."""
import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

ENVIRONMENTS = (".venv", "data/sam3/xpu-env", "data/ocr-v6/env")
MANIFEST = "runtime-bundle.json"
PYTHON = Path("data/runtime-python/cpython-3.12")
LOCKS = ("artifacts/capacity_ocr_release.json", "artifacts/liquid_appearance_release.json",
         "artifacts/expanded/crop_verifier_release.json", "artifacts/fill/graduation_capacity_release.json")
APPEARANCE = "artifacts/detection_hillclimb/appearance_release.json"
EXACT_RELEASE = "artifacts/freeze.json"
RECOVERY = "artifacts/rfdetr_verified_release.json"
SAM_MODEL = "artifacts/sam3_checkpoint.json"
PACKAGES = "import importlib.metadata as m,json;print(json.dumps(sorted((x.metadata['Name'],x.version) for x in m.distributions())))"


def read(path):
    return json.loads(path.read_text())


def filehash(path):
    with path.open("rb") as payload:
        return file_digest(payload,"sha256").hexdigest()


def confined(path):
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid bundle path {path!r}.")
    return relative


def pinned_assets(root):
    frozen = read(root/EXACT_RELEASE)
    base = {str(Path("data/models")/frozen["backbone"]/file):value for file,value in frozen["bundle_sha256"].items()}
    recovery = read(root/RECOVERY)
    base[recovery["weights"]] = recovery["weights_sha256"]
    base["data/models/verifier/mobilenet_v3_small-047dcff4.pth"] = recovery["encoder_sha256"]
    base.update({str(Path("data/sam3/facebook-sam3")/name):value for name,value in read(root/SAM_MODEL)["files"].items()})
    locks = (*LOCKS, APPEARANCE) if (root/APPEARANCE).is_file() else LOCKS
    for path in locks:
        for file,value in read(root/path).get("models",{}).items():
            if file.startswith("data/"):
                if file in base and base[file]!=value:
                    raise ValueError(f"Release manifests disagree on {file}.")
                base[file] = value
    return base


def copy_bytes(origin, destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(["cp","--archive","--reflink=auto","--",str(origin),str(destination)],check=True)


def git_revision(root):
    return subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()


def package_list(environment):
    output = subprocess.check_output([str(environment/"bin/python"),"-c",PACKAGES],text=True)
    return json.loads(output)


def pack(root, destination):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    tracked_changes = subprocess.check_output(["git","-C",str(root),"status","--porcelain","--untracked-files=normal"],text=True)
    if tracked_changes.strip():
        raise ValueError("Record the project files in Git before packaging the exact release.")
    assets = pinned_assets(root)
    actual = {file:filehash(root/file) for file in assets}
    if actual!=assets:
        raise ValueError("One or more released weight files failed their checksums.")
    base_prefixes = {Path(subprocess.check_output([str(root/env/"bin/python"),"-c","import sys;print(sys.base_prefix)"],
                                                   text=True).strip()).resolve() for env in ENVIRONMENTS}
    if len(base_prefixes)!=1:
        raise ValueError("An offline package requires the same base Python for the app and both workers.")
    python = next(iter(base_prefixes))
    destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix="medsyringe-runtime-",dir=destination.parent) as temporary:
        stage = Path(temporary)/"bundle"
        stage.mkdir()
        payload = stage/"payload"
        for file in assets:
            copy_bytes(root/file,payload/file)
        for env in ENVIRONMENTS:
            copy_bytes(root/env,payload/env)
        copy_bytes(python,payload/PYTHON)
        packages = {env:package_list(root/env) for env in ENVIRONMENTS}
        summary = dict(revision=git_revision(root), assets=assets, environments=packages,
                       python=str(PYTHON), platform=os.uname().machine,
                       asset_bytes=sum((root/f).stat().st_size for f in assets),
                       worker_files="Linux x86-64 environment snapshots. Relink the packaged Python on restore.")
        (stage/MANIFEST).write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
        stage.replace(destination)
    print(f"Packaged {len(assets)} checked files and {len(ENVIRONMENTS)} environments at {destination}.",flush=True)


def place_python(root, environment, python):
    target = root/environment
    bin_dir = target/"bin"
    interior = root/python/"bin/python3.12"
    for name in ("python", "python3", "python3.12"):
        entry = bin_dir/name
        if entry.exists() or entry.is_symlink():
            entry.unlink()
        entry.symlink_to(os.path.relpath(interior,bin_dir))
    config = target/"pyvenv.cfg"
    lines = [line for line in config.read_text().splitlines() if not line.startswith("home = ")]
    config.write_text(f"home = {interior.parent}\n" + "\n".join(lines) + "\n")
    for executable in bin_dir.iterdir():
        if executable.is_symlink() or not executable.is_file():
            continue
        with executable.open("rb") as script:
            line = script.readline(512)
            if not line.startswith(b"#!/") or b"/bin/python" not in line:
                continue
            tail = script.read()
        executable.write_bytes(f"#!{bin_dir/'python'}\n".encode()+tail)


def verify(root, manifest):
    if manifest["platform"]!=os.uname().machine or git_revision(root)!=manifest["revision"]:
        raise ValueError("The pinned Git revision or operating-system architecture differs from the runtime bundle.")
    if manifest["assets"]!=pinned_assets(root):
        raise ValueError("The restored assets do not match this release's model files.")
    for file,hash_ in manifest["assets"].items():
        if filehash(root/confined(file))!=hash_:
            raise ValueError(f"Runtime asset failed verification: {file}.")
    for environment,packages in manifest["environments"].items():
        if package_list(root/confined(environment))!=packages:
            raise ValueError(f"Runtime dependency set differs: {environment}.")
    print("Runtime weights and environment versions match the recorded release.",flush=True)


def restore(root, bundle):
    manifest = read(bundle/MANIFEST)
    if git_revision(root)!=manifest["revision"] or manifest["platform"]!=os.uname().machine:
        raise ValueError("Use a checkout at the recorded release revision and architecture.")
    if manifest["assets"]!=pinned_assets(root) or tuple(manifest["environments"].keys())!=tuple(sorted(ENVIRONMENTS)):
        raise ValueError("The bundle does not declare this release's model and environment paths.")
    sources = [*(confined(p) for p in manifest["assets"]),*(confined(p) for p in ENVIRONMENTS),confined(manifest["python"])]
    collisions = [str(path) for path in sources if (root/path).exists() or (root/path).is_symlink()]
    if collisions:
        raise FileExistsError(f"Restore into a clean checkout. These paths already exist: {', '.join(collisions[:3])}.")
    for file,hash_ in manifest["assets"].items():
        if filehash(bundle/"payload"/confined(file))!=hash_:
            raise ValueError(f"Packaged model failed verification: {file}.")
    for relative in sources:
        copy_bytes(bundle/"payload"/relative,root/relative)
    for env in ENVIRONMENTS:
        place_python(root,env,manifest["python"])
    verify(root,manifest)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("pack","restore","verify"))
    parser.add_argument("bundle",type=Path)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    args = parser.parse_args()
    current = args.root.resolve()
    if args.action=="verify":
        verify(current,read(args.bundle/MANIFEST))
    else:
        (pack if args.action=="pack" else restore)(current,args.bundle.resolve())
