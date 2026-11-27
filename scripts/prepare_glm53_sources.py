"""Check out the CANN 9.2 sources and apply their registered patch series."""

import subprocess
import sys
import tomllib
from pathlib import Path


def prepare_sources(manifest: Path, destination: Path) -> None:
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    patches = data["patches"]
    by_id = {patch["id"]: patch for patch in patches}
    for name, source in data["glm53"]["sources"].items():
        series = []
        if "patch" in source:
            anchor = by_id[source["patch"]]
            revision = anchor["base_commit"]
            series = [p for p in patches if p["target"] == anchor["target"] and p["base_commit"] == revision]
        else:
            revision = source["commit"]

        checkout = destination / name
        subprocess.run(["git", "clone", "--no-checkout", source["url"], checkout], check=True)
        subprocess.run(["git", "checkout", "--detach", revision], cwd=checkout, check=True)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        if head != revision:
            raise ValueError(f"{name}: expected {revision}, checked out {head}")
        for patch in series:
            path = manifest.parent / patch["category"] / patch["file"]
            subprocess.run(["git", "am", path], cwd=checkout, check=True)


if __name__ == "__main__":
    prepare_sources(Path(__file__).resolve().parents[1] / "patches/MANIFEST.toml", Path(sys.argv[1]))
