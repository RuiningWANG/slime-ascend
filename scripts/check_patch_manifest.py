#!/usr/bin/env python3
"""Verify every registered patch before installation or submodule reset."""

from __future__ import annotations

import hashlib
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "patches" / "MANIFEST.toml"


def main() -> int:
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    problems: list[str] = []
    patch_ids = set()
    for item in data["patches"]:
        if item["id"] in patch_ids:
            problems.append(f"duplicate patch ID: {item['id']}")
        patch_ids.add(item["id"])
        if not re.fullmatch(r"[0-9a-f]{40}", item["base_commit"]):
            problems.append(f"{item['id']}: expected a full base commit")
        path = REPO_ROOT / "patches" / item["category"] / item["file"]
        if not path.is_file():
            problems.append(f"missing: {path.relative_to(REPO_ROOT)}")
            continue
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != item["sha256"]:
            problems.append(f"sha256 mismatch: {path.relative_to(REPO_ROOT)}")
        header = content.partition(b"\ndiff --git ")[0]
        base = re.search(rb"^Base-commit:[ \t]*(\S+)", header, re.MULTILINE | re.IGNORECASE)
        if base and base[1] != item["base_commit"].encode():
            problems.append(f"Base-commit mismatch: {path.relative_to(REPO_ROOT)}")
    for name, source in data["glm53"]["sources"].items():
        if "patch" in source:
            if "commit" in source or source["patch"] not in patch_ids:
                problems.append(f"{name}: expected a registered patch ID and no separate commit")
        elif not re.fullmatch(r"[0-9a-f]{40}", source.get("commit", "")):
            problems.append(f"{name}: expected a full upstream commit")
    # Build entry points must never reference a patch absent from the checkout.
    for entrypoint in (
        REPO_ROOT / "docker/Dockerfile.a3.ubuntu22.04.cann910.latest",
        REPO_ROOT / "scripts/quick_install.sh",
    ):
        for reference in set(re.findall(r"patches/third_party/[A-Za-z0-9_./-]+[.]patch", entrypoint.read_text())):
            if not (REPO_ROOT / reference).is_file():
                problems.append(f"missing build patch: {reference} ({entrypoint.relative_to(REPO_ROOT)})")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"Verified {len(data['patches'])} registered patches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
