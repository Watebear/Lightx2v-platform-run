#!/usr/bin/env python3
"""Verify the MLU590 config and shell-script bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if len(cases) != manifest["case_count"]:
        raise SystemExit("Case count does not match manifest")

    for case in cases:
        config = ROOT / case["config"]
        script = ROOT / case["script"]
        case_dir = config.parent
        if digest(config) != case["config_sha256"]:
            raise SystemExit(f"Config hash mismatch: {config}")
        if digest(script) != case["script_sha256"]:
            raise SystemExit(f"Shell hash mismatch: {script}")
        json.loads(config.read_text(encoding="utf-8"))
        if sorted(path.name for path in case_dir.iterdir()) != sorted(("config.json", script.name)):
            raise SystemExit(f"Expected only config.json and the run script in {case_dir}")
        subprocess.run(["bash", "-n", str(script)], check=True)

    print(f"Verified {len(cases)} MLU590 cases: configs, shell syntax, and hashes.")


if __name__ == "__main__":
    main()
