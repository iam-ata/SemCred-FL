from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_AUTHORS = ("Nahideh Derakhshanfard", "Ata Mohammadi")


def test_release_metadata_is_consistent() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    citation = yaml.safe_load((PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    zenodo = json.loads((PROJECT_ROOT / ".zenodo.json").read_text(encoding="utf-8"))

    assert project["version"] == citation["version"] == zenodo["version"] == "0.1.0"
    assert tuple(item["name"] for item in project["authors"]) == EXPECTED_AUTHORS
    assert (
        tuple(f"{item['given-names']} {item['family-names']}" for item in citation["authors"])
        == EXPECTED_AUTHORS
    )
    assert (
        tuple(
            " ".join(reversed(item["name"].split(", ", maxsplit=1))) for item in zenodo["creators"]
        )
        == EXPECTED_AUTHORS
    )
    assert citation["title"] == zenodo["title"]
    assert zenodo["upload_type"] == "software"


def test_public_release_audit_passes() -> None:
    completed = subprocess.run(
        (sys.executable, "scripts/audit_public_release.py"),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "status=ready" in completed.stdout
    assert "datasets_included=false" in completed.stdout
    assert "manuscript_included=false" in completed.stdout
