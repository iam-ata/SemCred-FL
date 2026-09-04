"""Fail closed when the GitHub release candidate contains unsafe or inconsistent files."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_REPOSITORY_FILE_BYTES = 50 * 1024 * 1024
EXPECTED_VERSION = "0.1.0"
EXPECTED_AUTHORS = ("Nahideh Derakhshanfard", "Ata Mohammadi")
REQUIRED_FILES = (
    ".gitattributes",
    ".github/dependabot.yml",
    ".github/workflows/ci.yml",
    ".gitignore",
    ".zenodo.json",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "environment.yml",
    "pyproject.toml",
)
FORBIDDEN_TOP_LEVEL = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".staging",
        "artifacts",
        "Baselines PDF",
        "Datasets",
        "paper",
        "tmp",
    }
)
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".joblib",
        ".npy",
        ".npz",
        ".parquet",
        ".pdf",
        ".pt",
        ".pth",
        ".rar",
        ".sqlite3",
        ".zip",
    }
)
SECRET_PATTERNS = {
    "OpenAI API key": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def _git_candidate_files(root: Path) -> tuple[Path, ...]:
    """Return tracked and untracked non-ignored files exactly as Git sees them."""

    completed = subprocess.run(
        ("git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"),
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git candidate discovery failed: {message}")
    names = tuple(name for name in completed.stdout.decode("utf-8").split("\0") if name)
    files: list[Path] = []
    for name in names:
        candidate = (root / name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"Git candidate escapes the repository: {name}") from error
        if candidate.is_file():
            files.append(candidate)
    return tuple(sorted(files))


def _metadata_errors(root: Path) -> list[str]:
    """Return cross-format version and creator inconsistencies."""

    errors: list[str] = []
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    zenodo = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))

    versions = {
        "pyproject.toml": str(project.get("version")),
        "CITATION.cff": str(citation.get("version")),
        ".zenodo.json": str(zenodo.get("version")),
    }
    for source, version in versions.items():
        if version != EXPECTED_VERSION:
            errors.append(f"{source} version is {version}, expected {EXPECTED_VERSION}")

    project_authors = tuple(item.get("name") for item in project.get("authors", ()))
    citation_authors = tuple(
        f"{item.get('given-names')} {item.get('family-names')}"
        for item in citation.get("authors", ())
    )
    zenodo_authors = tuple(
        " ".join(reversed(item.get("name", "").split(", ", maxsplit=1)))
        for item in zenodo.get("creators", ())
    )
    for source, authors in {
        "pyproject.toml": project_authors,
        "CITATION.cff": citation_authors,
        ".zenodo.json": zenodo_authors,
    }.items():
        if authors != EXPECTED_AUTHORS:
            errors.append(f"{source} authors are inconsistent: {authors!r}")

    title = citation.get("title")
    if not isinstance(title, str) or zenodo.get("title") != title:
        errors.append("CITATION.cff and .zenodo.json titles differ")
    if citation.get("cff-version") != "1.2.0" or citation.get("type") != "software":
        errors.append("CITATION.cff schema or resource type is invalid")
    if zenodo.get("upload_type") != "software" or zenodo.get("access_right") != "open":
        errors.append("Zenodo software type or access right is invalid")
    return errors


def audit_public_release(root: Path = PROJECT_ROOT) -> tuple[Path, ...]:
    """Validate the complete GitHub candidate set and return it on success.

    Raises:
        RuntimeError: If Git discovery fails or any release gate is violated.
    """

    root = root.resolve()
    errors: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    errors.extend(f"required release file is missing: {name}" for name in missing)

    candidates = _git_candidate_files(root)
    relative_candidates = tuple(path.relative_to(root) for path in candidates)
    for relative, path in zip(relative_candidates, candidates, strict=True):
        if relative.parts and relative.parts[0] in FORBIDDEN_TOP_LEVEL:
            errors.append(f"forbidden release path is visible to Git: {relative.as_posix()}")
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary/data format is visible to Git: {relative.as_posix()}")
        size = path.stat().st_size
        if size > MAX_REPOSITORY_FILE_BYTES:
            errors.append(
                f"repository file exceeds the 50 MiB release threshold: "
                f"{relative.as_posix()} ({size} bytes)"
            )
        if size <= 2 * 1024 * 1024:
            content = path.read_bytes()
            if b"\x00" not in content:
                for label, pattern in SECRET_PATTERNS.items():
                    if pattern.search(content):
                        errors.append(f"possible {label} in Git candidate: {relative.as_posix()}")

    errors.extend(_metadata_errors(root))
    if errors:
        detail = "\n".join(f"- {error}" for error in sorted(set(errors)))
        raise RuntimeError(f"Public release audit failed:\n{detail}")
    return candidates


def main() -> int:
    """Run the public-release audit and print a compact status block."""

    try:
        candidates = audit_public_release()
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as error:
        print("status=blocked")
        print(str(error))
        return 2
    total_bytes = sum(path.stat().st_size for path in candidates)
    print("status=ready")
    print(f"version={EXPECTED_VERSION}")
    print(f"candidate_files={len(candidates)}")
    print(f"candidate_bytes={total_bytes}")
    print("datasets_included=false")
    print("generated_artifacts_included=false")
    print("manuscript_included=false")
    print("high_confidence_secrets_detected=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
