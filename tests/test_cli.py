"""Tests for the configuration-inspection CLI."""

from __future__ import annotations

import pytest

from semcred_fl.cli import main


def test_show_default(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show-default"]) == 0
    output = capsys.readouterr().out
    assert '"phase":"pilot"' in output
    assert "run_id=" in output


def test_validate_default(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate-default"]) == 0
    assert capsys.readouterr().out.startswith("valid run_id=")
