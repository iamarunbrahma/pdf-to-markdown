"""CLI config-file handling tests."""

from __future__ import annotations

import os

from pdf2md.cli import main

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _fx(name):
    path = os.path.join(FIXTURES, name)
    assert os.path.isfile(path), f"missing fixture {name}"
    return path


def test_missing_explicit_config_is_an_error(capsys):
    rc = main(
        [
            "--pdf_path",
            _fx("basic.pdf"),
            "--config",
            "does-not-exist.yaml",
            "--no-progress",
        ]
    )
    assert rc == 2
    assert "config file not found" in capsys.readouterr().err


def test_directory_as_config_is_an_error(tmp_path, capsys):
    rc = main(
        [
            "--pdf_path",
            _fx("basic.pdf"),
            "--config",
            str(tmp_path),
            "--no-progress",
        ]
    )
    assert rc == 2
    assert "config file not found" in capsys.readouterr().err
