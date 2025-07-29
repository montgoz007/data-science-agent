import pytest
import pandas as pd
import os
from pathlib import Path
from typer.testing import CliRunner
from get_kaggle_data import app, human_size, dir_size
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

runner = CliRunner()

@contextmanager
def within_tmp_path(tmp_path):
    old_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(old_dir)

def test_human_size_unit_conversion():
    assert human_size(1023) == "1023.0B"
    assert human_size(1024) == "1.0KiB"
    assert human_size(1024**2) == "1.0MiB"

def test_dir_size_bytes(tmp_path):
    testfile = tmp_path / "sample.txt"
    testfile.write_text("A" * 2048)
    result = dir_size(str(tmp_path))
    assert isinstance(result, str)
    assert "KiB" in result or "MiB" in result

@patch("get_kaggle_data.subprocess.run")
def test_cli_creates_parquet_and_split(mock_subproc, tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "test_user")
    monkeypatch.setenv("KAGGLE_KEY", "test_key")
    mock_subproc.return_value = MagicMock(returncode=0, stdout="Mocked")

    with within_tmp_path(tmp_path):
        raw = Path("data/mock_comp/raw")
        raw.mkdir(parents=True)

        df = pd.DataFrame({
            "Name": ["Jordan", "Taylor"],
            "Cabin": ["Z99", "C88"],
            "Pclass": [3, 2],
            "Fare": [14.50, 72.00]
        })
        df.to_csv(raw / "train.csv", index=False)

        result = runner.invoke(app, ["mock_comp"])
        assert result.exit_code == 0

        proc = Path("data/mock_comp/processed")
        splits = proc / "splits"
        assert (proc / "train.parquet").exists()
        assert (splits / "train.parquet").exists()
        assert (splits / "test.parquet").exists()

@patch("get_kaggle_data.subprocess.run")
def test_cli_skips_existing_splits(mock_subproc, tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "test_user")
    monkeypatch.setenv("KAGGLE_KEY", "test_key")
    mock_subproc.return_value = MagicMock(returncode=0, stdout="Mocked")

    with within_tmp_path(tmp_path):
        base = Path("data/mock_comp")
        raw = base / "raw"
        proc = base / "processed"
        splits = proc / "splits"
        raw.mkdir(parents=True)
        proc.mkdir(parents=True)
        splits.mkdir(parents=True)

        df = pd.DataFrame({
            "Name": ["Alex"],
            "Cabin": ["B12"],
            "Pclass": [2],
            "Fare": [85.0]
        })
        df.to_csv(raw / "train.csv", index=False)
        df.to_parquet(splits / "train.parquet", index=False)
        df.to_parquet(splits / "test.parquet", index=False)

        result = runner.invoke(app, ["mock_comp"])
        assert result.exit_code == 0
        assert "⚠️  Splits already exist" in result.stdout