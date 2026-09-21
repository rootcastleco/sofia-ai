"""Unit tests for Automated Fine-Tuning and Dataset Curation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sofia_ai.learning.finetune import (
    AutoFineTuner,
    DatasetCurator,
    FineTuneJob,
    FineTuneStatus,
)


def test_dataset_curator_add_and_validate(tmp_path: Path) -> None:
    curator = DatasetCurator()
    assert not curator.validate()["valid"]

    curator.add_example("What is vibration RMS?", "RMS is 2.4 mm/s, within normal ISO limits.")
    val = curator.validate()
    assert val["valid"]
    assert val["total_examples"] == 1
    assert val["estimated_tokens"] > 0

    out_file = tmp_path / "dataset.jsonl"
    exported = curator.export_jsonl(out_file)
    assert exported.exists()

    lines = [json.loads(line) for line in exported.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert len(lines[0]["messages"]) == 3
    assert lines[0]["messages"][1]["role"] == "user"
    assert lines[0]["messages"][2]["role"] == "assistant"


def test_dataset_curator_from_telemetry(tmp_path: Path) -> None:
    curator = DatasetCurator()
    records = [
        {
            "device_id": "pump-01",
            "vibration_rms": 4.2,
            "severity": "WARNING",
            "findings": ["Bearing unbalance detected"],
            "recommendation": "Perform dynamic balancing.",
        },
        {
            "device_id": "motor-02",
            "temperature_c": 65.0,
            "severity": "NORMAL",
            "findings": ["Nominal baseline."],
            "recommendation": "Continue standard operations.",
        },
    ]
    count = curator.from_telemetry_records(records)
    assert count == 2
    assert len(curator.examples) == 2

    out_file = tmp_path / "telemetry_ft.jsonl"
    curator.export_jsonl(out_file)
    assert out_file.exists()


def test_auto_fine_tuner_dry_run(tmp_path: Path) -> None:
    tuner = AutoFineTuner(dry_run=True)
    assert not tuner.is_online

    dummy_data = tmp_path / "train.jsonl"
    dummy_data.write_text('{"messages": []}\n', encoding="utf-8")

    job = tuner.create_job(dummy_data, model="nvidia/llama-3.1-8b-instruct")
    assert job.status == FineTuneStatus.SUCCEEDED
    assert job.fine_tuned_model is not None
    assert "finetuned" in job.fine_tuned_model

    registry_path = tmp_path / "registry.json"
    reg = tuner.register_checkpoint(job, registry_path=registry_path)
    assert reg["model_id"] == job.fine_tuned_model
    assert registry_path.exists()

    loaded_reg = json.loads(registry_path.read_text(encoding="utf-8"))
    assert job.fine_tuned_model in loaded_reg["fine_tuned_models"]


def test_auto_fine_tuner_online_mock(tmp_path: Path) -> None:
    tuner = AutoFineTuner(
        provider="nvidia",
        api_key="nvapi-fake-key",
        dry_run=False,
    )
    assert tuner.is_online

    dummy_data = tmp_path / "train.jsonl"
    dummy_data.write_text('{"messages": []}\n', encoding="utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"id": "ftjob-online-123", "status": "queued", "model": "nvidia/llama-3.1-8b-instruct"}'
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        job = tuner.create_job(dummy_data)
        assert job.job_id == "ftjob-online-123"
        assert job.status == FineTuneStatus.QUEUED


def test_auto_fine_tuner_get_status_mock() -> None:
    tuner = AutoFineTuner(
        provider="openai",
        api_key="sk-openai-fake",
        dry_run=False,
    )
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"id": "ftjob-456", "status": "succeeded", "fine_tuned_model": "ft:gpt-4o-mini:rootcastle"}'
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        job = tuner.get_job_status("ftjob-456")
        assert job.status == FineTuneStatus.SUCCEEDED
        assert job.fine_tuned_model == "ft:gpt-4o-mini:rootcastle"


def test_auto_tune_from_telemetry_end_to_end(tmp_path: Path) -> None:
    tuner = AutoFineTuner(dry_run=True)
    records = [
        {"device_id": "compressor-01", "rms": 1.2, "severity": "NORMAL"},
    ]
    dataset_file = tmp_path / "auto_ft.jsonl"
    registry_file = tmp_path / "registry.json"

    job = tuner.auto_tune_from_telemetry(
        records=records,
        dataset_output_path=dataset_file,
        registry_path=registry_file,
    )
    assert job.status == FineTuneStatus.SUCCEEDED
    assert dataset_file.exists()
    assert registry_file.exists()
