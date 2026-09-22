"""Unit tests for Provider Capability Detection and Honest Fine-Tuning (SOFIA-API-*).

Verifies:
1. SOFIA-API-001: Provider Capability Detection & FeatureNotSupportedError.
2. SOFIA-API-002: Honest Dry-Run Reporting (zero fabricated loss or accuracy values).
3. SOFIA-API-003: Real Remote File Upload Semantics.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sofia_ai.core.errors import FeatureNotSupportedError
from sofia_ai.learning.finetune import (
    AutoFineTuner,
    FineTuneStatus,
)


class TestProviderCapabilities:
    """Verification of SOFIA-API-001: Provider Capability Detection."""

    def test_provider_capability_declarations(self) -> None:
        openai_tuner = AutoFineTuner(provider="openai", dry_run=True)
        assert openai_tuner.capabilities.file_upload is True
        assert openai_tuner.capabilities.fine_tuning is True
        assert openai_tuner.capabilities.job_cancel is True

        nvidia_tuner = AutoFineTuner(provider="nvidia", dry_run=True)
        assert nvidia_tuner.capabilities.file_upload is False
        assert nvidia_tuner.capabilities.fine_tuning is False
        assert nvidia_tuner.capabilities.job_cancel is False

        openrouter_tuner = AutoFineTuner(provider="openrouter", dry_run=True)
        assert openrouter_tuner.capabilities.file_upload is False
        assert openrouter_tuner.capabilities.fine_tuning is False
        assert openrouter_tuner.capabilities.job_cancel is False

    def test_unsupported_operation_raises_feature_not_supported(self, tmp_path: Path) -> None:
        # Online NVIDIA tuner should reject file_upload and fine_tuning
        tuner = AutoFineTuner(provider="nvidia", api_key="nvapi-test-key", dry_run=False)
        dummy_file = tmp_path / "dataset.jsonl"
        dummy_file.write_text('{"messages": []}\n', encoding="utf-8")

        with pytest.raises(FeatureNotSupportedError) as exc_info:
            tuner.upload_dataset(dummy_file)
        assert exc_info.value.code == "FEATURE_NOT_SUPPORTED"

        with pytest.raises(FeatureNotSupportedError) as exc_info:
            tuner.create_job(dummy_file)
        assert exc_info.value.code == "FEATURE_NOT_SUPPORTED"

        with pytest.raises(FeatureNotSupportedError) as exc_info:
            tuner.cancel_job("ftjob-123")
        assert exc_info.value.code == "FEATURE_NOT_SUPPORTED"


class TestHonestDryRun:
    """Verification of SOFIA-API-002: Honest Dry-Run Reporting."""

    def test_dry_run_produces_no_fabricated_metrics(self, tmp_path: Path) -> None:
        tuner = AutoFineTuner(dry_run=True)
        dummy_file = tmp_path / "dataset.jsonl"
        dummy_file.write_text('{"messages": []}\n', encoding="utf-8")

        job = tuner.create_job(dummy_file)
        assert job.simulated is True
        assert job.status == FineTuneStatus.DRY_RUN
        # Invariant: zero fabricated metrics
        assert job.metrics is None
        assert job.fine_tuned_model is None

    def test_dry_run_cancel_job(self, tmp_path: Path) -> None:
        tuner = AutoFineTuner(dry_run=True)
        dummy_file = tmp_path / "dataset.jsonl"
        dummy_file.write_text('{"messages": []}\n', encoding="utf-8")

        job = tuner.create_job(dummy_file)
        cancelled = tuner.cancel_job(job.job_id)
        assert cancelled.status == FineTuneStatus.CANCELLED
        assert cancelled.simulated is True


class TestRemoteFileUpload:
    """Verification of SOFIA-API-003: Real Remote File Upload Semantics."""

    def test_dry_run_simulated_upload(self, tmp_path: Path) -> None:
        tuner = AutoFineTuner(dry_run=True)
        dummy_file = tmp_path / "dataset.jsonl"
        dummy_file.write_text('{"messages": []}\n', encoding="utf-8")

        file_id = tuner.upload_dataset(dummy_file)
        assert file_id.startswith("file-dryrun-")

    def test_online_upload_multipart_request(self, tmp_path: Path) -> None:
        tuner = AutoFineTuner(provider="openai", api_key="sk-test-key", dry_run=False)
        dummy_file = tmp_path / "dataset.jsonl"
        dummy_file.write_text('{"messages": [{"role": "user", "content": "test"}]}\n', encoding="utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"id": "file-remote-xyz123", "purpose": "fine-tune"}'
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            file_id = tuner.upload_dataset(dummy_file)
            assert file_id == "file-remote-xyz123"

            # Verify the request payload sent to /files
            call_args = mock_urlopen.call_args[0]
            req = call_args[0]
            assert "/files" in req.full_url
            assert "multipart/form-data" in req.headers["Content-type"]
            assert b'name="purpose"\r\n\r\nfine-tune' in req.data
            assert b'filename="dataset.jsonl"' in req.data
