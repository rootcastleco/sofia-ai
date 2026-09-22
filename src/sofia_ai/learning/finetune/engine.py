"""Automated Fine-Tuning Engine for Sofia AI.

Provides automated dataset curation from edge telemetry and diagnostics,
zero-dependency API communication (via standard library urllib.request) with
fine-tuning endpoints (NVIDIA NIM, OpenAI-compatible, OpenRouter), job tracking,
and automated model checkpoint registration.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from ...core.errors import FeatureNotSupportedError
from ...security.secrets import secret

logger = logging.getLogger(__name__)

__all__ = [
    "AutoFineTuner",
    "DatasetCurator",
    "FineTuneJob",
    "FineTuneStatus",
    "ProviderCapabilities",
]

DEFAULT_NVIDIA_BASE_URL: Final[str] = "https://integrate.api.nvidia.com/v1"
DEFAULT_OPENAI_BASE_URL: Final[str] = "https://api.openai.com/v1"


class FineTuneStatus(enum.StrEnum):
    """Fine-tuning job lifecycle statuses."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DRY_RUN = "dry_run"


@dataclass(frozen=True)
class ProviderCapabilities:
    """Declared capabilities for a remote provider adapter."""

    provider: str
    file_upload: bool
    fine_tuning: bool
    job_cancel: bool
    inference: bool = True


@dataclass
class FineTuneJob:
    """Represents a fine-tuning job and its metadata."""

    job_id: str
    model: str
    status: FineTuneStatus
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    fine_tuned_model: str | None = None
    training_file: str | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] | None = None
    error_message: str | None = None
    simulated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert job representation to JSON-serializable dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data


class DatasetCurator:
    """Curates telemetry, diagnostic events, and scientific data into training pairs.

    Exports data in standard JSONL chat format:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """

    DEFAULT_SYSTEM_PROMPT: Final[str] = (
        "You are Sofia AI, a scientific intelligence and industrial diagnostics "
        "assistant developed by Rootcastle Engineering & Innovation. Analyze physical "
        "telemetry, vibration, electrical, acoustic, and DSP features with scientific "
        "rigor and deterministic safety."
    )

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.examples: list[dict[str, Any]] = []

    def add_example(
        self,
        user_prompt: str,
        assistant_response: str,
        system_prompt: str | None = None,
    ) -> None:
        """Add a single training conversation example."""
        sys_p = system_prompt or self.system_prompt
        self.examples.append({
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_prompt.strip()},
                {"role": "assistant", "content": assistant_response.strip()},
            ]
        })

    def from_telemetry_records(
        self,
        records: list[dict[str, Any]],
        instruction_template: str = (
            "Analyze the following telemetry record and provide health assessment:\n{data}"
        ),
    ) -> int:
        """Convert raw telemetry or diagnostic records into training examples."""
        count = 0
        for rec in records:
            data_str = json.dumps(rec, sort_keys=True, indent=2)
            user_prompt = instruction_template.format(data=data_str)

            severity = rec.get("severity", "NORMAL")
            findings = rec.get("findings", ["Nominal operating parameters."])
            recommendation = rec.get(
                "recommendation", "Continue routine monitoring schedule."
            )

            assistant_response = (
                f"Diagnostic Assessment: {severity}\n"
                f"Key Findings: {', '.join(findings) if isinstance(findings, list) else findings}\n"
                f"Action Recommended: {recommendation}"
            )

            self.add_example(user_prompt, assistant_response)
            count += 1
        return count

    def validate(self) -> dict[str, Any]:
        """Validate dataset structure, token count estimate, and completeness."""
        if not self.examples:
            return {"valid": False, "total_examples": 0, "error": "Dataset is empty"}

        total_chars = 0
        for idx, ex in enumerate(self.examples):
            messages = ex.get("messages", [])
            if len(messages) < 2:
                return {
                    "valid": False,
                    "error": f"Example {idx} has fewer than 2 messages",
                }
            for msg in messages:
                content = msg.get("content", "")
                if not content:
                    return {
                        "valid": False,
                        "error": f"Example {idx} has empty content",
                    }
                total_chars += len(content)

        # Rough token estimate: ~4 characters per token
        approx_tokens = total_chars // 4
        return {
            "valid": True,
            "total_examples": len(self.examples),
            "estimated_tokens": approx_tokens,
            "error": None,
        }

    def export_jsonl(self, output_path: str | Path) -> Path:
        """Export curated examples to a JSONL file."""
        val = self.validate()
        if not val["valid"]:
            raise ValueError(f"Cannot export invalid dataset: {val['error']}")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for ex in self.examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        logger.info("Exported %d examples to %s", len(self.examples), path)
        return path


class AutoFineTuner:
    """Automated fine-tuning engine supporting NVIDIA NIM and OpenAI-compatible APIs.

    Operates with zero dependencies using standard Python urllib.request.
    Can operate in dry-run/offline mode for verification without network access.
    """

    def __init__(
        self,
        provider: str = "nvidia",
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "nvidia/llama-3.1-8b-instruct",
        dry_run: bool = False,
    ) -> None:
        self.provider = provider.lower()
        self.dry_run = dry_run
        self.default_model = default_model

        if self.provider == "nvidia":
            self.base_url = (
                base_url or os.environ.get("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL
            ).rstrip("/")
            self.api_key = (
                api_key
                or secret("NVIDIA_API_KEY")
                or os.environ.get("NVIDIA_API_KEY")
                or os.environ.get("SOFIA_NVIDIA_API_KEY")
            )
        elif self.provider in ("openai", "openrouter"):
            default_url = (
                "https://openrouter.ai/api/v1"
                if self.provider == "openrouter"
                else DEFAULT_OPENAI_BASE_URL
            )
            self.base_url = (base_url or default_url).rstrip("/")
            key_name = (
                "OPENROUTER_API_KEY" if self.provider == "openrouter" else "OPENAI_API_KEY"
            )
            self.api_key = (
                api_key
                or secret(key_name)
                or os.environ.get(key_name)
                or os.environ.get(f"SOFIA_{key_name}")
            )
        else:
            self.base_url = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
            self.api_key = api_key

        self.jobs: dict[str, FineTuneJob] = {}

    PROVIDER_CAPABILITIES: Final[dict[str, ProviderCapabilities]] = {
        "openai": ProviderCapabilities(
            provider="openai",
            file_upload=True,
            fine_tuning=True,
            job_cancel=True,
        ),
        "nvidia": ProviderCapabilities(
            provider="nvidia",
            file_upload=False,
            fine_tuning=False,
            job_cancel=False,
        ),
        "openrouter": ProviderCapabilities(
            provider="openrouter",
            file_upload=False,
            fine_tuning=False,
            job_cancel=False,
        ),
    }

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of the current provider."""
        return self.PROVIDER_CAPABILITIES.get(
            self.provider,
            ProviderCapabilities(
                provider=self.provider,
                file_upload=False,
                fine_tuning=False,
                job_cancel=False,
            ),
        )

    @property
    def is_online(self) -> bool:
        """True if valid API key is available and dry_run is disabled."""
        return bool(not self.dry_run and self.api_key and self.api_key.strip())

    def upload_dataset(self, dataset_path: str | Path) -> str:
        """Upload dataset file to remote provider or simulate upload in dry-run mode."""
        dataset_p = Path(dataset_path)
        if not dataset_p.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_p}")

        if not self.is_online:
            simulated_file_id = f"file-dryrun-{int(time.time())}"
            logger.info("Simulated dataset upload: %s -> %s", dataset_p, simulated_file_id)
            return simulated_file_id

        if not self.capabilities.file_upload:
            raise FeatureNotSupportedError(
                f"Provider '{self.provider}' does not support remote file upload.",
                details={"provider": self.provider, "capability": "file_upload"},
            )

        boundary = f"----SofiaBoundary{int(time.time() * 1000)}"
        file_bytes = dataset_p.read_bytes()
        filename = dataset_p.name

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="purpose"\r\n\r\n'
            f"fine-tune\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: application/json\r\n\r\n"
        ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()

        url = f"{self.base_url}/files"
        if not (url.startswith("https://") or url.startswith("http://")):
            raise ValueError(f"Insecure or invalid URL scheme: {url}")
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Sofia-AI/3.0 (Rootcastle)",
        }
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=60.0) as response:
                resp_body = json.loads(response.read().decode("utf-8"))
                file_id = resp_body.get("id")
                if not file_id:
                    raise RuntimeError("Upload response did not contain a file ID")
                logger.info("Uploaded %s to provider %s as %s", dataset_p, self.provider, file_id)
                return str(file_id)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            logger.exception("File upload HTTP %d: %s", exc.code, err_body)
            raise RuntimeError(f"File upload HTTP {exc.code}: {err_body}") from exc
        except Exception as exc:
            logger.exception("Failed to upload dataset: %s", exc)
            raise RuntimeError(f"File upload failed: {exc}") from exc

    def create_job(
        self,
        dataset_path: str | Path,
        model: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ) -> FineTuneJob:
        """Create a remote fine-tuning job."""
        if not self.is_online:
            sim_job = FineTuneJob(
                job_id=f"ftjob-dry-{int(time.time())}",
                model=model or self.default_model,
                status=FineTuneStatus.DRY_RUN,
                created_at=time.time(),
                hyperparameters=hyperparameters or {},
                simulated=True,
                metrics=None,
            )
            self.jobs[sim_job.job_id] = sim_job
            logger.info(
                "Offline/dry-run mode: Created simulated fine-tuning job %s with model %s",
                sim_job.job_id,
                sim_job.model,
            )
            return sim_job

        if not self.capabilities.fine_tuning:
            raise FeatureNotSupportedError(
                f"Provider '{self.provider}' does not support remote fine-tuning.",
                details={"provider": self.provider, "capability": "fine_tuning"},
            )

        file_id = self.upload_dataset(dataset_path)

        url = f"{self.base_url}/fine_tuning/jobs"
        if not (url.startswith("https://") or url.startswith("http://")):
            raise ValueError(f"Insecure or invalid URL scheme: {url}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Sofia-AI/3.0 (Rootcastle)",
        }
        payload: dict[str, Any] = {
            "training_file": file_id,
            "model": model or self.default_model,
        }
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30.0) as response:
                body = json.loads(response.read().decode("utf-8"))
                job_id = body.get("id", f"ftjob-{int(time.time())}")
                status_raw = body.get("status", "running")
                status = FineTuneStatus(status_raw) if status_raw in FineTuneStatus._value2member_map_ else FineTuneStatus.RUNNING
                job = FineTuneJob(
                    job_id=job_id,
                    model=model or self.default_model,
                    status=status,
                    created_at=time.time(),
                    hyperparameters=hyperparameters or {},
                    simulated=False,
                )
                self.jobs[job_id] = job
                logger.info("Created fine-tuning job %s via %s", job_id, self.provider)
                return job
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            logger.exception("Fine-tuning API error %d: %s", exc.code, err_body)
            raise RuntimeError(f"Fine-tuning API HTTP {exc.code}: {err_body}") from exc
        except Exception as exc:
            logger.exception("Failed to submit fine-tuning job: %s", exc)
            raise RuntimeError(f"Fine-tuning connection failed: {exc}") from exc

    def cancel_job(self, job_id: str) -> FineTuneJob:
        """Cancel an ongoing fine-tuning job."""
        if not self.is_online:
            job = self.jobs.get(job_id)
            if job is None:
                job = FineTuneJob(
                    job_id=job_id,
                    model=self.default_model,
                    status=FineTuneStatus.CANCELLED,
                    finished_at=time.time(),
                    simulated=True,
                )
                self.jobs[job_id] = job
            else:
                job.status = FineTuneStatus.CANCELLED
                job.finished_at = time.time()
            return job

        if not self.capabilities.job_cancel:
            raise FeatureNotSupportedError(
                f"Provider '{self.provider}' does not support remote job cancellation.",
                details={"provider": self.provider, "capability": "job_cancel"},
            )

        url = f"{self.base_url}/fine_tuning/jobs/{job_id}/cancel"
        if not (url.startswith("https://") or url.startswith("http://")):
            raise ValueError(f"Insecure or invalid URL scheme: {url}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Sofia-AI/3.0 (Rootcastle)",
        }
        req = urllib.request.Request(url, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=15.0) as response:
                body = json.loads(response.read().decode("utf-8"))
                job = self.jobs.get(
                    job_id,
                    FineTuneJob(job_id=job_id, model=body.get("model", ""), status=FineTuneStatus.CANCELLED),
                )
                job.status = FineTuneStatus.CANCELLED
                job.finished_at = time.time()
                self.jobs[job_id] = job
                return job
        except Exception as exc:
            logger.exception("Failed to cancel fine-tune job %s: %s", job_id, exc)
            raise RuntimeError(f"Failed to cancel job {job_id}: {exc}") from exc

    def get_job_status(self, job_id: str) -> FineTuneJob:
        """Poll the status of an ongoing fine-tuning job."""
        if job_id in self.jobs and (self.jobs[job_id].status == FineTuneStatus.SUCCEEDED or not self.is_online):
            return self.jobs[job_id]

        url = f"{self.base_url}/fine_tuning/jobs/{job_id}"
        if not (url.startswith("https://") or url.startswith("http://")):
            raise ValueError(f"Insecure or invalid URL scheme: {url}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Sofia-AI/3.0 (Rootcastle)",
        }
        req = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=15.0) as response:
                body = json.loads(response.read().decode("utf-8"))
                status_raw = body.get("status", "running")
                status = FineTuneStatus(status_raw) if status_raw in FineTuneStatus._value2member_map_ else FineTuneStatus.RUNNING

                job = self.jobs.get(
                    job_id,
                    FineTuneJob(job_id=job_id, model=body.get("model", ""), status=status),
                )
                job.status = status
                job.fine_tuned_model = body.get("fine_tuned_model")
                if status == FineTuneStatus.SUCCEEDED and not job.finished_at:
                    job.finished_at = time.time()
                self.jobs[job_id] = job
                return job
        except Exception as exc:
            logger.warning("Could not fetch status for job %s: %s", job_id, exc)
            return self.jobs.get(
                job_id,
                FineTuneJob(
                    job_id=job_id,
                    model=self.default_model,
                    status=FineTuneStatus.FAILED,
                    error_message=str(exc),
                ),
            )

    def register_checkpoint(
        self,
        job: FineTuneJob,
        registry_path: str | Path = "models/registry.json",
    ) -> dict[str, Any]:
        """Register the fine-tuned model checkpoint in the Sofia model registry."""
        if job.status not in (FineTuneStatus.SUCCEEDED, FineTuneStatus.DRY_RUN):
            raise ValueError(f"Cannot register checkpoint for job with status: {job.status}")

        reg_p = Path(registry_path)
        reg_p.parent.mkdir(parents=True, exist_ok=True)

        registry_data: dict[str, Any] = {}
        if reg_p.exists():
            try:
                registry_data = json.loads(reg_p.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to parse existing registry, starting fresh: %s", exc)

        models = registry_data.setdefault("fine_tuned_models", {})
        model_id = job.fine_tuned_model or f"sofia-ft-{job.job_id}"
        entry = {
            "model_id": model_id,
            "base_model": job.model,
            "registered_at": time.time(),
            "job_id": job.job_id,
            "hyperparameters": job.hyperparameters,
            "metrics": job.metrics,
            "provider": self.provider,
        }
        models[model_id] = entry
        reg_p.write_text(json.dumps(registry_data, indent=2), encoding="utf-8")
        logger.info("Registered fine-tuned checkpoint: %s in %s", model_id, reg_p)
        return entry

    def auto_tune_from_telemetry(
        self,
        records: list[dict[str, Any]],
        dataset_output_path: str | Path = "data/finetune_dataset.jsonl",
        model: str | None = None,
        registry_path: str | Path = "models/registry.json",
    ) -> FineTuneJob:
        """Complete automated pipeline: curate -> export -> submit job -> register."""
        curator = DatasetCurator()
        curator.from_telemetry_records(records)
        dataset_file = curator.export_jsonl(dataset_output_path)

        job = self.create_job(dataset_file, model=model)
        if job.status in (FineTuneStatus.SUCCEEDED, FineTuneStatus.DRY_RUN):
            self.register_checkpoint(job, registry_path=registry_path)

        return job
