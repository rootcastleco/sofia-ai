# Automatic Fine-Tuning & Dataset Curation Pipeline

> **Rootcastle Engineering & Innovation** | Specification Trace: `SOFIA-LRN-005` - `006`

---

## 1. Overview

Edge devices collect vast quantities of high-fidelity physical telemetry, but domain-specific Large Language Models (LLMs) often lack context about local machine quirks, specialized ISO standard interpretations, or facility-specific operating envelopes.

Sofia's **Automated Fine-Tuning Subsystem (`sofia_ai.learning.finetune`)** bridges this gap by providing:
1. **Autonomous Telemetry Curation (`DatasetCurator`)**: Ingests sensor readings, FFT spectral peaks, harmonic power distortions, and diagnostic events, formatting them into standardized JSONL conversational pairs with system prompt injection, token estimation, and integrity validation.
2. **Zero-Dependency API Submission (`AutoFineTuner`)**: Submits fine-tuning jobs to **NVIDIA NIM** (`api.nvidia.com`), **OpenAI-compatible**, or **OpenRouter** endpoints using Python standard library `urllib.request`.
3. **Automated Checkpoint Registry (`models/registry.json`)**: Tracks job progress, downloads or references resulting fine-tuned model IDs, and automatically registers them for immediate use in Sofia's AI Copilot.

---

## 2. Dataset Format & Schema

Curated datasets strictly adhere to the standard multi-turn chat completion JSONL schema:

```json
{"messages": [{"role": "system", "content": "You are Sofia AI..."}, {"role": "user", "content": "Analyze telemetry record: {...}"}, {"role": "assistant", "content": "Diagnostic Assessment: WARNING\nKey Findings: Bearing unbalance detected\nAction Recommended: Perform dynamic balancing."}]}
```

### Dataset Validation Checks
* **Message Completeness**: Each record must have at least 2 messages (system/user/assistant).
* **Non-Empty Content**: Prevents blank prompts from poisoning gradient calculations.
* **Token Estimation**: Calculates token volume ($\approx 4$ characters per token) to verify dataset compliance with provider fine-tuning minimums.

---

## 3. Python End-to-End Workflow

```python
from sofia_ai.learning import AutoFineTuner, DatasetCurator

# 1. Collect telemetry & diagnostic events from edge fleet
records = [
    {
        "device_id": "chiller-compressor-01",
        "vibration_rms": 4.8,
        "thd_v": 3.2,
        "cavitation_index": 0.45,
        "severity": "WARNING",
        "findings": ["High cavitation index in refrigerant expansion valve"],
        "recommendation": "Inspect valve seat for erosion and clear debris."
    },
    {
        "device_id": "feedwater-pump-03",
        "vibration_rms": 1.1,
        "thd_v": 1.4,
        "cavitation_index": 0.05,
        "severity": "NORMAL",
        "findings": ["Operating within nominal ISO 10816 Class II zone A."],
        "recommendation": "Continue standard predictive maintenance schedule."
    }
]

# 2. Automated Fine-Tuning Pipeline
tuner = AutoFineTuner(
    provider="nvidia", # Or "openai", "openrouter"
    default_model="nvidia/llama-3.1-8b-instruct"
)

# Curates dataset, exports JSONL, creates job, and registers checkpoint
job = tuner.auto_tune_from_telemetry(
    records=records,
    dataset_output_path="data/fleet_telemetry_ft.jsonl",
    registry_path="models/registry.json"
)

print(f"Fine-Tuning Job Created: {job.job_id}")
print(f"Status: {job.status}")
print(f"Model: {job.model}")
if job.fine_tuned_model:
    print(f"Registered Checkpoint: {job.fine_tuned_model}")
```

---

## 4. TypeScript / Node.js Workflow

```typescript
import { AutoFineTuner, DatasetCurator } from "@rootcastle/sofia-engine";

const curator = new DatasetCurator();
curator.fromTelemetryRecords([
  {
    device_id: "induction-motor-04",
    vuf_percent: 2.8,
    severity: "CRITICAL",
    findings: ["Severe 3-phase voltage unbalance exceeding 2% limit"],
    recommendation: "De-energize motor immediately to prevent stator winding failure.",
  },
]);

const jsonlData = curator.toJSONL();

const tuner = new AutoFineTuner({ provider: "nvidia", dryRun: true });
const job = await tuner.createJob(jsonlData);
console.log("Job status:", job.status, "Checkpoint:", job.fineTunedModel);
```
