# AI Engine & Copilot

Sofia Engine includes an integrated **AI Copilot & Diagnostics Engine** (`sofia_ai.copilot`) designed specifically for technical decision-support in industrial operations.

The AI Engine can connect to high-performance LLM backends using **NVIDIA NIM** or **OpenRouter**, or operate completely **offline** in air-gapped environments.

---

## 1. Supported AI Providers

| Provider | Supported Backends / Models | Environment Keys | Primary Use Case |
|---|---|---|---|
| **NVIDIA NIM** | `nvidia/llama-3.1-nemotron-70b-instruct`, `meta/llama-3.1-70b-instruct` | `NVIDIA_API_KEY`, `SOFIA_NVIDIA_API_KEY` | High-throughput edge server inference via [build.nvidia.com](https://build.nvidia.com) |
| **OpenRouter** | Claude 3.5 Sonnet, GPT-4o, Llama 3.3, DeepSeek R1, Mistral | `OPENROUTER_API_KEY`, `SOFIA_OPENROUTER_API_KEY` | Multi-model routing and advanced reasoning via [openrouter.ai](https://openrouter.ai) |
| **Offline (Default)** | Deterministic Evidence Formatter | *(None required)* | Air-gapped, zero-network edge gateways |

---

## 2. Setting Up API Keys

Set your preferred API keys in your environment:

### Linux / macOS:
```bash
export NVIDIA_API_KEY="nvapi-..."
# or
export OPENROUTER_API_KEY="sk-or-..."
```

### Windows (PowerShell):
```powershell
$env:NVIDIA_API_KEY = "nvapi-..."
# or
$env:OPENROUTER_API_KEY = "sk-or-..."
```

---

## 3. Python Usage

```python
from sofia_ai.copilot import AIEngine, CopilotRequest

# 1. Initialize engine (auto-detects available NVIDIA or OpenRouter keys)
engine = AIEngine(provider="auto")
print(f"Active Provider: {engine.provider_name}")

# 2. Query the copilot with structured evidence
response = engine.ask(CopilotRequest(
    question="Why is Crest Factor 4.8 considered dangerous on pump-01?",
    device_id="pump-01",
    evidence=(
        {"source": "vibration_x", "metric": "crest_factor", "observed": 4.8, "reference": 3.2},
        {"source": "vibration_x", "metric": "rms", "observed": 3.8, "unit": "mm/s"}
    ),
    audience="maintenance"
))

print(response.text)
```

---

## 4. CLI Usage (`sofia ask`)

You can consult the AI Copilot directly from the terminal:

```bash
# Check key availability
sofia doctor

# Ask using auto-detected provider
sofia ask "What does high kurtosis indicate in rolling bearings?" --device pump-01

# Explicitly use NVIDIA NIM
sofia ask "Explain spectral sidebands around gear mesh frequency." --provider nvidia

# Explicitly use OpenRouter with a specific model
sofia ask "Analyze bearing health" --provider openrouter --model "anthropic/claude-3.5-sonnet"
```

---

## 5. TypeScript / Node.js Usage

Sofia's npm package (`@rootcastle/sofia-engine`) also provides full AI Engine support:

```typescript
import { AIEngine, NvidiaProvider, OpenRouterProvider } from "@rootcastle/sofia-engine";

// Auto-discovery from process.env.NVIDIA_API_KEY or process.env.OPENROUTER_API_KEY
const ai = new AIEngine("auto");

const answer = await ai.ask("Explain ISO 10816 Zone C vibration limits", {
  deviceId: "turbine-02",
  evidence: ["rms: 5.2 mm/s", "kurtosis: 4.1"]
});

console.log(answer);
```
