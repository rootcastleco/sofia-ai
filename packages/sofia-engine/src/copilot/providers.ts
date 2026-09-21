/**
 * AI Engine Providers for Sofia Copilot in TypeScript / Node.js.
 * Supports NVIDIA NIM (build.nvidia.com) and OpenRouter (openrouter.ai).
 */

declare const process: { env: Record<string, string | undefined> } | undefined;

export interface CopilotMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface AIEngineOptions {
  apiKey?: string;
  model?: string;
  baseUrl?: string;
  timeoutMs?: number;
}

export class NvidiaProvider {
  readonly providerName = "nvidia";
  private readonly apiKey: string | undefined;
  private readonly model: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;

  constructor(options: AIEngineOptions = {}) {
    this.apiKey =
      options.apiKey ||
      (typeof process !== "undefined"
        ? process.env["NVIDIA_API_KEY"] || process.env["SOFIA_NVIDIA_API_KEY"]
        : undefined);
    this.model = options.model || "nvidia/llama-3.1-nemotron-70b-instruct";
    this.baseUrl = (options.baseUrl || "https://integrate.api.nvidia.com/v1").replace(/\/$/, "");
    this.timeoutMs = options.timeoutMs || 30000;
  }

  get isAvailable(): boolean {
    return Boolean(this.apiKey && this.apiKey.trim().length > 0);
  }

  async complete(prompt: string, maxTokens = 512, temperature = 0.2): Promise<string> {
    if (!this.isAvailable) {
      throw new Error("NVIDIA API key not found. Set NVIDIA_API_KEY or SOFIA_NVIDIA_API_KEY.");
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
          "User-Agent": "Sofia-Engine/2.0 (Rootcastle)"
        },
        body: JSON.stringify({
          model: this.model,
          messages: [
            {
              role: "system",
              content:
                "You are Sofia Copilot, an expert industrial condition monitoring and diagnostics assistant developed by Rootcastle Engineering & Innovation. Analyze evidence objectively based on physics, ISO vibration standards, and DSP. Never invent sensor readings."
            },
            { role: "user", content: prompt }
          ],
          max_tokens: maxTokens,
          temperature
        }),
        signal: controller.signal
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`NVIDIA API HTTP ${response.status}: ${errorText}`);
      }

      const data = (await response.json()) as { choices?: Array<{ message?: { content?: string } }> };
      return data.choices?.[0]?.message?.content?.trim() || "";
    } finally {
      clearTimeout(timer);
    }
  }
}

export class OpenRouterProvider {
  readonly providerName = "openrouter";
  private readonly apiKey: string | undefined;
  private readonly model: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;

  constructor(options: AIEngineOptions = {}) {
    this.apiKey =
      options.apiKey ||
      (typeof process !== "undefined"
        ? process.env["OPENROUTER_API_KEY"] || process.env["SOFIA_OPENROUTER_API_KEY"]
        : undefined);
    this.model = options.model || "anthropic/claude-3.5-sonnet";
    this.baseUrl = (options.baseUrl || "https://openrouter.ai/api/v1").replace(/\/$/, "");
    this.timeoutMs = options.timeoutMs || 30000;
  }

  get isAvailable(): boolean {
    return Boolean(this.apiKey && this.apiKey.trim().length > 0);
  }

  async complete(prompt: string, maxTokens = 512, temperature = 0.2): Promise<string> {
    if (!this.isAvailable) {
      throw new Error("OpenRouter API key not found. Set OPENROUTER_API_KEY or SOFIA_OPENROUTER_API_KEY.");
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
          "HTTP-Referer": "https://rootcastle.com/",
          "X-Title": "Sofia Engine (Rootcastle)",
          "User-Agent": "Sofia-Engine/2.0 (Rootcastle)"
        },
        body: JSON.stringify({
          model: this.model,
          messages: [
            {
              role: "system",
              content:
                "You are Sofia Copilot, an expert industrial condition monitoring and diagnostics assistant developed by Rootcastle Engineering & Innovation. Analyze evidence objectively based on physics, ISO vibration standards, and DSP. Never invent sensor readings."
            },
            { role: "user", content: prompt }
          ],
          max_tokens: maxTokens,
          temperature
        }),
        signal: controller.signal
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`OpenRouter API HTTP ${response.status}: ${errorText}`);
      }

      const data = (await response.json()) as { choices?: Array<{ message?: { content?: string } }> };
      return data.choices?.[0]?.message?.content?.trim() || "";
    } finally {
      clearTimeout(timer);
    }
  }
}

export class AIEngine {
  private readonly provider: NvidiaProvider | OpenRouterProvider | null = null;
  readonly providerName: string;

  constructor(provider: "auto" | "nvidia" | "openrouter" | "offline" = "auto", options: AIEngineOptions = {}) {
    if (provider === "nvidia") {
      this.provider = new NvidiaProvider(options);
      this.providerName = "nvidia";
    } else if (provider === "openrouter") {
      this.provider = new OpenRouterProvider(options);
      this.providerName = "openrouter";
    } else if (provider === "auto") {
      const nvidia = new NvidiaProvider(options);
      if (nvidia.isAvailable) {
        this.provider = nvidia;
        this.providerName = "nvidia";
      } else {
        const openrouter = new OpenRouterProvider(options);
        if (openrouter.isAvailable) {
          this.provider = openrouter;
          this.providerName = "openrouter";
        } else {
          this.provider = null;
          this.providerName = "offline";
        }
      }
    } else {
      this.provider = null;
      this.providerName = "offline";
    }
  }

  async ask(question: string, context?: { deviceId?: string; evidence?: string[] }): Promise<string> {
    if (!this.provider) {
      return `[Sofia Offline Copilot] Device: ${context?.deviceId || "unknown"}\nQuestion: ${question}\nEvidence: ${context?.evidence?.join(", ") || "none"}`;
    }

    const prompt = `Device: ${context?.deviceId || "unknown"}\nQuestion: ${question}\nEvidence: ${context?.evidence?.join("\n") || "none"}`;
    return this.provider.complete(prompt);
  }
}
