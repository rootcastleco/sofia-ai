/**
 * Statistical anomaly detectors for industrial telemetry.
 */

import { InferenceResult } from "../contracts.js";

export interface AnomalyDetector {
  readonly modelId: string;
  readonly modelVersion: string;
  infer(value: number): InferenceResult;
}

/**
 * Calculates the median of an array of numbers.
 */
export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 !== 0) {
    return sorted[mid]!;
  }
  return (sorted[mid - 1]! + sorted[mid]!) / 2;
}

/**
 * Robust Median Absolute Deviation (MAD) Anomaly Detector.
 * Provides 50% breakdown point against outlier poisoning.
 */
export class MadDetector implements AnomalyDetector {
  readonly modelId = "detector.mad";
  readonly modelVersion = "2.0.0";

  private readonly history: number[] = [];
  private readonly windowSize: number;
  private readonly threshold: number;
  private readonly warmup: number;

  constructor(options: { windowSize?: number; threshold?: number; warmup?: number } = {}) {
    this.windowSize = options.windowSize ?? 32;
    this.threshold = options.threshold ?? 3.0;
    this.warmup = options.warmup ?? 8;
  }

  infer(value: number): InferenceResult {
    const start = performance.now();
    this.history.push(value);
    if (this.history.length > this.windowSize) {
      this.history.shift();
    }

    if (this.history.length < this.warmup) {
      return {
        modelId: this.modelId,
        modelVersion: this.modelVersion,
        outcome: "NORMAL",
        score: 0,
        confidence: 0.1,
        uncertainty: 0.9,
        latencyMs: performance.now() - start,
        timestamp: Date.now() / 1000
      };
    }

    const med = median(this.history);
    const absDiffs = this.history.map((v) => Math.abs(v - med));
    const mad = median(absDiffs);

    // Consistency constant: 1 / norm.ppf(0.75) ≈ 1.4826, or multiply diff by 0.6745
    const denom = mad === 0 ? 1e-6 : mad * 1.4826;
    const modifiedZ = Math.abs(value - med) / denom;

    const isAnomaly = modifiedZ > this.threshold;
    const confidence = Math.min(1.0, Math.max(0.5, 0.5 + modifiedZ / (2 * this.threshold)));
    const uncertainty = 1.0 - confidence;

    return {
      modelId: this.modelId,
      modelVersion: this.modelVersion,
      outcome: isAnomaly ? "ANOMALY" : "NORMAL",
      score: modifiedZ,
      confidence,
      uncertainty,
      latencyMs: performance.now() - start,
      timestamp: Date.now() / 1000,
      evidence: { value, median: med, mad, score: modifiedZ }
    };
  }
}

/**
 * Exponentially Weighted Moving Average (EWMA) Detector.
 */
export class EwmaDetector implements AnomalyDetector {
  readonly modelId = "detector.ewma";
  readonly modelVersion = "2.0.0";

  private ewma: number | null = null;
  private readonly lambda: number;
  private readonly threshold: number;

  constructor(options: { lambda?: number; threshold?: number } = {}) {
    this.lambda = options.lambda ?? 0.2;
    this.threshold = options.threshold ?? 3.0;
  }

  infer(value: number): InferenceResult {
    const start = performance.now();
    if (this.ewma === null) {
      this.ewma = value;
      return {
        modelId: this.modelId,
        modelVersion: this.modelVersion,
        outcome: "NORMAL",
        score: 0,
        confidence: 0.5,
        uncertainty: 0.5,
        latencyMs: performance.now() - start,
        timestamp: Date.now() / 1000
      };
    }

    this.ewma = this.lambda * value + (1 - this.lambda) * this.ewma;
    const diff = Math.abs(value - this.ewma);
    const isAnomaly = diff > this.threshold;

    return {
      modelId: this.modelId,
      modelVersion: this.modelVersion,
      outcome: isAnomaly ? "ANOMALY" : "NORMAL",
      score: diff,
      confidence: 0.85,
      uncertainty: 0.15,
      latencyMs: performance.now() - start,
      timestamp: Date.now() / 1000
    };
  }
}
