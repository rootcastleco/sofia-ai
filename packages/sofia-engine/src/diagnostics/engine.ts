/**
 * Diagnostic evidence fusion and machine health evaluation.
 */

import {
  DataQuality,
  HealthEvent,
  HealthScore,
  Severity
} from "../contracts.js";

export interface RuleContext {
  deviceId: string;
  channel: string;
  score: number;
  confidence: number;
  quality: DataQuality;
  featureValues?: Record<string, number>;
  unit?: string;
}

export class DiagnosticEngine {
  readonly ruleVersion = "2.0.0";

  evaluate(context: RuleContext): HealthEvent | null {
    if (context.score < 2.0) {
      return null;
    }

    let severity = Severity.LOW;
    if (context.score >= 5.0) {
      severity = Severity.CRITICAL;
    } else if (context.score >= 3.5) {
      severity = Severity.HIGH;
    } else if (context.score >= 2.5) {
      severity = Severity.MEDIUM;
    }

    // Degrade confidence by data quality
    let qualityFactor = 1.0;
    switch (context.quality) {
      case DataQuality.GOOD:
        qualityFactor = 1.0;
        break;
      case DataQuality.STALE:
        qualityFactor = 0.5;
        break;
      case DataQuality.DUPLICATE:
        qualityFactor = 0.3;
        break;
      case DataQuality.OUT_OF_RANGE:
        qualityFactor = 0.2;
        break;
      default:
        qualityFactor = 0.1;
    }

    const finalConfidence = context.confidence * qualityFactor;
    const finalUncertainty = 1.0 - finalConfidence;

    return {
      eventId: `evt_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      deviceId: context.deviceId,
      channel: context.channel,
      severity,
      eventType: "TELEMETRY_ANOMALY",
      confidence: finalConfidence,
      uncertainty: finalUncertainty,
      recommendation: `Inspect ${context.channel} on ${context.deviceId}: score=${context.score.toFixed(2)} exceeds threshold.`,
      evidence: [
        {
          source: "detector",
          metric: "anomaly_score",
          observedValue: context.score,
          description: `Score exceeded baseline in ${context.channel}`,
          weight: 1.0
        }
      ],
      timestamp: Date.now() / 1000,
      ruleVersion: this.ruleVersion
    };
  }
}

/**
 * Computes holistic machine health score (0 - 100) with dynamic uncertainty band.
 */
export function computeHealthScore(events: HealthEvent[]): HealthScore {
  const timestamp = Date.now() / 1000;
  if (events.length === 0) {
    return {
      score: 100,
      lower: 100,
      upper: 100,
      confidence: 1.0,
      timestamp
    };
  }

  let totalPenalty = 0;
  let totalUncertainty = 0;

  for (const event of events) {
    let penalty = 0;
    switch (event.severity) {
      case Severity.CRITICAL:
        penalty = 100;
        break;
      case Severity.HIGH:
        penalty = 65;
        break;
      case Severity.MEDIUM:
        penalty = 35;
        break;
      case Severity.LOW:
        penalty = 15;
        break;
      default:
        penalty = 0;
    }

    totalPenalty += penalty * event.confidence;
    totalUncertainty += event.uncertainty * penalty;
  }

  const score = Math.max(0, 100 - Math.min(100, totalPenalty));
  const delta = Math.min(score, Math.min(100 - score, totalUncertainty * 0.5));

  return {
    score,
    lower: Math.max(0, score - delta),
    upper: Math.min(100, score + delta),
    confidence: Math.max(0.1, 1.0 - totalUncertainty / 100),
    timestamp
  };
}
