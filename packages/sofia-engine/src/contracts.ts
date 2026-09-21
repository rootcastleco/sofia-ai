/**
 * Core data contracts for Sofia Engine.
 * Matches Python implementation in `sofia_ai.core.contracts`.
 */

export enum DataQuality {
  GOOD = "GOOD",
  STALE = "STALE",
  MISSING = "MISSING",
  INVALID = "INVALID",
  OUT_OF_RANGE = "OUT_OF_RANGE",
  DUPLICATE = "DUPLICATE",
  ESTIMATED = "ESTIMATED",
  UNSYNCHRONIZED = "UNSYNCHRONIZED"
}

export enum Severity {
  NORMAL = "NORMAL",
  LOW = "LOW",
  MEDIUM = "MEDIUM",
  HIGH = "HIGH",
  CRITICAL = "CRITICAL"
}

export enum MachineState {
  UNKNOWN = "UNKNOWN",
  OFFLINE = "OFFLINE",
  STARTING = "STARTING",
  OPERATIONAL = "OPERATIONAL",
  STANDBY = "STANDBY",
  MAINTENANCE = "MAINTENANCE",
  FAULT = "FAULT",
  EMERGENCY_STOP = "EMERGENCY_STOP"
}

export enum CommandVerdict {
  APPROVE = "APPROVE",
  DENY = "DENY"
}

export interface TelemetrySample {
  timestamp: number;
  deviceId: string;
  channel: string;
  value: number;
  unit: string;
  quality: DataQuality;
  sequenceNumber: number;
  sourceId?: string;
  ingestionTime?: number;
  tags?: Record<string, string>;
}

export interface SignalWindow {
  values: Float64Array;
  sampleRate: number;
  startIndex: number;
  startTime: number;
  channel: string;
  unit: string;
  deviceId: string;
  qualityMask?: boolean[];
}

export interface FeatureVector {
  names: readonly string[];
  values: Float64Array;
  extractorId: string;
  extractorVersion: string;
  timestamp?: number;
  asDict(): Record<string, number>;
}

export interface InferenceResult {
  modelId: string;
  modelVersion: string;
  outcome: "NORMAL" | "ANOMALY";
  score: number;
  confidence: number;
  uncertainty: number;
  latencyMs: number;
  timestamp: number;
  evidence?: Record<string, number>;
}

export interface DiagnosticEvidence {
  source: string;
  metric: string;
  observedValue: number;
  referenceValue?: number;
  description: string;
  weight: number;
}

export interface HealthEvent {
  eventId: string;
  deviceId: string;
  channel: string;
  severity: Severity;
  eventType: string;
  confidence: number;
  uncertainty: number;
  recommendation: string;
  evidence: DiagnosticEvidence[];
  timestamp: number;
  ruleVersion: string;
}

export interface HealthScore {
  score: number; // 0 - 100
  lower: number; // Lower uncertainty bound
  upper: number; // Upper uncertainty bound
  confidence: number;
  timestamp: number;
}

export interface CommandRequest {
  action: string;
  deviceId: string;
  machineState: MachineState;
  confidence: number;
  operatorApproved?: boolean;
  interlocksClear?: boolean;
  nonce?: string;
  timestamp?: number;
  parameters?: Record<string, unknown>;
}

export interface CommandDecision {
  verdict: CommandVerdict;
  action: string;
  deviceId: string;
  reason?: string;
  timestamp: number;
  decisionId: string;
}
