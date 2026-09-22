/**
 * Core data contracts for Sofia Engine.
 * Matches Python implementation in `sofia_ai.core.contracts`.
 */

export enum DataQuality {
  GOOD = "GOOD",
  DEGRADED = "DEGRADED",
  SATURATED = "SATURATED",
  STALE = "STALE",
  MISSING = "MISSING",
  INVALID = "INVALID",
  OUT_OF_RANGE = "OUT_OF_RANGE",
  DUPLICATE = "DUPLICATE",
  ESTIMATED = "ESTIMATED",
  UNSYNCHRONIZED = "UNSYNCHRONIZED"
}

export const SignalQuality = DataQuality;
export type SignalQuality = DataQuality;

export enum Severity {
  NORMAL = "NORMAL",
  LOW = "LOW",
  MEDIUM = "MEDIUM",
  HIGH = "HIGH",
  CRITICAL = "CRITICAL"
}

export enum MachineState {
  UNKNOWN = "UNKNOWN",
  OFF = "OFF",
  OFFLINE = "OFFLINE",
  STARTUP = "STARTUP",
  STARTING = "STARTING",
  IDLE = "IDLE",
  RUNNING = "RUNNING",
  OPERATIONAL = "OPERATIONAL",
  DEGRADED = "DEGRADED",
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

export type Sample = TelemetrySample;

export interface SignalMetadata {
  channel: string;
  deviceId: string;
  sampleRate: number;
  unit: string;
  sourceId?: string;
  tags?: Record<string, string>;
}

export interface SignalFrame {
  data: Float64Array;
  metadata: SignalMetadata;
  startTime: number;
  endTime: number;
  quality: DataQuality;
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

export interface Feature {
  name: string;
  value: number;
  unit?: string;
  uncertainty?: number;
  algorithm?: string;
}

export interface FeatureVector {
  names: readonly string[];
  values: Float64Array;
  extractorId: string;
  extractorVersion: string;
  timestamp?: number;
  schemaVersion?: string;
  uncertainties?: Float64Array;
  features?: Feature[];
  asDict(): Record<string, number>;
}

export interface InferenceRequest {
  modelId: string;
  featureVector: FeatureVector;
  timestamp?: number;
  context?: Record<string, unknown>;
}

export interface RuntimeFault {
  faultCode: string;
  message: string;
  component: string;
  timestamp: number;
  details?: Record<string, unknown>;
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
