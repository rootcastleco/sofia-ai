/**
 * Deterministic Safety Policy Engine with default DENY posture.
 */

import {
  CommandDecision,
  CommandRequest,
  CommandVerdict,
  MachineState
} from "../contracts.js";

export interface PolicyConfig {
  allowedActions: Set<string>;
  minConfidence?: number;
  operatorApprovalRequired?: boolean;
  validStates?: Set<MachineState>;
  ttlSeconds?: number;
}

export class PolicyEngine {
  private readonly allowedActions: Set<string>;
  private readonly minConfidence: number;
  private readonly operatorApprovalRequired: boolean;
  private readonly validStates: Set<MachineState>;
  private readonly ttlSeconds: number;
  private readonly seenNonces: Set<string> = new Set();

  constructor(config: PolicyConfig) {
    this.allowedActions = new Set(config.allowedActions);
    this.minConfidence = config.minConfidence ?? 0.85;
    this.operatorApprovalRequired = config.operatorApprovalRequired ?? true;
    this.validStates = config.validStates ?? new Set([MachineState.OPERATIONAL, MachineState.STANDBY]);
    this.ttlSeconds = config.ttlSeconds ?? 60;
  }

  evaluate(request: CommandRequest): CommandDecision {
    const timestamp = Date.now() / 1000;
    const decisionId = `dec_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;

    // Gate 1: Allowlist
    if (!this.allowedActions.has(request.action)) {
      return {
        verdict: CommandVerdict.DENY,
        action: request.action,
        deviceId: request.deviceId,
        reason: "ACTION_NOT_ALLOWED",
        timestamp,
        decisionId
      };
    }

    // Gate 2: Confidence Floor
    if (request.confidence < this.minConfidence) {
      return {
        verdict: CommandVerdict.DENY,
        action: request.action,
        deviceId: request.deviceId,
        reason: "CONFIDENCE_TOO_LOW",
        timestamp,
        decisionId
      };
    }

    // Gate 3: Machine State
    if (!this.validStates.has(request.machineState)) {
      return {
        verdict: CommandVerdict.DENY,
        action: request.action,
        deviceId: request.deviceId,
        reason: "INVALID_MACHINE_STATE",
        timestamp,
        decisionId
      };
    }

    // Gate 4: Operator Approval Gate
    if (this.operatorApprovalRequired && !request.operatorApproved) {
      return {
        verdict: CommandVerdict.DENY,
        action: request.action,
        deviceId: request.deviceId,
        reason: "OPERATOR_APPROVAL_REQUIRED",
        timestamp,
        decisionId
      };
    }

    // Gate 5: Physical Interlocks
    if (request.interlocksClear === false) {
      return {
        verdict: CommandVerdict.DENY,
        action: request.action,
        deviceId: request.deviceId,
        reason: "INTERLOCK_ACTIVE",
        timestamp,
        decisionId
      };
    }

    // Gate 6: Nonce & Replay Defense
    if (request.nonce) {
      if (this.seenNonces.has(request.nonce)) {
        return {
          verdict: CommandVerdict.DENY,
          action: request.action,
          deviceId: request.deviceId,
          reason: "REPLAY_DETECTED",
          timestamp,
          decisionId
        };
      }
      this.seenNonces.add(request.nonce);

      // Keep seenNonces bounded
      if (this.seenNonces.size > 5000) {
        const first = this.seenNonces.values().next().value;
        if (first) this.seenNonces.delete(first);
      }
    }

    // Gate 7: TTL Window
    if (request.timestamp && Math.abs(timestamp - request.timestamp) > this.ttlSeconds) {
      return {
        verdict: CommandVerdict.DENY,
        action: request.action,
        deviceId: request.deviceId,
        reason: "EXPIRED_TTL",
        timestamp,
        decisionId
      };
    }

    // All gates passed
    return {
      verdict: CommandVerdict.APPROVE,
      action: request.action,
      deviceId: request.deviceId,
      reason: "POLICY_CHECKS_PASSED",
      timestamp,
      decisionId
    };
  }
}
