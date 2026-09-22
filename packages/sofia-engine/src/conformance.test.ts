import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  mean,
  rms,
  peak,
  peakToPeak,
  crestFactor,
  shapeFactor,
  impulseFactor,
  marginFactor,
  zeroCrossingRate
} from "./dsp/statistics.js";
import { computeSymmetricalComponents } from "./dsp/electrical.js";
import { SofiaAsmVM, OpCode, Register, VMStatus, VMFault } from "./learning/asm.js";
import { DataQuality, SignalQuality } from "./contracts.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "../../..");

test("contracts: SignalQuality alias and DataQuality enums", () => {
  assert.equal(SignalQuality.GOOD, "GOOD");
  assert.equal(DataQuality.DEGRADED, "DEGRADED");
  assert.equal(DataQuality.SATURATED, "SATURATED");
  assert.equal(SignalQuality, DataQuality);
});

test("dsp: symmetrical components match golden vectors", () => {
  const goldenPath = path.join(ROOT, "tests/golden/dsp_golden.json");
  const golden = JSON.parse(fs.readFileSync(goldenPath, "utf-8"));
  const expected = golden.three_phase_unbalanced;

  const result = computeSymmetricalComponents(
    expected.va_amp,
    0.0,
    expected.vb_amp,
    -120.0,
    expected.vc_amp,
    120.0
  );

  assert.ok(Math.abs(result.v0ZeroSeqV - expected.expected_v0) < 0.05);
  assert.ok(Math.abs(result.v1PosSeqV - expected.expected_v1) < 0.05);
  assert.ok(Math.abs(result.v2NegSeqV - expected.expected_v2) < 0.05);
  assert.ok(Math.abs(result.vufPercent - expected.expected_vuf_percent) < 0.05);
});

test("dsp: statistical features match embedded golden vectors", () => {
  const goldenPath = path.join(ROOT, "embedded/tests/golden_vectors.json");
  const golden = JSON.parse(fs.readFileSync(goldenPath, "utf-8"));

  for (const c of golden.cases) {
    const samples = c.samples;
    const exp = c.expected;

    assert.ok(Math.abs(mean(samples) - exp.mean) < 1e-9, `${c.name}: mean`);
    assert.ok(Math.abs(rms(samples) - exp.rms) < 1e-9, `${c.name}: rms`);
    assert.ok(Math.abs(peak(samples) - exp.peak) < 1e-9, `${c.name}: peak`);
    assert.ok(Math.abs(peakToPeak(samples) - exp.peak_to_peak) < 1e-9, `${c.name}: peakToPeak`);
    assert.ok(Math.abs(crestFactor(samples) - exp.crest_factor) < 1e-9, `${c.name}: crestFactor`);
    assert.ok(Math.abs(shapeFactor(samples) - exp.shape_factor) < 1e-9, `${c.name}: shapeFactor`);
    assert.ok(Math.abs(impulseFactor(samples) - exp.impulse_factor) < 1e-9, `${c.name}: impulseFactor`);
    assert.ok(Math.abs(marginFactor(samples) - exp.margin_factor) < 1e-9, `${c.name}: marginFactor`);
    assert.ok(Math.abs(zeroCrossingRate(samples) - exp.zero_crossing_rate) < 1e-9, `${c.name}: zcr`);
  }
});

test("asm vm: hardening and cycle bounds", () => {
  const vm = new SofiaAsmVM(1024);

  // 1. Empty program returns FAULT / INVALID_PROGRAM
  const resEmpty = vm.run([]);
  assert.equal(resEmpty.status, VMStatus.FAULT);
  assert.equal(resEmpty.fault, VMFault.INVALID_PROGRAM);

  // 2. Halting program returns HALTED
  const programHalt = [
    { opcode: OpCode.LOAD_CONST, arg1: Register.R0, imm: 42.0 },
    { opcode: OpCode.HALT }
  ];
  const resHalt = vm.run(programHalt);
  assert.equal(resHalt.status, VMStatus.HALTED);
  assert.equal(resHalt.fault, VMFault.NONE);
  assert.equal(vm.getReg(Register.R0), 42.0);

  // 3. Infinite loop bounded by cycle limit
  const infiniteLoop = [
    { opcode: OpCode.LOAD_CONST, arg1: Register.R0, imm: 1.0 },
    { opcode: OpCode.JMP, imm: 0 }
  ];
  vm.reset();
  const resLoop = vm.run(infiniteLoop, 50);
  assert.equal(resLoop.status, VMStatus.CYCLE_LIMIT);
  assert.equal(resLoop.fault, VMFault.CYCLE_LIMIT);
  assert.equal(resLoop.cycles, 50);
});
