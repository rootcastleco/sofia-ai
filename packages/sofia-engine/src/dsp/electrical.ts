/**
 * Electrical power quality and 3-phase symmetrical components.
 * Ported from Rootcastle REI SignalLab.
 */

export interface PowerMetrics {
  vRms: number;
  iRms: number;
  activePowerW: number;
  reactivePowerVar: number;
  apparentPowerVa: number;
  powerFactor: number;
  thdVPct: number;
  thdIPct: number;
  fundamentalFreqHz: number;
}

export interface SymmetricalComponents {
  v0ZeroSeqV: number;
  v1PosSeqV: number;
  v2NegSeqV: number;
  vufPercent: number; // Voltage Unbalance Factor
}

export interface PowerQualityEvent {
  eventType: "VOLTAGE_SAG" | "VOLTAGE_SWELL" | "INTERRUPTION" | "INRUSH_CURRENT" | "NORMAL";
  severity: "normal" | "warning" | "alarm";
  description: string;
}

export function computeThd(signal: ArrayLike<number>, fs: number, fundFreq = 50.0): number {
  const n = signal.length;
  if (n < 16) return 0;

  // Simple DFT magnitude for harmonics 1 to 50
  let h1Mag = 0;
  let harmonicSumSq = 0;
  const twoPi = 2 * Math.PI;

  for (let h = 1; h <= 50; h++) {
    const f = h * fundFreq;
    if (f >= fs / 2) break;

    let real = 0;
    let imag = 0;
    const angleStep = (twoPi * f) / fs;

    for (let i = 0; i < n; i++) {
      const v = signal[i]!;
      const angle = angleStep * i;
      real += v * Math.cos(angle);
      imag -= v * Math.sin(angle);
    }

    const mag = Math.sqrt(real * real + imag * imag) / n;
    if (h === 1) {
      h1Mag = mag;
    } else {
      harmonicSumSq += mag * mag;
    }
  }

  if (h1Mag < 1e-6) return 0;
  return Math.min(500.0, (Math.sqrt(harmonicSumSq) / h1Mag) * 100.0);
}

export function computePowerMetrics(
  vSig: ArrayLike<number>,
  iSig: ArrayLike<number>,
  fs: number,
  nomFreq = 50.0
): PowerMetrics {
  const n = Math.min(vSig.length, iSig.length);
  if (n === 0) throw new Error("Signals cannot be empty");

  let sumVSq = 0;
  let sumISq = 0;
  let sumVI = 0;

  for (let i = 0; i < n; i++) {
    const v = vSig[i]!;
    const cur = iSig[i]!;
    sumVSq += v * v;
    sumISq += cur * cur;
    sumVI += v * cur;
  }

  const vRms = Math.sqrt(sumVSq / n);
  const iRms = Math.sqrt(sumISq / n);
  const activePowerW = sumVI / n;
  const apparentPowerVa = vRms * iRms;

  const reactivePowerVar =
    apparentPowerVa >= Math.abs(activePowerW)
      ? Math.sqrt(Math.max(0, apparentPowerVa * apparentPowerVa - activePowerW * activePowerW))
      : 0;

  const powerFactor =
    apparentPowerVa > 1e-6
      ? Math.max(-1, Math.min(1, activePowerW / apparentPowerVa))
      : 1.0;

  const thdVPct = computeThd(vSig, fs, nomFreq);
  const thdIPct = computeThd(iSig, fs, nomFreq);

  return {
    vRms: Math.round(vRms * 1000) / 1000,
    iRms: Math.round(iRms * 1000) / 1000,
    activePowerW: Math.round(activePowerW * 100) / 100,
    reactivePowerVar: Math.round(reactivePowerVar * 100) / 100,
    apparentPowerVa: Math.round(apparentPowerVa * 100) / 100,
    powerFactor: Math.round(powerFactor * 10000) / 10000,
    thdVPct: Math.round(thdVPct * 100) / 100,
    thdIPct: Math.round(thdIPct * 100) / 100,
    fundamentalFreqHz: nomFreq
  };
}

export function computeSymmetricalComponents(
  vaAmp: number,
  vaPhaseDeg: number,
  vbAmp: number,
  vbPhaseDeg: number,
  vcAmp: number,
  vcPhaseDeg: number
): SymmetricalComponents {
  const toRad = Math.PI / 180;
  const aAngle = 120 * toRad;
  const aR = Math.cos(aAngle);
  const aI = Math.sin(aAngle);
  const a2R = Math.cos(2 * aAngle);
  const a2I = Math.sin(2 * aAngle);

  // Phasors: Va, Vb, Vc
  const vaR = vaAmp * Math.cos(vaPhaseDeg * toRad);
  const vaI = vaAmp * Math.sin(vaPhaseDeg * toRad);
  const vbR = vbAmp * Math.cos(vbPhaseDeg * toRad);
  const vbI = vbAmp * Math.sin(vbPhaseDeg * toRad);
  const vcR = vcAmp * Math.cos(vcPhaseDeg * toRad);
  const vcI = vcAmp * Math.sin(vcPhaseDeg * toRad);

  // V0 = (Va + Vb + Vc) / 3
  const v0R = (vaR + vbR + vcR) / 3;
  const v0I = (vaI + vbI + vcI) / 3;
  const v0 = Math.sqrt(v0R * v0R + v0I * v0I);

  // V1 = (Va + a*Vb + a^2*Vc) / 3
  const v1R = (vaR + (aR * vbR - aI * vbI) + (a2R * vcR - a2I * vcI)) / 3;
  const v1I = (vaI + (aR * vbI + aI * vbR) + (a2R * vcI + a2I * vcR)) / 3;
  const v1 = Math.sqrt(v1R * v1R + v1I * v1I);

  // V2 = (Va + a^2*Vb + a*Vc) / 3
  const v2R = (vaR + (a2R * vbR - a2I * vbI) + (aR * vcR - aI * vcI)) / 3;
  const v2I = (vaI + (a2R * vbI + a2I * vbR) + (aR * vcI + aI * vcR)) / 3;
  const v2 = Math.sqrt(v2R * v2R + v2I * v2I);

  const vuf = v1 > 1e-6 ? (v2 / v1) * 100 : 0;

  return {
    v0ZeroSeqV: Math.round(v0 * 100) / 100,
    v1PosSeqV: Math.round(v1 * 100) / 100,
    v2NegSeqV: Math.round(v2 * 100) / 100,
    vufPercent: Math.round(vuf * 100) / 100
  };
}

export function detectPowerEvents(
  vSig: ArrayLike<number>,
  iSig: ArrayLike<number>,
  vNomRms = 230.0
): PowerQualityEvent[] {
  const events: PowerQualityEvent[] = [];
  const n = vSig.length;
  if (n === 0) return events;

  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += vSig[i]! * vSig[i]!;
  const vRmsActual = Math.sqrt(sumSq / n);

  const ratio = vNomRms > 0 ? vRmsActual / vNomRms : 1.0;
  if (ratio < 0.1) {
    events.push({
      eventType: "INTERRUPTION",
      severity: "alarm",
      description: `Voltage Interruption (${vRmsActual.toFixed(1)}V RMS, <10% nominal)`
    });
  } else if (ratio < 0.9) {
    events.push({
      eventType: "VOLTAGE_SAG",
      severity: "warning",
      description: `Voltage Sag (${vRmsActual.toFixed(1)}V RMS, ${(ratio * 100).toFixed(1)}% nominal)`
    });
  } else if (ratio > 1.1) {
    events.push({
      eventType: "VOLTAGE_SWELL",
      severity: "warning",
      description: `Voltage Swell (${vRmsActual.toFixed(1)}V RMS, ${(ratio * 100).toFixed(1)}% nominal)`
    });
  }

  if (iSig && iSig.length > 0) {
    let sumISq = 0;
    let iPeak = 0;
    for (let i = 0; i < iSig.length; i++) {
      const cur = Math.abs(iSig[i]!);
      sumISq += cur * cur;
      if (cur > iPeak) iPeak = cur;
    }
    const iRms = Math.sqrt(sumISq / iSig.length);
    if (iRms > 0 && iPeak / iRms > 3.0) {
      events.push({
        eventType: "INRUSH_CURRENT",
        severity: "warning",
        description: `Inrush Current / Peak-to-RMS ratio: ${(iPeak / iRms).toFixed(2)}`
      });
    }
  }

  if (events.length === 0) {
    events.push({
      eventType: "NORMAL",
      severity: "normal",
      description: `Normal Power Quality (${vRmsActual.toFixed(1)}V RMS, nominal)`
    });
  }

  return events;
}
