/**
 * Time-domain statistical feature extractors.
 * Pure mathematical functions for vibration & telemetry signals.
 */

export function mean(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += values[i]!;
  }
  return sum / n;
}

export function rms(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) return 0;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i]!;
    sumSq += v * v;
  }
  return Math.sqrt(sumSq / n);
}

export function peak(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) return 0;
  let max = 0;
  for (let i = 0; i < n; i++) {
    const abs = Math.abs(values[i]!);
    if (abs > max) max = abs;
  }
  return max;
}

export function peakToPeak(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) return 0;
  let min = values[0]!;
  let max = values[0]!;
  for (let i = 1; i < n; i++) {
    const v = values[i]!;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return max - min;
}

export function variance(values: ArrayLike<number>): number {
  const n = values.length;
  if (n < 2) return 0;
  const m = mean(values);
  let sumSqDiff = 0;
  for (let i = 0; i < n; i++) {
    const diff = values[i]! - m;
    sumSqDiff += diff * diff;
  }
  return sumSqDiff / (n - 1);
}

export function std(values: ArrayLike<number>): number {
  return Math.sqrt(variance(values));
}

export function skewness(values: ArrayLike<number>): number {
  const n = values.length;
  if (n < 3) return 0;
  const m = mean(values);
  const s = std(values);
  if (s === 0) return 0;

  let sumCube = 0;
  for (let i = 0; i < n; i++) {
    const diff = values[i]! - m;
    sumCube += diff * diff * diff;
  }
  return sumCube / (n * s * s * s);
}

export function kurtosis(values: ArrayLike<number>): number {
  const n = values.length;
  if (n < 4) return 0;
  const m = mean(values);
  const s = std(values);
  if (s === 0) return 0;

  let sumQuad = 0;
  for (let i = 0; i < n; i++) {
    const diff = values[i]! - m;
    sumQuad += diff * diff * diff * diff;
  }
  return sumQuad / (n * s * s * s * s);
}

export function crestFactor(values: ArrayLike<number>): number {
  const r = rms(values);
  if (r === 0) return 0;
  return peak(values) / r;
}

export function shapeFactor(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) return 0;
  let sumAbs = 0;
  for (let i = 0; i < n; i++) {
    sumAbs += Math.abs(values[i]!);
  }
  const meanAbs = sumAbs / n;
  if (meanAbs === 0) return 0;
  return rms(values) / meanAbs;
}

export function impulseFactor(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) return 0;
  let sumAbs = 0;
  for (let i = 0; i < n; i++) {
    sumAbs += Math.abs(values[i]!);
  }
  const meanAbs = sumAbs / n;
  if (meanAbs === 0) return 0;
  return peak(values) / meanAbs;
}

export function marginFactor(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) return 0;
  let sumSqrt = 0;
  for (let i = 0; i < n; i++) {
    sumSqrt += Math.sqrt(Math.abs(values[i]!));
  }
  const denom = (sumSqrt / n) * (sumSqrt / n);
  if (denom === 0) return 0;
  return peak(values) / denom;
}

export function zeroCrossingRate(values: ArrayLike<number>): number {
  const n = values.length;
  if (n < 2) return 0;
  let crossings = 0;
  for (let i = 1; i < n; i++) {
    if ((values[i]! >= 0 && values[i - 1]! < 0) || (values[i]! < 0 && values[i - 1]! >= 0)) {
      crossings++;
    }
  }
  return crossings / (n - 1);
}

export function extractTimeDomainFeatures(values: ArrayLike<number>): Record<string, number> {
  return {
    mean: mean(values),
    rms: rms(values),
    peak: peak(values),
    peak_to_peak: peakToPeak(values),
    variance: variance(values),
    std: std(values),
    skewness: skewness(values),
    kurtosis: kurtosis(values),
    crest_factor: crestFactor(values),
    shape_factor: shapeFactor(values),
    impulse_factor: impulseFactor(values),
    margin_factor: marginFactor(values),
    zero_crossing_rate: zeroCrossingRate(values)
  };
}
