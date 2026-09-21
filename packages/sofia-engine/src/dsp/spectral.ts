/**
 * Spectral estimation and frequency-domain analysis.
 * Implements Welch PSD with Parseval energy conservation.
 */

/**
 * In-place Radix-2 Cooley-Tukey Fast Fourier Transform.
 * Length must be a power of two.
 */
export function fft(real: Float64Array, imag: Float64Array): void {
  const n = real.length;
  if (n <= 1) return;
  if ((n & (n - 1)) !== 0) {
    throw new Error(`FFT length must be a power of 2, received ${n}`);
  }

  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      const tempR = real[i]!;
      real[i] = real[j]!;
      real[j] = tempR;

      const tempI = imag[i]!;
      imag[i] = imag[j]!;
      imag[j] = tempI;
    }
    let k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  // Butterfly computations
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wStepR = Math.cos(angle);
    const wStepI = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let wR = 1;
      let wI = 0;

      for (let k = 0; k < halfLen; k++) {
        const uR = real[i + k]!;
        const uI = imag[i + k]!;

        const vR = real[i + k + halfLen]! * wR - imag[i + k + halfLen]! * wI;
        const vI = real[i + k + halfLen]! * wI + imag[i + k + halfLen]! * wR;

        real[i + k] = uR + vR;
        imag[i + k] = uI + vI;

        real[i + k + halfLen] = uR - vR;
        imag[i + k + halfLen] = uI - vI;

        const nextWR = wR * wStepR - wI * wStepI;
        wI = wR * wStepI + wI * wStepR;
        wR = nextWR;
      }
    }
  }
}

export function createHannWindow(length: number): Float64Array {
  const w = new Float64Array(length);
  for (let i = 0; i < length; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
  }
  return w;
}

export interface PsdResult {
  frequencies: Float64Array;
  psd: Float64Array;
  df: number;
}

/**
 * Welch's averaged modified periodogram with Parseval power conservation.
 */
export function welchPsd(
  values: ArrayLike<number>,
  sampleRate: number,
  segmentLength = 512,
  overlap = 256
): PsdResult {
  // Ensure power of two
  let nfft = 1;
  while (nfft < segmentLength) nfft <<= 1;

  const window = createHannWindow(segmentLength);
  let winSumSq = 0;
  for (let i = 0; i < segmentLength; i++) {
    winSumSq += window[i]! * window[i]!;
  }

  const step = segmentLength - overlap;
  const numSegments = Math.max(1, Math.floor((values.length - segmentLength) / step) + 1);

  const numBins = (nfft >> 1) + 1;
  const psdAccum = new Float64Array(numBins);
  const real = new Float64Array(nfft);
  const imag = new Float64Array(nfft);

  for (let seg = 0; seg < numSegments; seg++) {
    const offset = seg * step;
    real.fill(0);
    imag.fill(0);

    for (let i = 0; i < segmentLength; i++) {
      if (offset + i < values.length) {
        real[i] = values[offset + i]! * window[i]!;
      }
    }

    fft(real, imag);

    // Periodogram scaling
    const scale = 1 / (sampleRate * winSumSq);

    // DC bin
    psdAccum[0] = psdAccum[0]! + (real[0]! * real[0]! + imag[0]! * imag[0]!) * scale;

    // Positive bins (double for one-sided spectrum)
    for (let k = 1; k < numBins - 1; k++) {
      const magSq = real[k]! * real[k]! + imag[k]! * imag[k]!;
      psdAccum[k] = psdAccum[k]! + 2 * magSq * scale;
    }

    // Nyquist bin
    const nyq = numBins - 1;
    psdAccum[nyq] = psdAccum[nyq]! + (real[nyq]! * real[nyq]! + imag[nyq]! * imag[nyq]!) * scale;
  }

  const psd = new Float64Array(numBins);
  const frequencies = new Float64Array(numBins);
  const df = sampleRate / nfft;

  for (let k = 0; k < numBins; k++) {
    psd[k] = psdAccum[k]! / numSegments;
    frequencies[k] = k * df;
  }

  return { frequencies, psd, df };
}

export function spectralCentroid(frequencies: Float64Array, psd: Float64Array): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < frequencies.length; i++) {
    const p = psd[i]!;
    num += frequencies[i]! * p;
    den += p;
  }
  return den === 0 ? 0 : num / den;
}

export function spectralSpread(frequencies: Float64Array, psd: Float64Array, centroid?: number): number {
  const c = centroid ?? spectralCentroid(frequencies, psd);
  let num = 0;
  let den = 0;
  for (let i = 0; i < frequencies.length; i++) {
    const p = psd[i]!;
    const diff = frequencies[i]! - c;
    num += diff * diff * p;
    den += p;
  }
  return den === 0 ? 0 : Math.sqrt(num / den);
}

export function spectralFlatness(psd: Float64Array): number {
  let logSum = 0;
  let sum = 0;
  const n = psd.length;
  if (n === 0) return 1.0;

  for (let i = 0; i < n; i++) {
    const p = Math.max(psd[i]!, 1e-12);
    logSum += Math.log(p);
    sum += p;
  }

  const geometricMean = Math.exp(logSum / n);
  const arithmeticMean = sum / n;
  if (arithmeticMean === 0) return 0;
  return geometricMean / arithmeticMean;
}

export function bandEnergy(
  frequencies: Float64Array,
  psd: Float64Array,
  fLow: number,
  fHigh: number
): number {
  let energy = 0;
  for (let i = 0; i < frequencies.length; i++) {
    const f = frequencies[i]!;
    if (f >= fLow && f <= fHigh) {
      energy += psd[i]!;
    }
  }
  return energy;
}
