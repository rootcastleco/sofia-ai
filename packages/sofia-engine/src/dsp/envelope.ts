/**
 * Demodulated envelope analysis via Hilbert transform.
 */

import { fft } from "./spectral.js";

/**
 * Computes the analytic signal envelope of a real-valued time-series.
 */
export function envelope(values: ArrayLike<number>): Float64Array {
  const n = values.length;
  if (n === 0) return new Float64Array(0);

  // Next power of two
  let nfft = 1;
  while (nfft < n) nfft <<= 1;

  const real = new Float64Array(nfft);
  const imag = new Float64Array(nfft);

  for (let i = 0; i < n; i++) {
    real[i] = values[i]!;
  }

  // Forward FFT
  fft(real, imag);

  // Construct analytic signal in frequency domain:
  // H(0) = 1, H(Nyquist) = 1, H(1..Nyq-1) = 2, H(Nyq+1..N-1) = 0
  const half = nfft >> 1;
  for (let i = 1; i < half; i++) {
    real[i] = real[i]! * 2;
    imag[i] = imag[i]! * 2;
  }
  for (let i = half + 1; i < nfft; i++) {
    real[i] = 0;
    imag[i] = 0;
  }

  // Inverse FFT (via forward FFT on conjugate, divided by N)
  for (let i = 0; i < nfft; i++) {
    imag[i] = -imag[i]!;
  }
  fft(real, imag);

  const env = new Float64Array(n);
  const invN = 1 / nfft;

  for (let i = 0; i < n; i++) {
    const r = real[i]! * invN;
    const im = -imag[i]! * invN;
    env[i] = Math.sqrt(r * r + im * im);
  }

  return env;
}
