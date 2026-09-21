/**
 * Advanced Quantum Computing Emulator in TypeScript.
 * Pure complex statevector simulation in C^(2^n).
 */

export class QuantumCircuit {
  readonly numQubits: number;
  readonly dim: number;
  private readonly real: Float64Array;
  private readonly imag: Float64Array;

  constructor(numQubits: number) {
    if (numQubits < 1 || numQubits > 16) {
      throw new Error(`numQubits must be in 1..16, received ${numQubits}`);
    }
    this.numQubits = numQubits;
    this.dim = 1 << numQubits;
    this.real = new Float64Array(this.dim);
    this.imag = new Float64Array(this.dim);
    this.real[0] = 1.0; // Ground state |0...0>
  }

  get probabilities(): Float64Array {
    const probs = new Float64Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      const r = this.real[i]!;
      const im = this.imag[i]!;
      probs[i] = r * r + im * im;
    }
    return probs;
  }

  h(qubit: number): this {
    const stride = 1 << qubit;
    const invSqrt2 = 1.0 / Math.SQRT2;

    for (let i = 0; i < this.dim; i += 2 * stride) {
      for (let j = 0; j < stride; j++) {
        const idx0 = i + j;
        const idx1 = idx0 + stride;

        const r0 = this.real[idx0]!;
        const i0 = this.imag[idx0]!;
        const r1 = this.real[idx1]!;
        const i1 = this.imag[idx1]!;

        this.real[idx0] = (r0 + r1) * invSqrt2;
        this.imag[idx0] = (i0 + i1) * invSqrt2;
        this.real[idx1] = (r0 - r1) * invSqrt2;
        this.imag[idx1] = (i0 - i1) * invSqrt2;
      }
    }
    return this;
  }

  x(qubit: number): this {
    const stride = 1 << qubit;
    for (let i = 0; i < this.dim; i += 2 * stride) {
      for (let j = 0; j < stride; j++) {
        const idx0 = i + j;
        const idx1 = idx0 + stride;

        const r0 = this.real[idx0]!;
        const i0 = this.imag[idx0]!;
        this.real[idx0] = this.real[idx1]!;
        this.imag[idx0] = this.imag[idx1]!;
        this.real[idx1] = r0;
        this.imag[idx1] = i0;
      }
    }
    return this;
  }

  z(qubit: number): this {
    const stride = 1 << qubit;
    for (let i = 0; i < this.dim; i += 2 * stride) {
      for (let j = 0; j < stride; j++) {
        const idx1 = i + j + stride;
        this.real[idx1] = -this.real[idx1]!;
        this.imag[idx1] = -this.imag[idx1]!;
      }
    }
    return this;
  }

  ry(qubit: number, theta: number): this {
    const stride = 1 << qubit;
    const c = Math.cos(theta / 2);
    const s = Math.sin(theta / 2);

    for (let i = 0; i < this.dim; i += 2 * stride) {
      for (let j = 0; j < stride; j++) {
        const idx0 = i + j;
        const idx1 = idx0 + stride;

        const r0 = this.real[idx0]!;
        const i0 = this.imag[idx0]!;
        const r1 = this.real[idx1]!;
        const i1 = this.imag[idx1]!;

        this.real[idx0] = c * r0 - s * r1;
        this.imag[idx0] = c * i0 - s * i1;
        this.real[idx1] = s * r0 + c * r1;
        this.imag[idx1] = s * i0 + c * i1;
      }
    }
    return this;
  }

  cx(control: number, target: number): this {
    const cMask = 1 << control;
    const tMask = 1 << target;

    for (let i = 0; i < this.dim; i++) {
      if ((i & cMask) !== 0 && (i & tMask) === 0) {
        const pair = i ^ tMask;
        const r0 = this.real[i]!;
        const i0 = this.imag[i]!;
        this.real[i] = this.real[pair]!;
        this.imag[i] = this.imag[pair]!;
        this.real[pair] = r0;
        this.imag[pair] = i0;
      }
    }
    return this;
  }

  expectationZ(qubit: number): number {
    const qMask = 1 << qubit;
    let expVal = 0;
    const probs = this.probabilities;
    for (let i = 0; i < this.dim; i++) {
      const sign = (i & qMask) !== 0 ? -1.0 : 1.0;
      expVal += sign * probs[i]!;
    }
    return expVal;
  }

  measure(shots = 1024): Record<string, number> {
    const probs = this.probabilities;
    const counts: Record<string, number> = {};

    for (let s = 0; s < shots; s++) {
      const rand = Math.random();
      let cumulative = 0;
      let selectedIdx = this.dim - 1;

      for (let i = 0; i < this.dim; i++) {
        cumulative += probs[i]!;
        if (rand < cumulative) {
          selectedIdx = i;
          break;
        }
      }

      const bitstring = selectedIdx.toString(2).padStart(this.numQubits, "0");
      counts[bitstring] = (counts[bitstring] ?? 0) + 1;
    }
    return counts;
  }

  getState(): { real: Float64Array; imag: Float64Array } {
    return { real: this.real.slice(), imag: this.imag.slice() };
  }
}

export function angleEncodingCircuit(features: ArrayLike<number>): QuantumCircuit {
  const n = features.length;
  const qc = new QuantumCircuit(Math.min(n, 16));
  for (let i = 0; i < qc.numQubits; i++) {
    qc.ry(i, 2.0 * features[i]!);
  }
  return qc;
}

export function computeQuantumFidelity(a: QuantumCircuit, b: QuantumCircuit): number {
  if (a.numQubits !== b.numQubits) throw new Error("Qubit counts must match");
  const stateA = a.getState();
  const stateB = b.getState();

  let realProd = 0;
  let imagProd = 0;

  for (let i = 0; i < a.dim; i++) {
    const rA = stateA.real[i]!;
    const iA = stateA.imag[i]!;
    const rB = stateB.real[i]!;
    const iB = stateB.imag[i]!;

    // Conjugate product (rA - j*iA) * (rB + j*iB)
    realProd += rA * rB + iA * iB;
    imagProd += rA * iB - iA * rB;
  }

  return realProd * realProd + imagProd * imagProd;
}

export class QuantumKernel {
  readonly numQubits: number;

  constructor(numQubits: number) {
    this.numQubits = Math.min(numQubits, 16);
  }

  evaluate(x: ArrayLike<number>, y: ArrayLike<number>): number {
    const qcX = angleEncodingCircuit(x);
    const qcY = angleEncodingCircuit(y);
    return computeQuantumFidelity(qcX, qcY);
  }
}
