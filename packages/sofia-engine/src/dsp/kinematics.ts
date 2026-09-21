/**
 * Rotating machinery kinematics: bearing fault frequencies & gear meshing.
 */

export interface BearingGeometry {
  /** Number of rolling elements */
  nb: number;
  /** Roller diameter (same unit as pitch diameter) */
  d: number;
  /** Pitch diameter (same unit as roller diameter) */
  D: number;
  /** Contact angle in degrees (default: 0) */
  alphaDeg?: number;
}

export interface BearingFrequencies {
  /** Ball Pass Frequency Outer Race (Hz) */
  bpfo: number;
  /** Ball Pass Frequency Inner Race (Hz) */
  bpfi: number;
  /** Ball Spin Frequency (Hz) */
  bsf: number;
  /** Fundamental Train Frequency (Cage) (Hz) */
  ftf: number;
}

export function calculateBearingFrequencies(
  shaftHz: number,
  geometry: BearingGeometry
): BearingFrequencies {
  const { nb, d, D, alphaDeg = 0 } = geometry;
  const alphaRad = (alphaDeg * Math.PI) / 180;
  const cosAlpha = Math.cos(alphaRad);
  const ratio = (d / D) * cosAlpha;

  const bpfo = (nb / 2) * shaftHz * (1 - ratio);
  const bpfi = (nb / 2) * shaftHz * (1 + ratio);
  const bsf = (D / (2 * d)) * shaftHz * (1 - ratio * ratio);
  const ftf = 0.5 * shaftHz * (1 - ratio);

  return { bpfo, bpfi, bsf, ftf };
}

export function calculateGearMeshFrequency(shaftHz: number, numberOfTeeth: number): number {
  return shaftHz * numberOfTeeth;
}
