# Advanced Quantum Computing Emulation

Sofia Engine provides a mathematically sound, high-precision **Quantum Computing Emulator** (`sofia_ai.quantum`) that simulates pure complex statevectors in Hilbert space $\mathbb{C}^{2^n}$ using pure NumPy.

---

## 1. Mathematical Architecture

### Statevector Representation
A pure quantum state $|\psi\rangle$ of $n$ qubits is represented by a complex statevector of dimension $2^n$:

$$|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle, \quad \alpha_i \in \mathbb{C}, \quad \sum_{i=0}^{2^n-1} |\alpha_i|^2 = 1$$

### Universal Gate Library
Sofia's `QuantumCircuit` implements full unitary transformations:

* **Hadamard ($H$)**: Creates equal quantum superposition:
  $$H = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$
* **Pauli Gates ($X, Y, Z$)**:
  $$X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad Y = \begin{bmatrix} 0 & -j \\ j & 0 \end{bmatrix}, \quad Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$
* **Phase & T Gates ($S, T$)**:
  $$S = \begin{bmatrix} 1 & 0 \\ 0 & j \end{bmatrix}, \quad T = \begin{bmatrix} 1 & 0 \\ 0 & e^{j \pi/4} \end{bmatrix}$$
* **Parametric Rotations ($R_x, R_y, R_z$)**:
  $$R_y(\theta) = \begin{bmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{bmatrix}$$
* **Entangling Gates (CNOT / $CX$, Controlled-Z / $CZ$)**:
  $$CX = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$

---

## 2. Quantum Feature Maps & Quantum Kernels

### Angle Encoding
Encodes real-valued sensor features $\mathbf{x} = [x_1, \dots, x_n]$ into rotation angles of individual qubits:

$$|\mathbf{x}\rangle = \bigotimes_{i=1}^n \left[\cos(x_i)|0\rangle + \sin(x_i)|1\rangle\right]$$

### Quantum Kernel Estimation
For quantum support vector machines and non-linear anomaly detection, Sofia evaluates quantum state transition fidelity:

$$K(\mathbf{x}_1, \mathbf{x}_2) = |\langle \psi(\mathbf{x}_1) | \psi(\mathbf{x}_2) \rangle|^2 \in [0, 1]$$

Where $K(\mathbf{x}_1, \mathbf{x}_2) = 1.0$ indicates identical states and $0.0$ indicates orthogonal quantum states.

---

## 3. Python Usage

```python
from sofia_ai.quantum import QuantumCircuit, QuantumKernel

# 1. Simulate Bell State Entanglement
qc = QuantumCircuit(num_qubits=2)
qc.h(0).cx(0, 1)

# Measurement sampling (1024 shots)
shots = qc.measure(shots=1024)
print("Measurement counts:", shots) # {'00': ~512, '11': ~512}

# 2. Quantum Kernel Anomaly Scoring
kernel = QuantumKernel(num_qubits=4)
normal_baseline = [0.1, 0.2, 0.15, 0.18]
test_sample = [0.12, 0.22, 0.14, 0.19]

similarity = kernel.evaluate(normal_baseline, test_sample)
print(f"Quantum Kernel Similarity: {similarity:.4f}")
```

---

## 4. TypeScript Usage

```typescript
import { QuantumCircuit, QuantumKernel } from "@rootcastle/sofia-engine";

const qc = new QuantumCircuit(2);
qc.h(0).cx(0, 1);
const counts = qc.measure(1000);
console.log("Shots:", counts);

const kernel = new QuantumKernel(3);
const sim = kernel.evaluate([0.1, 0.5, 0.9], [0.1, 0.5, 0.9]);
console.log("Fidelity:", sim); // 1.0000
```
