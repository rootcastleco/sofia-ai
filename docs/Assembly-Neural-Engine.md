# In-Situ Assembly Neural Self-Training Engine

> **Rootcastle Engineering & Innovation** | Specification Trace: `SOFIA-LRN-001` - `004`

---

## 1. Architectural Motivation

Modern edge and microcontroller intelligence typically suffers from two critical bottlenecks:
1. **Framework Bloat**: Heavy C++/Python frameworks (PyTorch, TensorFlow Lite, ONNX Runtime) introduce megabytes of binary footprint, dynamic memory allocation, non-deterministic garbage collection, and extensive dependency chains.
2. **Read-Only / Static Models**: Almost all edge ML engines are strictly inference-only. Adapting a model to a machine's unique physical drift, bearing wear-in, or seasonal process shifts requires sending raw telemetry to the cloud for retraining.

**Sofia Engine** solves both challenges with an **In-Situ Assembly Virtual Machine & Neural Engine (`sofia_ai.learning.asm`)**:
* **Deterministic Execution**: Operates over a fixed 64-bit float memory array with 16 hardware-like registers (`R0`-`R7`, `ACC`, `LR`, `ERR`, `PC`, `SP`).
* **Self-Training Bytecode**: Forward inference, MSE loss computation, analytical error backpropagation, and SGD weight updates are compiled directly into virtual assembly instructions.
* **Native Code Emitters**: Translates neural models into native x86_64 AVX2, ARM Cortex-M Thumb-2 SIMD, and WebAssembly (WAT).

---

## 2. Assembly Virtual Machine Architecture

```
+---------------------------------------------------------------------------------------+
|                                  SofiaAsmVM Architecture                              |
+---------------------------------------------------------------------------------------+
|  REGISTERS:                                                                           |
|  [ R0 - R7 ]   General-purpose 64-bit float registers                                |
|  [ ACC ]       Accumulator (stores arithmetic results & dot products)                 |
|  [ LR ]        Learning Rate register                                                 |
|  [ ERR ]       Error / Loss gradient register                                         |
|  [ PC ]        Program Counter                                                        |
|  [ SP ]        Stack Pointer                                                          |
|                                                                                       |
|  MEMORY MAP (Fixed Float64Array):                                                     |
|  0x0000 - 0x003F : Input Vector [X]                                                   |
|  0x0040 - 0x00FF : Hidden Layer Activations [H]                                       |
|  0x0100 - 0x013F : Output Predictions [Y_hat]                                         |
|  0x0140 - 0x017F : Ground Truth Target [Y]                                            |
|  0x0180 - 0x0FFF : Layer 1 Weights & Biases [W1, B1]                                  |
|  0x1000 - 0x1FFF : Layer 2 Weights & Biases [W2, B2]                                  |
|  0x2000 - 0x2FFF : Output & Hidden Gradients [dY, dH, dW1, dW2]                       |
+---------------------------------------------------------------------------------------+
```

### Supported Opcodes

| Opcode | Mnemonic | Operands | Operation |
|---|---|---|---|
| `0x01` | `LOAD_CONST` | `R_dest, imm` | Load immediate float into register |
| `0x02` | `LOAD_MEM` | `R_dest, R_addr` | Load float from memory address |
| `0x03` | `STORE_MEM` | `R_addr, R_src` | Store register float into memory |
| `0x10` | `ADD` | `R_dest, R_src` | `dest = dest + src; ACC = dest` |
| `0x14` | `FMA` | `R_dest, R_s1, R_s2`| `dest = dest + (s1 * s2); ACC = dest` |
| `0x20` | `VEC_DOT` | `R_addrA, R_addrB, R_len` | `ACC = sum(mem[A+i] * mem[B+i])` |
| `0x21` | `VEC_FMA` | `R_dest, R_src, R_scalar, len` | `mem[dest+i] += mem[src+i] * scalar` |
| `0x22` | `VEC_SUB` | `R_dest, R_a, R_b, len` | `mem[dest+i] = mem[a+i] - mem[b+i]` |
| `0x30` | `ACT_RELU` | `R_addr, len` | `mem[addr+i] = max(0, mem[addr+i])` |
| `0x40` | `UPDATE_SGD` | `R_w, R_grad, R_lr, len` | `mem[w+i] -= lr * mem[grad+i]` |
| `0x41` | `COMPUTE_MSE`| `R_pred, R_target, len` | `ERR = mean((pred - target)^2)` |
| `0xFF` | `HALT` | None | Terminates program execution |

---

## 3. Mathematical Formulation of Self-Training

The model implements a 2-layer perceptron with in-situ backpropagation:

1. **Forward Pass**:
   $$h_j = \text{ReLU}\left(\sum_{i} W^{(1)}_{ji} x_i + b^{(1)}_j\right)$$
   $$\hat{y}_k = \sum_{j} W^{(2)}_{kj} h_j + b^{(2)}_k$$

2. **MSE Loss & Output Gradient**:
   $$L = \frac{1}{K} \sum_{k=1}^K (\hat{y}_k - y_k)^2$$
   $$\delta^{(2)}_k = \frac{\partial L}{\partial \hat{y}_k} = \hat{y}_k - y_k$$

3. **Weight Gradients & SGD Update**:
   $$\frac{\partial L}{\partial W^{(2)}_{kj}} = \delta^{(2)}_k \cdot h_j$$
   $$W^{(2)}_{kj} \leftarrow W^{(2)}_{kj} - \eta \cdot \frac{\partial L}{\partial W^{(2)}_{kj}}$$

All steps execute sequentially inside `SofiaAsmVM.run()`.

---

## 4. Python Usage Example

```python
import numpy as np
from sofia_ai.learning import AssemblyNeuralNetwork

# Initialize network: 4 inputs -> 8 hidden -> 2 outputs
model = AssemblyNeuralNetwork(
    input_dim=4,
    hidden_dim=8,
    output_dim=2,
    learning_rate=0.02,
    seed=42
)

# Training data (e.g. vibration RMS, peak, kurtosis, crest factor)
x_train = np.array([1.2, 3.4, 2.9, 4.1])
y_target = np.array([0.05, 0.95]) # Normal vs Anomaly

# On-device training loop executed purely in assembly VM bytecode
print("Starting in-situ assembly training...")
for epoch in range(100):
    loss = model.train_step(x_train, y_target)
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Assembly MSE Loss: {loss:.6f}")

# Inference
prediction = model.forward(x_train)
print("Inference Result:", prediction)

# Generate bare-metal assembly
print("\n--- Native ARM Cortex-M Thumb-2 Code ---")
print(model.emit_arm_assembly())
```

---

## 5. TypeScript / Node.js Usage

```typescript
import { AssemblyNeuralNetwork } from "@rootcastle/sofia-engine";

const model = new AssemblyNeuralNetwork(3, 6, 1, 0.05);

const x = [0.5, 1.2, -0.3];
const y = [1.0];

for (let i = 0; i < 50; i++) {
  const loss = model.trainStep(x, y);
  if (i % 10 === 0) {
    console.log(`Step ${i}: Loss = ${loss}`);
  }
}

const pred = model.forward(x);
console.log("Prediction:", pred);
```
