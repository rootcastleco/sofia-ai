# Embedded export boundary

Sofia Engine does **not** run the Python stack on a microcontroller. Instead it
defines an explicit export boundary: a feature contract that is implemented twice
— once in Python (gateway/workstation) and once in portable C99 (MCU) — with
golden vectors proving the two agree.

```
  Python:  sofia_ai/features/statistical.py   ── FeatureVector contract ──┐
                                                                          │
  C99:     embedded/src/sofia_features.c      ── same contract ───────────┤
                                                                          │
  Binding: embedded/tests/golden_vectors.{h,json}  ◄── tools/gen_golden_vectors.py
```

## What is in this directory

| Path | Purpose |
|---|---|
| `src/sofia_features.h/.c` | Portable C99 feature runtime (reference) |
| `src/sofia_fixed.h` | Header-only Q16.16 fixed-point helpers |
| `tests/test_sofia_features.c` | Host test: closed-form maths + embedded rules |
| `tests/golden_vectors.h/.json` | Generated Python↔C binding vectors |
| `Makefile`, `CMakeLists.txt` | Host build only |

## Embedded engineering rules honoured

* No heap allocation anywhere. Every function writes into caller-owned storage.
* No recursion. All loops are bounded by a caller-supplied length or a compile-time constant.
* No global mutable state. The `sofia_workspace_t` is explicit and caller-owned.
* All externally supplied lengths are range-checked before use.
* Every division is guarded against a negligible denominator.
* Non-finite input is rejected by `sofia_features_validate()` before any maths.
* Fixed-point conversion saturates rather than wrapping.

## Concurrency assumptions

This module is **not** thread-safe and **not** ISR-safe. A single workspace must
be owned by one execution context. If an ISR fills the sample buffer, hand
ownership to the application context (double buffer / ring buffer with explicit
producer-consumer ownership) before calling any function here.

## Building and testing on the host

```bash
cd embedded
make test
# or
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

Regenerate the binding vectors after any change to the Python feature contract:

```bash
python tools/gen_golden_vectors.py
```

## Target status

Honest status per target. Nothing here is claimed unless it was actually built.

| Target | Status | Notes |
|---|---|---|
| Host (Linux/macOS/Windows, C99) | **PARTIALLY VERIFIED** | Source is complete and the test suite is written; no C compiler was available in the environment that produced this commit, so the host build was **NOT EXECUTED** here. Run `make test` to verify. |
| STM32 Cortex-M (CMSIS-DSP) | **NOT VERIFIED** | No ARM toolchain or hardware in CI. Mapping guidance below is documentation only. |
| ESP32 (ESP-IDF / Xtensa, RISC-V) | **NOT VERIFIED** | No ESP-IDF installation or hardware in CI. |
| TensorFlow Lite Micro | **NOT VERIFIED** | No TFLM build or conversion is included in this repository. |
| Arduino / other MCU | **NOT VERIFIED** | Out of scope. |

## CMSIS-DSP mapping guidance (documentation only)

Where an ARM CMSIS-DSP library is available, these substitutions reduce code size
and use hand-optimised kernels. They are **not** wired into this repository and
have not been compiled:

| Sofia function | CMSIS-DSP equivalent | Notes |
|---|---|---|
| `sofia_mean` | `arm_mean_f32` | |
| `sofia_rms` | `arm_rms_f32` | |
| `sofia_variance` / `sofia_std` | `arm_var_f32` | CMSIS uses population variance, matching Sofia |
| `sofia_peak` | `arm_max_abs_f32` | |
| `sofia_energy` (sum of squares) | `arm_power_f32` | |
| `sofia_spectral_features` | `arm_rfft_fast_f32` + `arm_cmplx_mag_f32` | Requires an initialised instance and a power-of-two length |
| matrix ops | `arm_mat_*_f32` | For multivariate detectors |

When substituting, keep Sofia's validation boundary: CMSIS kernels do not check
for NaN/Inf, so `sofia_features_validate()` must still run first.

## TinyML inference guidance (documentation only)

Sofia's model boundary is a `FeatureVector` with a fixed, named ordering
(`sofia_ai.features.extractor.FEATURE_NAMES`). To deploy a trained model:

1. Train with any framework; export to ONNX.
2. Convert to TensorFlow Lite, then to a TFLite Micro C array
   (`xxd -i model.tflite > model_data.cc`).
3. Feed the model the *same* feature ordering the Python extractor produces —
   order is part of the contract, and reordering inputs is a common source of
   silently wrong inference.
4. Record the model version and preprocessing version on-device; Sofia's
   manifest discipline applies at the edge too.

None of these steps are implemented or tested here.

## What this directory deliberately does not do

* It does not perform inference. Feature extraction only.
* It does not implement transport protocols (MQTT/Modbus/CAN) — those live on a
  gateway.
* It does not claim any functional-safety property.
