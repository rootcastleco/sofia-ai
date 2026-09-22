# Sofia Engine Runtime v3 — Performance Budgets & Benchmark Standards

> **Author:** Rootcastle Engineering & Innovation  
> **Status:** Active Normative Standard

---

## 1. Target Hardware Profiles

1. **Profile A — Workstation / Server:** x86_64, $\ge 4$ cores, $\ge 8$ GB RAM.
2. **Profile B — Edge Linux Gateway:** ARM64 / Cortex-A53 / A72 (Raspberry Pi 4 / CM4, NXP i.MX8), 1–4 GB RAM.
3. **Profile C — Microcontroller:** ARM Cortex-M4 / Cortex-M7 (168–480 MHz), 128–512 KB RAM, 1–2 MB Flash.

---

## 2. Quantitative Performance Budgets

| Operation | Workstation (Profile A) | Edge Gateway (Profile B) | Microcontroller (Profile C) |
|---|---|---|---|
| **FFT / PSD (1024 samples)** | $\le 0.5$ ms | $\le 2.0$ ms | $\le 10.0$ ms (Q16.16) |
| **FFT / PSD (4096 samples)** | $\le 2.0$ ms | $\le 8.0$ ms | $\le 45.0$ ms (Q16.16) |
| **Hilbert Envelope (2048 samples)** | $\le 1.0$ ms | $\le 5.0$ ms | $\le 25.0$ ms |
| **Full Statistical Feature Extraction** | $\le 0.8$ ms | $\le 3.5$ ms | $\le 15.0$ ms |
| **Electrical Power & THD (50 harmonics)**| $\le 1.5$ ms | $\le 6.0$ ms | $\le 30.0$ ms |
| **Sofia VM Forward Inference (32$\to$64$\to$4)** | $\le 50$ $\mu$s | $\le 250$ $\mu$s | $\le 1500$ $\mu$s |
| **Sofia VM Training Step (Forward+Backprop+SGD)**| $\le 150$ $\mu$s | $\le 800$ $\mu$s | $\le 5000$ $\mu$s |
| **Package Import Time (`import sofia_ai`)**| $\le 150$ ms | $\le 500$ ms | N/A (Compiled C) |
| **Embedded Binary Footprint (Flash)** | N/A | N/A | $\le 64$ KB |
| **Embedded Static RAM Footprint** | N/A | N/A | $\le 32$ KB |

---

## 3. Benchmark Methodology & Reporting Rules

1. **Statistical Reporting:**
   Benchmarks must record and report:
   - Sample count ($N \ge 100$ iterations after warm-up).
   - Median ($p_{50}$), 95th percentile ($p_{95}$), and 99th percentile ($p_{99}$).
   - System environment: OS, CPU model, architecture, Python version, compiler flags.
2. **Zero Universalization:**
   Never state hardware-specific measurements as universal performance facts in user documentation. All performance tables must state the exact test hardware.
3. **Regression Threshold:**
   CI regression tests must flag any pull request introducing $> 25\%$ performance degradation on synthetic benchmarks compared to committed baseline.
