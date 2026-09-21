# Multi-Domain Industrial Signal Processing

Sofia Engine extends far beyond mechanical vibration, providing comprehensive digital signal processing for **electrical power grids, acoustic emissions, process telemetry, and multi-axis inertial dynamics**.

These algorithms are derived and ported from **[Rootcastle REI SignalLab](https://github.com/rootcastleco/rei-signallab)**.

---

## 1. Electrical Power Quality (IEEE 519 / IEC 61000-4-30)

Electrical telemetry from voltage ($v(t)$) and current ($i(t)$) probes is analyzed for power efficiency and power quality disturbances:

### 1. Power Metrics ($P, Q, S, PF$)
* **Active Power ($P$)**:
  $$P = \frac{1}{N} \sum_{n=0}^{N-1} v_n \cdot i_n$$
* **Apparent Power ($S$)**:
  $$S = V_{rms} \cdot I_{rms}$$
* **Reactive Power ($Q$)**:
  $$Q = \sqrt{S^2 - P^2}$$
* **Power Factor ($PF$)**:
  $$PF = \frac{P}{S}, \quad PF \in [-1.0, 1.0]$$

### 2. Total Harmonic Distortion ($\text{THD}$)
Evaluated up to the 50th harmonic order:
$$\text{THD} = \frac{\sqrt{\sum_{h=2}^{50} V_h^2}}{V_1} \times 100\%$$

Where individual harmonic magnitudes are verified against IEEE 519 limits ($3.0\%$ for odd harmonics, $1.5\%$ for even harmonics).

### 3. Fortescue 3-Phase Symmetrical Components
For 3-phase unbalanced AC power systems, Sofia decomposes line voltages into symmetrical components:
$$\begin{bmatrix} V_0 \\ V_1 \\ V_2 \end{bmatrix} = \frac{1}{3} \begin{bmatrix} 1 & 1 & 1 \\ 1 & a & a^2 \\ 1 & a^2 & a \end{bmatrix} \begin{bmatrix} V_a \\ V_b \\ V_c \end{bmatrix}, \quad a = e^{j 120^\circ}$$

* **Voltage Unbalance Factor ($\text{VUF}$)**:
  $$\text{VUF} = \frac{|V_2|}{|V_1|} \times 100\%$$
  A $\text{VUF} > 2.0\%$ triggers warnings for three-phase motor de-rating.

### 4. Power Disturbance Events
* **Voltage Sag**: $V_{rms} \in [10\%, 90\%]$ of nominal.
* **Voltage Swell**: $V_{rms} > 110\%$ of nominal.
* **Interruption**: $V_{rms} < 10\%$ of nominal.
* **Inrush Current**: Instantaneous peak-to-RMS ratio $> 3.0$.

---

## 2. Acoustic Emission & Cavitation Monitoring (ASTM E1316)

High-frequency acoustic emission (AE) and ultrasound telemetry detect stress wave bursts from micro-cracking, partial electrical discharge, and fluid cavitation:

* **AE Energy**:
  $$E_{AE} = \sum_{n=0}^{N-1} v_n^2 \Delta t$$
* **Counts**: Number of rising threshold crossings ($> 3\sigma$).
* **Rise Time**: Elapsed time from first threshold crossing to peak amplitude.
* **Duration**: Total time the burst envelope remains above threshold.
* **Cavitation Index ($C_p$)**:
  $$C_p = \frac{\sum_{k: 5\text{kHz} \le f_k \le 20\text{kHz}} |X_k|^2}{\sum_k |X_k|^2}$$
  Detects pump and valve cavitation before mechanical pitting occurs.

---

## 3. Thermal & Fluid Process Dynamics

* **Temperature Rate of Change**: $\frac{dT}{dt}$ (in $^\circ\text{C/s}$ or $^\circ\text{C/min}$) identifies sudden thermal runaways or heat-sink failures.
* **Thermal Inertia & Delta**: Dynamic difference between peak and ambient process temperature.
* **Hydraulic Pressure Pulsation**: Peak-to-peak pressure swings and pressure crest factor for detecting water hammer and valve chatter.

---

## 4. Multi-Axis Inertial Dynamics (IMU)

* **Acceleration Vector Magnitude**:
  $$|\mathbf{a}| = \sqrt{a_x^2 + a_y^2 + a_z^2}$$
* **Dynamic Tilt (Pitch & Roll)**:
  $$\text{Pitch} = \text{atan2}(-a_x, \sqrt{a_y^2 + a_z^2}), \quad \text{Roll} = \text{atan2}(a_y, a_z)$$
* **Dynamic Jerk**:
  $$\mathbf{j} = \frac{d\mathbf{a}}{dt}$$
  Monitors mechanical shock and high-stress robot arm reversals.
