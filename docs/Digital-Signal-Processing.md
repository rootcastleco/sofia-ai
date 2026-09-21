# Digital Signal Processing (DSP)

Sofia Engine's signal processing architecture (`sofia_ai.signal` and `sofia_ai.features`) is built on rigorous digital signal processing theory. All functions are pure, deterministic, and implemented in pure NumPy with zero external framework overhead.

---

## 1. Time-Domain Statistical Features

Given a discrete time-series window $\mathbf{x} = [x_0, x_1, \dots, x_{N-1}]^T \in \mathbb{R}^N$ sampled at rate $f_s$:

### Primary Measures
* **Mean ($\bar{x}$)**:
  $$\bar{x} = \frac{1}{N} \sum_{n=0}^{N-1} x_n$$

* **Root Mean Square (RMS)** (reflects overall vibration energy per ISO 10816):
  $$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x_n^2}$$

* **Peak ($x_{pk}$) & Peak-to-Peak ($x_{p2p}$)**:
  $$x_{pk} = \max_{n} |x_n|, \quad x_{p2p} = \max_{n}(x_n) - \min_{n}(x_n)$$

* **Sample Variance ($\sigma^2$) & Standard Deviation ($\sigma$)**:
  $$\sigma^2 = \frac{1}{N-1}\sum_{n=0}^{N-1}(x_n - \bar{x})^2, \quad \sigma = \sqrt{\sigma^2}$$

### Higher-Order Statistical Moments
* **Sample Skewness ($S$)** (asymmetry of probability distribution):
  $$S = \frac{\frac{1}{N}\sum_{n=0}^{N-1}(x_n - \bar{x})^3}{\left(\frac{1}{N}\sum_{n=0}^{N-1}(x_n - \bar{x})^2\right)^{3/2}}$$

* **Sample Kurtosis ($K$)** (peakedness and heavy tails, sensitive to early bearing spalls):
  $$K = \frac{\frac{1}{N}\sum_{n=0}^{N-1}(x_n - \bar{x})^4}{\left(\frac{1}{N}\sum_{n=0}^{N-1}(x_n - \bar{x})^2\right)^2}$$

### Dimensionless Shape Indicators
* **Crest Factor ($CF$)**:
  $$CF = \frac{x_{pk}}{\text{RMS}(\mathbf{x})}$$
  A healthy rotating machine exhibits $CF \approx 3.0 \text{ to } 3.5$. Repetitive impacts from bearing rolling element impacts cause $CF > 5.0$.

* **Shape Factor ($SF$)**:
  $$SF = \frac{\text{RMS}(\mathbf{x})}{\frac{1}{N}\sum_{n=0}^{N-1}|x_n|}$$

* **Impulse Factor ($IF$)**:
  $$IF = \frac{x_{pk}}{\frac{1}{N}\sum_{n=0}^{N-1}|x_n|}$$

* **Margin Factor ($MF$)**:
  $$MF = \frac{x_{pk}}{\left(\frac{1}{N}\sum_{n=0}^{N-1}\sqrt{|x_n|}\right)^2}$$

* **Zero Crossing Rate ($ZCR$)**:
  $$ZCR = \frac{1}{N-1} \sum_{n=1}^{N-1} \mathbb{I}\left(x_n \cdot x_{n-1} < 0\right)$$

---

## 2. Spectral Estimation & Energy Conservation

### Discrete Fourier Transform (DFT)
The discrete spectrum of the windowed signal is:

$$X_k = \sum_{n=0}^{N-1} x_n w_n e^{-j \frac{2\pi}{N} k n}, \quad k = 0, \dots, N-1$$

Where $w_n$ represents the analysis window.

### Parseval's Theorem & Energy Conservation
Sofia Engine guarantees strict energy conservation across time and frequency representations:

$$\sum_{n=0}^{N-1} |x_n|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X_k|^2$$

When computing the one-sided Power Spectral Density (PSD) $P(f_k)$ via Welch’s averaged periodogram:
1. The DC component ($k=0$) and Nyquist frequency ($k=N/2$) are retained without scaling.
2. Positive frequencies ($0 < k < N/2$) are doubled ($2 \cdot |X_k|^2$) to preserve total power.
3. Coherent gain factor $S_1 = \sum w_n$ and noise equivalent bandwidth factor $S_2 = \sum w_n^2$ are applied.

### Spectral Descriptors
* **Spectral Centroid ($f_c$)** (center of spectral mass):
  $$f_c = \frac{\sum_{k=0}^{M-1} f_k P(f_k)}{\sum_{k=0}^{M-1} P(f_k)}$$

* **Spectral Spread ($\sigma_f$)** (spectral bandwidth around the centroid):
  $$\sigma_f = \sqrt{\frac{\sum_{k=0}^{M-1}(f_k - f_c)^2 P(f_k)}{\sum_{k=0}^{M-1} P(f_k)}}$$

* **Spectral Flatness (Wiener Entropy)**:
  $$\gamma_\infty = \frac{\exp\left(\frac{1}{M}\sum_{k=0}^{M-1} \ln P(f_k)\right)}{\frac{1}{M}\sum_{k=0}^{M-1} P(f_k)}$$
  $\gamma_\infty \to 0$ indicates a pure tone / harmonic peak; $\gamma_\infty \to 1$ represents uniform white noise.

* **Band Energy ($E_{[f_{low}, f_{high}]}$)**:
  $$E_{[f_{low}, f_{high}]} = \sum_{k: f_{low} \le f_k \le f_{high}} P(f_k) \Delta f$$

---

## 3. Envelope Analysis via the Hilbert Transform

Bearing defect impacts produce structural resonances modulated at fault characteristic frequencies. Sofia computes the analytic signal $\tilde{x}(t)$:

$$\tilde{x}(t) = x(t) + j \cdot \hat{x}(t)$$

Where the Hilbert transform $\hat{x}(t) = \mathcal{H}\{x(t)\}$ is given by:

$$\hat{x}(t) = \frac{1}{\pi} \text{p.v.} \int_{-\infty}^{\infty} \frac{x(\tau)}{t - \tau} \, d\tau$$

In the frequency domain:

$$\mathcal{F}\{\hat{x}(t)\} = -j \cdot \text{sgn}(f) \cdot X(f)$$

The **demodulated instantaneous envelope** is extracted as:

$$A(t) = |\tilde{x}(t)| = \sqrt{x(t)^2 + \hat{x}(t)^2}$$

The spectrum of $A(t)$ exposes periodic micro-impacts even when submerged in high-amplitude machine noise.

---

## 4. Rotating Machinery Kinematics

For assets rotating at shaft speed $f_r$ (in Hz, where $f_r = \text{RPM} / 60$):

```
                       +-------------------------------+
                       |      BEARING GEOMETRY         |
                       |  Nb  = Number of rollers      |
                       |  d   = Roller diameter        |
                       |  D   = Pitch diameter         |
                       |  α   = Contact angle          |
                       +-------------------------------+
                                      |
         +--------------------+-------+--------------------+
         |                    |                            |
         v                    v                            v
   Outer Race (BPFO)    Inner Race (BPFI)          Ball Spin (BSF)
```

### Bearing Fault Characteristic Frequencies

1. **Ball Pass Frequency Outer Race (BPFO)**:
   $$\text{BPFO} = \frac{N_b}{2} f_r \left(1 - \frac{d}{D} \cos \alpha\right)$$

2. **Ball Pass Frequency Inner Race (BPFI)**:
   $$\text{BPFI} = \frac{N_b}{2} f_r \left(1 + \frac{d}{D} \cos \alpha\right)$$

3. **Ball Spin Frequency (BSF)**:
   $$\text{BSF} = \frac{D}{2d} f_r \left[1 - \left(\frac{d}{D}\cos \alpha\right)^2\right]$$

4. **Fundamental Train Frequency (FTF / Cage)**:
   $$\text{FTF} = \frac{1}{2} f_r \left(1 - \frac{d}{D}\cos \alpha\right)$$

### Gearbox Dynamics
* **Gear Mesh Frequency (GMF)**:
  $$\text{GMF} = N_{teeth} \cdot f_r$$
* **Sideband Modulation**: Faults on gear teeth manifest as sidebands spaced at running speed multiples:
  $$f_{sideband} = \text{GMF} \pm k \cdot f_r, \quad k \in \{1, 2, 3, \dots\}$$

---

## 5. Pure Signal Filters & Conditioning

* **Butterworth IIR Filtering**: Implemented as second-order section (SOS) cascades to maintain numerical stability on low-precision floating point.
* **Moving Average & Median Filters**: Constant-memory rolling smoothing for transient artifact suppression.
* **Linear & Constant Detrending**: Removes DC bias and baseline wander before FFT computation:
  $$x_{detrended}[n] = x[n] - (\hat{a} \cdot n + \hat{b})$$
* **Resampling**: Decimation and band-limited linear interpolation supporting fractional rate conversion.
