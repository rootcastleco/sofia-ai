/*
 * SOFIA ENGINE — portable C99 feature runtime (reference implementation)
 * Copyright 2026 Rootcastle Engineering & Innovation. Apache-2.0. See LICENSE.
 *
 * See sofia_features.h for the engineering contract. Summary:
 *   - no heap allocation, no recursion, no global mutable state
 *   - all externally supplied lengths are range-checked
 *   - every division is guarded
 *   - non-finite input is rejected before any maths
 */

#include "sofia_features.h"

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* internal helpers                                                    */
/* ------------------------------------------------------------------ */

static int sofia_is_pow2(size_t n)
{
    return n != 0u && (n & (n - 1u)) == 0u;
}

static int sofia_is_finite_double(double v)
{
    /* Portable finiteness test: NaN fails v == v; infinities are excluded by
       comparing against the largest finite double. */
    return (v == v) && (v <= 1.7976931348623157e308) && (v >= -1.7976931348623157e308);
}

/** Guarded division: returns fallback when the denominator is negligible. */
static double sofia_safe_div(double num, double den, double fallback)
{
    double abs_den = fabs(den);
    if (!(abs_den > 1e-15)) {
        return fallback;
    }
    return num / den;
}

/* ------------------------------------------------------------------ */
/* lifecycle                                                           */
/* ------------------------------------------------------------------ */

sofia_status_t sofia_features_init(sofia_workspace_t *ws, size_t n)
{
    size_t half;
    size_t i;

    if (ws == NULL) {
        return SOFIA_ERR_NULL;
    }
    if (n == 0u || n > SOFIA_MAX_FFT_LEN || !sofia_is_pow2(n)) {
        return SOFIA_ERR_LENGTH;
    }

    half = n / 2u;
    for (i = 0u; i < half; ++i) {
        double angle = -2.0 * 3.14159265358979323846 * (double)i / (double)n;
        ws->cos_table[i] = cos(angle);
        ws->sin_table[i] = sin(angle);
    }
    ws->table_len = half;
    ws->initialised = 1;
    return SOFIA_OK;
}

const char *sofia_status_string(sofia_status_t status)
{
    switch (status) {
        case SOFIA_OK:              return "OK";
        case SOFIA_ERR_NULL:        return "NULL_ARGUMENT";
        case SOFIA_ERR_LENGTH:      return "INVALID_LENGTH";
        case SOFIA_ERR_NON_FINITE:  return "NON_FINITE_INPUT";
        case SOFIA_ERR_CAPACITY:    return "INSUFFICIENT_CAPACITY";
        default:                    return "UNKNOWN";
    }
}

/* ------------------------------------------------------------------ */
/* validation                                                          */
/* ------------------------------------------------------------------ */

sofia_status_t sofia_features_validate(const double *samples, size_t len)
{
    size_t i;
    if (samples == NULL) {
        return SOFIA_ERR_NULL;
    }
    if (len == 0u || len > SOFIA_MAX_FFT_LEN) {
        return SOFIA_ERR_LENGTH;
    }
    for (i = 0u; i < len; ++i) {
        if (!sofia_is_finite_double(samples[i])) {
            return SOFIA_ERR_NON_FINITE;
        }
    }
    return SOFIA_OK;
}

/* ------------------------------------------------------------------ */
/* primitives                                                          */
/* ------------------------------------------------------------------ */

double sofia_mean(const double *x, size_t len)
{
    double acc = 0.0;
    size_t i;
    if (x == NULL || len == 0u) {
        return 0.0;
    }
    for (i = 0u; i < len; ++i) {
        acc += x[i];
    }
    return acc / (double)len;
}

double sofia_rms(const double *x, size_t len)
{
    double acc = 0.0;
    size_t i;
    if (x == NULL || len == 0u) {
        return 0.0;
    }
    for (i = 0u; i < len; ++i) {
        acc += x[i] * x[i];
    }
    return sqrt(acc / (double)len);
}

double sofia_peak(const double *x, size_t len)
{
    double peak = 0.0;
    size_t i;
    if (x == NULL || len == 0u) {
        return 0.0;
    }
    for (i = 0u; i < len; ++i) {
        double v = fabs(x[i]);
        if (v > peak) {
            peak = v;
        }
    }
    return peak;
}

double sofia_peak_to_peak(const double *x, size_t len)
{
    double lo;
    double hi;
    size_t i;
    if (x == NULL || len == 0u) {
        return 0.0;
    }
    lo = x[0];
    hi = x[0];
    for (i = 1u; i < len; ++i) {
        if (x[i] < lo) { lo = x[i]; }
        if (x[i] > hi) { hi = x[i]; }
    }
    return hi - lo;
}

double sofia_variance(const double *x, size_t len)
{
    double m;
    double acc = 0.0;
    size_t i;
    if (x == NULL || len == 0u) {
        return 0.0;
    }
    m = sofia_mean(x, len);
    for (i = 0u; i < len; ++i) {
        double d = x[i] - m;
        acc += d * d;
    }
    return acc / (double)len;
}

double sofia_std(const double *x, size_t len)
{
    return sqrt(sofia_variance(x, len));
}

double sofia_crest_factor(const double *x, size_t len)
{
    return sofia_safe_div(sofia_peak(x, len), sofia_rms(x, len), 0.0);
}

double sofia_shape_factor(const double *x, size_t len)
{
    double acc = 0.0;
    size_t i;
    if (x == NULL || len == 0u) {
        return 0.0;
    }
    for (i = 0u; i < len; ++i) {
        acc += fabs(x[i]);
    }
    return sofia_safe_div(sofia_rms(x, len), acc / (double)len, 0.0);
}

double sofia_impulse_factor(const double *x, size_t len)
{
    double acc = 0.0;
    size_t i;
    if (x == NULL || len == 0u) {
        return 0.0;
    }
    for (i = 0u; i < len; ++i) {
        acc += fabs(x[i]);
    }
    return sofia_safe_div(sofia_peak(x, len), acc / (double)len, 0.0);
}

double sofia_margin_factor(const double *x, size_t len)
{
    double acc = 0.0;
    double root_mean;
    size_t i;
    if (x == NULL || len == 0u) {
        return 0.0;
    }
    for (i = 0u; i < len; ++i) {
        acc += sqrt(fabs(x[i]));
    }
    root_mean = acc / (double)len;
    return sofia_safe_div(sofia_peak(x, len), root_mean * root_mean, 0.0);
}

double sofia_skewness(const double *x, size_t len)
{
    double m;
    double s;
    double acc = 0.0;
    size_t i;
    if (x == NULL || len == 0u) {
        return 0.0;
    }
    m = sofia_mean(x, len);
    s = sofia_std(x, len);
    if (!(s > 1e-15)) {
        return 0.0;
    }
    for (i = 0u; i < len; ++i) {
        double z = (x[i] - m) / s;
        acc += z * z * z;
    }
    return acc / (double)len;
}

double sofia_kurtosis(const double *x, size_t len)
{
    double m;
    double s;
    double acc = 0.0;
    size_t i;
    if (x == NULL || len == 0u) {
        return 0.0;
    }
    m = sofia_mean(x, len);
    s = sofia_std(x, len);
    if (!(s > 1e-15)) {
        return 0.0;
    }
    for (i = 0u; i < len; ++i) {
        double z = (x[i] - m) / s;
        acc += z * z * z * z;
    }
    return acc / (double)len;
}

double sofia_zero_crossing_rate(const double *x, size_t len)
{
    double m;
    size_t crossings = 0u;
    size_t i;
    int prev_sign = 0;

    if (x == NULL || len < 2u) {
        return 0.0;
    }
    m = sofia_mean(x, len);
    for (i = 0u; i < len; ++i) {
        double d = x[i] - m;
        int sign;
        if (d > 0.0) {
            sign = 1;
        } else if (d < 0.0) {
            sign = -1;
        } else {
            continue; /* zero samples are skipped, matching the Python contract */
        }
        if (prev_sign != 0 && sign != prev_sign) {
            ++crossings;
        }
        prev_sign = sign;
    }
    return (double)crossings / (double)(len - 1u);
}

double sofia_energy(const double *x, size_t len)
{
    double acc = 0.0;
    size_t i;
    if (x == NULL || len == 0u) {
        return 0.0;
    }
    for (i = 0u; i < len; ++i) {
        acc += x[i] * x[i];
    }
    return acc;
}

/* ------------------------------------------------------------------ */
/* feature block                                                       */
/* ------------------------------------------------------------------ */

sofia_status_t sofia_stat_features(const double *samples,
                                   size_t len,
                                   sofia_stat_features_t *out,
                                   sofia_workspace_t *ws)
{
    sofia_status_t status;

    if (out == NULL || ws == NULL) {
        return SOFIA_ERR_NULL;
    }
    status = sofia_features_validate(samples, len);
    if (status != SOFIA_OK) {
        return status;
    }

    out->mean            = sofia_mean(samples, len);
    out->rms             = sofia_rms(samples, len);
    out->peak            = sofia_peak(samples, len);
    out->peak_to_peak    = sofia_peak_to_peak(samples, len);
    out->variance        = sofia_variance(samples, len);
    out->std             = sofia_std(samples, len);
    out->crest_factor    = sofia_crest_factor(samples, len);
    out->shape_factor    = sofia_shape_factor(samples, len);
    out->impulse_factor  = sofia_impulse_factor(samples, len);
    out->margin_factor   = sofia_margin_factor(samples, len);
    out->skewness        = sofia_skewness(samples, len);
    out->kurtosis        = sofia_kurtosis(samples, len);
    out->zero_crossing_rate = sofia_zero_crossing_rate(samples, len);
    out->energy          = sofia_energy(samples, len);
    return SOFIA_OK;
}

/* ------------------------------------------------------------------ */
/* spectral                                                            */
/* ------------------------------------------------------------------ */

/** In-place iterative radix-2 FFT. No recursion, no allocation. */
static void sofia_fft(double *re, double *im, size_t n,
                      const double *cos_table, const double *sin_table)
{
    size_t i, j, k;
    size_t step;

    /* Bit-reversal permutation (iterative). */
    j = 0u;
    for (i = 0u; i < n; ++i) {
        if (i < j) {
            double tmp = re[i];
            re[i] = re[j];
            re[j] = tmp;
            tmp = im[i];
            im[i] = im[j];
            im[j] = tmp;
        }
        k = n >> 1;
        while (k != 0u && j >= k) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    /* Cooley-Tukey butterflies. */
    for (step = 1u; step < n; step <<= 1) {
        size_t jump = step << 1;
        size_t group;
        for (group = 0u; group < step; ++group) {
            size_t tw = group * (n / jump);
            double wr = cos_table[tw];
            double wi = sin_table[tw];
            for (i = group; i < n; i += jump) {
                size_t pair = i + step;
                double xr = re[pair];
                double xi = im[pair];
                double tr = xr * wr - xi * wi;
                double ti = xr * wi + xi * wr;
                re[pair] = re[i] - tr;
                im[pair] = im[i] - ti;
                re[i] += tr;
                im[i] += ti;
            }
        }
    }
}

sofia_status_t sofia_spectral_features(const double *samples,
                                       size_t len,
                                       double sample_rate,
                                       sofia_spectral_features_t *out,
                                       sofia_workspace_t *ws)
{
    sofia_status_t status;
    size_t i;
    size_t bins;
    double centroid_num = 0.0;
    double centroid_den = 0.0;
    double peak_mag = 0.0;
    size_t peak_index = 0u;
    double total = 0.0;
    double mean_value;
    double df;

    if (out == NULL || ws == NULL) {
        return SOFIA_ERR_NULL;
    }
    if (!(sample_rate > 0.0) || !sofia_is_finite_double(sample_rate)) {
        return SOFIA_ERR_LENGTH;
    }
    if (!ws->initialised || ws->table_len != len / 2u) {
        status = sofia_features_init(ws, len);
        if (status != SOFIA_OK) {
            return status;
        }
    }
    status = sofia_features_validate(samples, len);
    if (status != SOFIA_OK) {
        return status;
    }
    if (!sofia_is_pow2(len)) {
        return SOFIA_ERR_LENGTH;
    }

    /* Copy into scratch, removing DC. */
    mean_value = sofia_mean(samples, len);
    for (i = 0u; i < len; ++i) {
        ws->re[i] = samples[i] - mean_value;
        ws->im[i] = 0.0;
    }
    sofia_fft(ws->re, ws->im, len, ws->cos_table, ws->sin_table);

    bins = len / 2u + 1u;
    df = sample_rate / (double)len;

    for (i = 0u; i < bins; ++i) {
        double mag = sqrt(ws->re[i] * ws->re[i] + ws->im[i] * ws->im[i]);
        double freq = (double)i * df;
        centroid_num += freq * mag;
        centroid_den += mag;
        total += mag;
        if (mag > peak_mag) {
            peak_mag = mag;
            peak_index = i;
        }
    }

    out->spectral_centroid_hz = sofia_safe_div(centroid_num, centroid_den, 0.0);
    out->peak_frequency_hz = (double)peak_index * df;
    out->peak_magnitude = peak_mag;
    out->total_magnitude = total;
    out->dominant_ratio = sofia_safe_div(peak_mag, total, 0.0);
    return SOFIA_OK;
}

/* ------------------------------------------------------------------ */
/* fixed-point helpers                                                 */
/* ------------------------------------------------------------------ */

/* Q16.16 range: +/-32767.99998 with a 1/65536 resolution. */
#define SOFIA_Q16_SCALE 65536.0
#define SOFIA_Q16_MAX 2147483647.0
#define SOFIA_Q16_MIN (-2147483648.0)

void sofia_q16_to_double(const int32_t *in, double *out, size_t len)
{
    size_t i;
    if (in == NULL || out == NULL) {
        return;
    }
    for (i = 0u; i < len; ++i) {
        out[i] = (double)in[i] / SOFIA_Q16_SCALE;
    }
}

void sofia_double_to_q16(const double *in, int32_t *out, size_t len)
{
    size_t i;
    if (in == NULL || out == NULL) {
        return;
    }
    for (i = 0u; i < len; ++i) {
        double v = in[i];
        double scaled;
        if (!sofia_is_finite_double(v)) {
            out[i] = 0; /* non-finite input is rejected upstream; saturate to zero */
            continue;
        }
        scaled = v * SOFIA_Q16_SCALE;
        if (scaled > SOFIA_Q16_MAX) {
            out[i] = (int32_t)2147483647;
        } else if (scaled < SOFIA_Q16_MIN) {
            out[i] = (int32_t)(-2147483647 - 1);
        } else {
            out[i] = (int32_t)scaled;
        }
    }
}
