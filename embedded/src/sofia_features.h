/*
 * SOFIA ENGINE — portable C99 feature runtime (reference implementation)
 *
 * Copyright 2026 Rootcastle Engineering & Innovation
 * Licensed under the Apache License, Version 2.0. See LICENSE.
 *
 * ---------------------------------------------------------------------------
 * Purpose
 * ---------------------------------------------------------------------------
 * This is the reference embedded implementation of the Sofia time-domain feature
 * contract, mirroring sofia_ai.features.statistical. It exists so a feature
 * vector computed on a gateway and the same vector computed on a microcontroller
 * are numerically comparable, with golden vectors proving it.
 *
 * ---------------------------------------------------------------------------
 * Embedded engineering rules honoured here
 * ---------------------------------------------------------------------------
 *  - No heap allocation. Every function writes into caller-provided storage.
 *  - No recursion. Every loop has a compile-time or caller-supplied bound.
 *  - No floating-point exceptions are left unhandled: non-finite inputs are
 *    rejected by sofia_features_validate() before any maths runs.
 *  - All externally supplied lengths are range-checked against the declared
 *    capacity of the output buffer.
 *  - No global mutable state. The workspace is explicit and owned by the caller.
 *  - Division by zero is guarded explicitly at every site.
 *
 * The FFT is an iterative radix-2 implementation with a pre-computed twiddle
 * table held in the workspace, so no allocation occurs at transform time.
 *
 * Concurrency: this module is not thread-safe and not ISR-safe by design.
 * A single workspace must be owned by one execution context. If an ISR fills the
 * sample buffer, hand ownership to the application context before calling any
 * function here.
 */

#ifndef SOFIA_FEATURES_H
#define SOFIA_FEATURES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum transform length supported by the workspace (compile-time bound). */
#define SOFIA_MAX_FFT_LEN 1024

/** Number of time-domain features in the canonical contract. */
#define SOFIA_N_STAT_FEATURES 14

/**
 * Status codes. Every fallible function returns one of these; there is no
 * error state stored in the workspace.
 */
typedef enum {
    SOFIA_OK = 0,
    SOFIA_ERR_NULL = 1,            /**< NULL pointer argument */
    SOFIA_ERR_LENGTH = 2,          /**< length is zero, or exceeds capacity */
    SOFIA_ERR_NON_FINITE = 3,      /**< input contains NaN or Inf */
    SOFIA_ERR_CAPACITY = 4         /**< output buffer too small */
} sofia_status_t;

/**
 * Fixed-size workspace. Allocate it once (static, or as a member of a larger
 * struct) and pass it to every call. This is the only state the module needs.
 */
typedef struct {
    double re[SOFIA_MAX_FFT_LEN];      /**< real scratch buffer */
    double im[SOFIA_MAX_FFT_LEN];      /**< imaginary scratch buffer */
    double cos_table[SOFIA_MAX_FFT_LEN / 2]; /**< twiddle cosines */
    double sin_table[SOFIA_MAX_FFT_LEN / 2]; /**< twiddle sines */
    size_t table_len;                  /**< twiddle table length actually built */
    int initialised;                   /**< 1 once tables are built */
} sofia_workspace_t;

/**
 * Time-domain feature block. Field order matches
 * sofia_ai.features.statistical.STATISTICAL_FEATURE_NAMES exactly:
 *
 *   mean, rms, peak, peak_to_peak, variance, std, crest_factor, shape_factor,
 *   impulse_factor, margin_factor, skewness, kurtosis, zero_crossing_rate, energy
 */
typedef struct {
    double mean;
    double rms;
    double peak;
    double peak_to_peak;
    double variance;
    double std;
    double crest_factor;
    double shape_factor;
    double impulse_factor;
    double margin_factor;
    double skewness;
    double kurtosis;
    double zero_crossing_rate;
    double energy;
} sofia_stat_features_t;

/**
 * Spectral summary block (magnitudes from an in-place real FFT).
 */
typedef struct {
    double spectral_centroid_hz;
    double peak_frequency_hz;
    double peak_magnitude;
    double total_magnitude;
    double dominant_ratio;
} sofia_spectral_features_t;

/* ------------------------------------------------------------------ */
/* lifecycle                                                           */
/* ------------------------------------------------------------------ */

/**
 * Initialise the workspace and build the twiddle tables.
 *
 * Must be called once before any other function. Safe to call again; it will
 * rebuild the tables. n must be a power of two and <= SOFIA_MAX_FFT_LEN.
 */
sofia_status_t sofia_features_init(sofia_workspace_t *ws, size_t n);

/* ------------------------------------------------------------------ */
/* validation                                                          */
/* ------------------------------------------------------------------ */

/**
 * Validate a sample buffer: non-NULL, length within bounds, all values finite.
 * Call this on every externally sourced buffer before feature extraction.
 */
sofia_status_t sofia_features_validate(const double *samples, size_t len);

/* ------------------------------------------------------------------ */
/* time-domain features                                                */
/* ------------------------------------------------------------------ */

/**
 * Compute the canonical time-domain feature block.
 *
 * @param samples  Input samples, length len.
 * @param len      Sample count, must be >= 1 and <= capacity of the workspace.
 * @param out      Destination feature block (caller owned).
 * @param ws       Initialised workspace used as scratch.
 */
sofia_status_t sofia_stat_features(const double *samples,
                                   size_t len,
                                   sofia_stat_features_t *out,
                                   sofia_workspace_t *ws);

/* ------------------------------------------------------------------ */
/* spectral features                                                   */
/* ------------------------------------------------------------------ */

/**
 * Compute one-sided magnitude spectral features using an iterative radix-2 FFT.
 *
 * @param samples    Input samples, length len (must be a power of two).
 * @param len        Sample count, <= SOFIA_MAX_FFT_LEN.
 * @param sample_rate Sample rate in Hz, must be > 0.
 * @param out        Destination spectral block (caller owned).
 * @param ws         Initialised workspace used as scratch.
 */
sofia_status_t sofia_spectral_features(const double *samples,
                                       size_t len,
                                       double sample_rate,
                                       sofia_spectral_features_t *out,
                                       sofia_workspace_t *ws);

/* ------------------------------------------------------------------ */
/* individual primitives (independently testable)                      */
/* ------------------------------------------------------------------ */

double sofia_mean(const double *x, size_t len);
double sofia_rms(const double *x, size_t len);
double sofia_peak(const double *x, size_t len);
double sofia_peak_to_peak(const double *x, size_t len);
double sofia_variance(const double *x, size_t len);
double sofia_std(const double *x, size_t len);
double sofia_crest_factor(const double *x, size_t len);
double sofia_shape_factor(const double *x, size_t len);
double sofia_impulse_factor(const double *x, size_t len);
double sofia_margin_factor(const double *x, size_t len);
double sofia_skewness(const double *x, size_t len);
double sofia_kurtosis(const double *x, size_t len);
double sofia_zero_crossing_rate(const double *x, size_t len);
double sofia_energy(const double *x, size_t len);

/* ------------------------------------------------------------------ */
/* fixed-point preprocessing (see sofia_fixed.h)                       */
/* ------------------------------------------------------------------ */

/**
 * Convert a Q16.16 fixed-point sample buffer to double.
 * Bounded by len; writes only into out[0..len-1].
 */
void sofia_q16_to_double(const int32_t *in, double *out, size_t len);

/**
 * Convert doubles to Q16.16 with saturation. Values outside the representable
 * range saturate rather than wrapping, because a wrap in a vibration signal is
 * a sign inversion and would be silently wrong.
 */
void sofia_double_to_q16(const double *in, int32_t *out, size_t len);

const char *sofia_status_string(sofia_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* SOFIA_FEATURES_H */
