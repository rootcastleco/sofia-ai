/*
 * SOFIA ENGINE — host-side test for the C99 reference runtime
 * Copyright 2026 Rootcastle Engineering & Innovation. Apache-2.0. See LICENSE.
 *
 * Asserts closed-form mathematical expectations and the embedded engineering
 * rules (bounds checking, non-finite rejection, no wrap-around). Runs on the
 * host with no hardware. It does NOT verify target-specific behaviour: that
 * requires a cross-toolchain and hardware (see docs/embedded.md).
 *
 * Build:  cc -std=c99 -Wall -Wextra -Werror -pedantic -o test_sofia \
 *             tests/test_sofia_features.c ../src/sofia_features.c -lm
 */

#include "../src/sofia_features.h"
#include "../src/sofia_fixed.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;
static int checks = 0;

static void check_true(int condition, const char *name)
{
    ++checks;
    if (!condition) {
        ++failures;
        printf("FAIL: %s\n", name);
    }
}

static void check_close(double actual, double expected, double tol, const char *name)
{
    double diff = fabs(actual - expected);
    ++checks;
    if (!(diff <= tol)) {
        ++failures;
        printf("FAIL: %s (actual %.12g expected %.12g diff %.3g)\n",
               name, actual, expected, diff);
    }
}

static void test_constant_signal(void)
{
    double x[64];
    size_t i;
    for (i = 0u; i < 64u; ++i) {
        x[i] = 3.0;
    }
    check_close(sofia_mean(x, 64), 3.0, 1e-12, "mean of constant");
    check_close(sofia_rms(x, 64), 3.0, 1e-12, "rms of constant");
    check_close(sofia_std(x, 64), 0.0, 1e-12, "std of constant");
    check_close(sofia_crest_factor(x, 64), 1.0, 1e-12, "crest factor of constant");
    check_close(sofia_variance(x, 64), 0.0, 1e-12, "variance of constant");
    check_close(sofia_peak_to_peak(x, 64), 0.0, 1e-12, "peak-to-peak of constant");
    check_close(sofia_zero_crossing_rate(x, 64), 0.0, 1e-12, "zcr of constant");
    check_close(sofia_energy(x, 64), 64.0 * 9.0, 1e-9, "energy of constant");
    /* std == 0 must not produce a division by zero in kurtosis/skewness. */
    check_close(sofia_kurtosis(x, 64), 0.0, 1e-12, "kurtosis guarded divide");
    check_close(sofia_skewness(x, 64), 0.0, 1e-12, "skewness guarded divide");
}

static void test_sine_signal(void)
{
    double x[256];
    size_t i;
    for (i = 0u; i < 256u; ++i) {
        x[i] = sin(2.0 * 3.14159265358979323846 * 25.0 * (double)i / 1000.0);
    }
    /* Unit-amplitude sine: RMS = 1/sqrt(2), peak = 1, crest = sqrt(2). */
    check_close(sofia_rms(x, 256), 1.0 / sqrt(2.0), 1e-3, "rms of sine");
    check_close(sofia_peak(x, 256), 1.0, 1e-3, "peak of sine");
    check_close(sofia_crest_factor(x, 256), sqrt(2.0), 1e-3, "crest of sine");
    check_close(sofia_peak_to_peak(x, 256), 2.0, 1e-3, "peak-to-peak of sine");
    /* 25 Hz at 1000 Hz sample rate => 50 zero crossings in 256 samples. */
    check_close(sofia_zero_crossing_rate(x, 256), 50.0 / 255.0, 5e-3, "zcr of sine");
    check_close(sofia_kurtosis(x, 256), 1.5, 0.05, "kurtosis of sine");
    check_close(sofia_skewness(x, 256), 0.0, 0.05, "skewness of sine");
    check_close(sofia_shape_factor(x, 256), 1.1107, 0.02, "shape factor of sine");
}

static void test_validation(void)
{
    double x[8];
    size_t i;
    for (i = 0u; i < 8u; ++i) {
        x[i] = (double)i;
    }
    check_true(sofia_features_validate(x, 8) == SOFIA_OK, "validate good input");
    check_true(sofia_features_validate(NULL, 8) == SOFIA_ERR_NULL, "validate NULL");
    check_true(sofia_features_validate(x, 0) == SOFIA_ERR_LENGTH, "validate zero length");
    check_true(sofia_features_validate(x, SOFIA_MAX_FFT_LEN + 1) == SOFIA_ERR_LENGTH,
               "validate over-length");

    x[3] = (double)NAN;
    check_true(sofia_features_validate(x, 8) == SOFIA_ERR_NON_FINITE, "validate NaN");
    x[3] = (double)INFINITY;
    check_true(sofia_features_validate(x, 8) == SOFIA_ERR_NON_FINITE, "validate Inf");
}

static void test_stat_features_contract(void)
{
    sofia_workspace_t ws;
    sofia_stat_features_t f;
    double x[128];
    size_t i;

    for (i = 0u; i < 128u; ++i) {
        x[i] = sin(2.0 * 3.14159265358979323846 * 25.0 * (double)i / 1000.0)
             + 0.05 * sin(2.0 * 3.14159265358979323846 * 50.0 * (double)i / 1000.0);
    }

    check_true(sofia_features_init(&ws, 128) == SOFIA_OK, "init 128");
    check_true(sofia_stat_features(x, 128, &f, &ws) == SOFIA_OK, "stat features ok");
    check_true(f.rms > 0.0, "rms positive");
    check_true(f.crest_factor >= 1.0, "crest factor >= 1");
    check_true(f.energy > 0.0, "energy positive");
    check_true(f.zero_crossing_rate >= 0.0 && f.zero_crossing_rate <= 1.0,
               "zcr within [0,1]");

    check_true(sofia_stat_features(x, 128, NULL, &ws) == SOFIA_ERR_NULL, "null output");
    check_true(sofia_stat_features(NULL, 128, &f, &ws) == SOFIA_ERR_NULL, "null input");
    check_true(sofia_stat_features(x, 0, &f, &ws) == SOFIA_ERR_LENGTH, "zero length input");
    check_true(sofia_features_init(&ws, 0) == SOFIA_ERR_LENGTH, "init zero");
    check_true(sofia_features_init(&ws, 100) == SOFIA_ERR_LENGTH, "init non power of two");
}

static void test_spectral_features(void)
{
    sofia_workspace_t ws;
    sofia_spectral_features_t s;
    double x[256];
    size_t i;

    for (i = 0u; i < 256u; ++i) {
        x[i] = sin(2.0 * 3.14159265358979323846 * 50.0 * (double)i / 1000.0);
    }
    check_true(sofia_features_init(&ws, 256) == SOFIA_OK, "init 256");
    check_true(sofia_spectral_features(x, 256, 1000.0, &s, &ws) == SOFIA_OK,
               "spectral ok");
    /* A 50 Hz tone at 1000 Hz / 256 samples => bin 12.8 -> nearest peak bin. */
    check_close(s.peak_frequency_hz, 50.0, 4.0, "peak frequency 50 Hz");
    check_true(s.peak_magnitude > 0.0, "peak magnitude positive");
    check_true(s.dominant_ratio > 0.0 && s.dominant_ratio <= 1.0,
               "dominant ratio in (0,1]");
    check_true(sofia_spectral_features(x, 256, 0.0, &s, &ws) == SOFIA_ERR_LENGTH,
               "spectral rejects zero sample rate");
    check_true(sofia_spectral_features(x, 100, 1000.0, &s, &ws) == SOFIA_ERR_LENGTH,
               "spectral rejects non power of two");
}

static void test_fixed_point(void)
{
    /* Saturation, not wrap-around. */
    check_true(sofia_q16_from_double(1.0) == SOFIA_Q16_ONE, "q16 one");
    check_true(sofia_q16_from_double(0.5) == SOFIA_Q16_HALF, "q16 half");
    check_true(sofia_q16_from_double(1e9) == 2147483647, "q16 saturates high");
    check_true(sofia_q16_from_double(-1e9) == -2147483648, "q16 saturates low");
    check_true(sofia_q16_from_double((double)NAN) == 0, "q16 NaN to zero");

    check_true(sofia_q16_add(2147483647, 1000) == 2147483647, "q16 add saturates");
    check_true(sofia_q16_sub(-2147483648, 1000) == -2147483648, "q16 sub saturates");
    check_true(sofia_q16_mul(SOFIA_Q16_ONE, SOFIA_Q16_ONE) == SOFIA_Q16_ONE,
               "q16 mul identity");
    check_true(sofia_q16_mul(SOFIA_Q16_HALF, SOFIA_Q16_HALF) == (SOFIA_Q16_ONE >> 2),
               "q16 mul quarter");
    check_true(sofia_q16_div(SOFIA_Q16_ONE, 0) == 0, "q16 div by zero returns 0");
    check_true(sofia_q16_div(SOFIA_Q16_ONE, SOFIA_Q16_ONE) == SOFIA_Q16_ONE,
               "q16 div identity");
    check_true(sofia_q16_abs(-2147483648) == 2147483647, "q16 abs saturates");
    check_close(sofia_q16_to_double_scalar(SOFIA_Q16_HALF), 0.5, 1e-12, "q16 to double");
}

static void test_q16_roundtrip(void)
{
    double in[32];
    int32_t q[32];
    double out[32];
    size_t i;
    for (i = 0u; i < 32u; ++i) {
        in[i] = 0.001 * (double)i - 0.5;
    }
    sofia_double_to_q16(in, q, 32);
    sofia_q16_to_double(q, out, 32);
    for (i = 0u; i < 32u; ++i) {
        check_close(out[i], in[i], 1.0 / 65536.0, "q16 roundtrip");
    }
}

int main(void)
{
    printf("SOFIA embedded runtime tests\n");
    test_constant_signal();
    test_sine_signal();
    test_validation();
    test_stat_features_contract();
    test_spectral_features();
    test_fixed_point();
    test_q16_roundtrip();

    printf("%d checks, %d failures\n", checks, failures);
    if (failures != 0) {
        printf("RESULT: FAILED\n");
        return 1;
    }
    printf("RESULT: PASSED\n");
    return 0;
}
