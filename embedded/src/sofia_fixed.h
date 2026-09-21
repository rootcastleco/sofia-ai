/*
 * SOFIA ENGINE — Q-format fixed-point preprocessing helpers
 * Copyright 2026 Rootcastle Engineering & Innovation. Apache-2.0. See LICENSE.
 *
 * Header-only so it can be compiled into a bare-metal image with no linker
 * dependencies. Every macro is bounded and every conversion saturates.
 *
 * Q16.16: 16 integer bits, 16 fractional bits.
 *   resolution  = 1 / 65536        = 1.5258789e-05
 *   range       = [-32768, 32767.9999847]
 *
 * Choose Q16.16 when the signal is already in engineering units with a modest
 * dynamic range (e.g. acceleration in g up to +/-2000 g after scaling). For raw
 * ADC counts use Q0.31 or the device's native width.
 */

#ifndef SOFIA_FIXED_H
#define SOFIA_FIXED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SOFIA_Q16_SHIFT 16
#define SOFIA_Q16_ONE (1 << SOFIA_Q16_SHIFT)          /* 65536 */
#define SOFIA_Q16_HALF (SOFIA_Q16_ONE >> 1)           /* 32768 */
#define SOFIA_Q16_INT_MAX 32767
#define SOFIA_Q16_INT_MIN (-32768)

/** Convert double to Q16.16 with saturation (no wrap-around). */
static inline int32_t sofia_q16_from_double(double v)
{
    double scaled;
    if (v != v) {              /* NaN */
        return 0;
    }
    scaled = v * (double)SOFIA_Q16_ONE;
    if (scaled > 2147483647.0) {
        return 2147483647;
    }
    if (scaled < -2147483648.0) {
        return -2147483648;
    }
    return (int32_t)scaled;
}

/** Convert Q16.16 to double. */
static inline double sofia_q16_to_double_scalar(int32_t q)
{
    return (double)q / (double)SOFIA_Q16_ONE;
}

/** Saturating Q16.16 addition. */
static inline int32_t sofia_q16_add(int32_t a, int32_t b)
{
    int64_t sum = (int64_t)a + (int64_t)b;
    if (sum > 2147483647LL) { return 2147483647; }
    if (sum < -2147483648LL) { return -2147483648; }
    return (int32_t)sum;
}

/** Saturating Q16.16 subtraction. */
static inline int32_t sofia_q16_sub(int32_t a, int32_t b)
{
    return sofia_q16_add(a, -b);
}

/** Saturating Q16.16 multiplication (uses 64-bit intermediate). */
static inline int32_t sofia_q16_mul(int32_t a, int32_t b)
{
    int64_t product = ((int64_t)a * (int64_t)b) >> SOFIA_Q16_SHIFT;
    if (product > 2147483647LL) { return 2147483647; }
    if (product < -2147483648LL) { return -2147483648; }
    return (int32_t)product;
}

/** Saturating Q16.16 division. Returns 0 when the divisor is zero. */
static inline int32_t sofia_q16_div(int32_t a, int32_t b)
{
    int64_t quotient;
    if (b == 0) {
        return 0;
    }
    quotient = ((int64_t)a << SOFIA_Q16_SHIFT) / (int64_t)b;
    if (quotient > 2147483647LL) { return 2147483647; }
    if (quotient < -2147483648LL) { return -2147483648; }
    return (int32_t)quotient;
}

/** Absolute value, saturating at INT32_MAX rather than wrapping INT32_MIN. */
static inline int32_t sofia_q16_abs(int32_t a)
{
    if (a < 0) {
        return (a == (int32_t)(-2147483647 - 1)) ? 2147483647 : -a;
    }
    return a;
}

#ifdef __cplusplus
}
#endif

#endif /* SOFIA_FIXED_H */
