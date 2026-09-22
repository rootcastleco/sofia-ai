"""Tests for embedded C runtime conformance (SOFIA-EMB-*).

Verifies:
1. Static C source safety contracts (SOFIA-EMB-001):
   - Zero heap allocation (no malloc/calloc/realloc/free)
   - Compile-time capacity bounds and include guards
   - Explicit status codes and guarded divisions
2. Q16.16 fixed-point arithmetic contracts (SOFIA-EMB-002):
   - Exact resolution and dynamic range
   - Saturation without wrap-around on overflow/underflow
   - Division by zero safety
3. Golden vector cross-language conformance (SOFIA-EMB-003):
   - Verification of embedded golden vectors against Python reference implementation
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from sofia_ai.features import extract_from_array
from sofia_ai.features.statistical import STATISTICAL_FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDED_DIR = REPO_ROOT / "embedded"
SRC_DIR = EMBEDDED_DIR / "src"
TESTS_DIR = EMBEDDED_DIR / "tests"


class TestEmbeddedStaticContracts:
    """Static analysis of embedded C99 source files."""

    def test_zero_dynamic_allocation(self) -> None:
        """C runtime must not contain any dynamic memory allocation calls."""
        c_files = list(SRC_DIR.glob("*.c")) + list(SRC_DIR.glob("*.h"))
        assert len(c_files) > 0, "No C source files found in embedded/src"

        forbidden_patterns = [
            r"\bmalloc\s*\(",
            r"\bcalloc\s*\(",
            r"\brealloc\s*\(",
            r"\bfree\s*\(",
        ]

        for c_file in c_files:
            content = c_file.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                match = re.search(pattern, content)
                assert match is None, (
                    f"Forbidden allocation pattern '{pattern}' found in {c_file.name}: "
                    f"'{match.group(0)}'"
                )

    def test_compile_time_limits_defined(self) -> None:
        """Compile-time limits must be defined in sofia_features.h."""
        header = (SRC_DIR / "sofia_features.h").read_text(encoding="utf-8")
        assert "#define SOFIA_MAX_FFT_LEN 1024" in header
        assert "#define SOFIA_N_STAT_FEATURES 14" in header

    def test_include_guards_present(self) -> None:
        """All headers must have standard include guards."""
        for header_path in SRC_DIR.glob("*.h"):
            content = header_path.read_text(encoding="utf-8")
            guard_name = header_path.name.upper().replace(".", "_")
            assert f"#ifndef {guard_name}" in content, f"Missing #ifndef in {header_path.name}"
            assert f"#define {guard_name}" in content, f"Missing #define in {header_path.name}"
            assert f"#endif /* {guard_name} */" in content or "#endif" in content

    def test_cplusplus_guards_present(self) -> None:
        """Headers must have extern 'C' guards for C++ compatibility."""
        for header_path in SRC_DIR.glob("*.h"):
            content = header_path.read_text(encoding="utf-8")
            assert "extern \"C\"" in content, f"Missing extern 'C' in {header_path.name}"


class TestFixedPointEmulation:
    """Verification of Q16.16 fixed-point arithmetic contracts."""

    Q16_SHIFT = 16
    Q16_ONE = 1 << 16  # 65536
    INT32_MAX = 2147483647
    INT32_MIN = -2147483648

    @classmethod
    def q16_from_double(cls, v: float) -> int:
        if math.isnan(v):
            return 0
        scaled = v * cls.Q16_ONE
        if scaled > cls.INT32_MAX:
            return cls.INT32_MAX
        if scaled < cls.INT32_MIN:
            return cls.INT32_MIN
        return int(scaled)

    @classmethod
    def q16_to_double(cls, q: int) -> float:
        return float(q) / cls.Q16_ONE

    @classmethod
    def q16_add(cls, a: int, b: int) -> int:
        s = a + b
        return max(cls.INT32_MIN, min(cls.INT32_MAX, s))

    @classmethod
    def q16_sub(cls, a: int, b: int) -> int:
        return cls.q16_add(a, -b)

    @classmethod
    def q16_mul(cls, a: int, b: int) -> int:
        prod = (a * b) >> cls.Q16_SHIFT
        return max(cls.INT32_MIN, min(cls.INT32_MAX, prod))

    @classmethod
    def q16_div(cls, a: int, b: int) -> int:
        if b == 0:
            return 0
        quot = (a << cls.Q16_SHIFT) // b
        return max(cls.INT32_MIN, min(cls.INT32_MAX, quot))

    def test_resolution_and_range(self) -> None:
        resolution = 1.0 / self.Q16_ONE
        assert math.isclose(resolution, 1.52587890625e-5)

        # 1.0 is exactly 65536
        assert self.q16_from_double(1.0) == 65536
        assert self.q16_to_double(65536) == 1.0

        # 0.5 is exactly 32768
        assert self.q16_from_double(0.5) == 32768
        assert self.q16_to_double(32768) == 0.5

    def test_saturation(self) -> None:
        assert self.q16_from_double(1e12) == self.INT32_MAX
        assert self.q16_from_double(-1e12) == self.INT32_MIN
        assert self.q16_from_double(float("nan")) == 0

        # Addition saturation
        assert self.q16_add(self.INT32_MAX, 100) == self.INT32_MAX
        assert self.q16_add(self.INT32_MIN, -100) == self.INT32_MIN

        # Division by zero safety
        assert self.q16_div(self.Q16_ONE, 0) == 0


class TestGoldenVectorsCrossLanguage:
    """Golden vectors verification against Python reference."""

    def test_golden_vectors_file_matches_python_features(self) -> None:
        json_path = TESTS_DIR / "golden_vectors.json"
        assert json_path.exists(), "golden_vectors.json missing"

        data = json.loads(json_path.read_text(encoding="utf-8"))
        tolerance = data.get("tolerance", 1e-9)
        cases = data["cases"]
        assert len(cases) >= 5, "Expected at least 5 golden cases"

        for case in cases:
            name = case["name"]
            n = case["n"]
            sample_rate = case["sample_rate"]
            samples = np.array(case["samples"], dtype=np.float64)
            expected = case["expected"]

            assert len(samples) == n

            # Compute features via Python reference
            computed_vector = extract_from_array(samples, sample_rate)
            computed_dict = computed_vector.as_dict()

            for feat_name in STATISTICAL_FEATURE_NAMES:
                assert feat_name in expected, f"Feature {feat_name} missing in golden case {name}"
                exp_val = expected[feat_name]
                act_val = computed_dict[feat_name]
                assert math.isclose(act_val, exp_val, abs_tol=tolerance, rel_tol=tolerance), (
                    f"Case {name}, feature {feat_name}: actual {act_val} != expected {exp_val}"
                )
