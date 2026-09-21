"""Unit tests for validation primitives and serialization (SOFIA-FR-011/012)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sofia_ai.core.errors import ValidationError
from sofia_ai.core.hashing import sha256_bytes, sha256_file, sha256_json, verify_checksum
from sofia_ai.core.serialization import (
    SofiaJSONEncoder,
    decode,
    dump_json,
    encode,
    load_json,
    read_jsonl,
    write_jsonl,
)
from sofia_ai.core.validation import (
    MAX_PAYLOAD_BYTES,
    as_float_array,
    ensure_finite,
    ensure_finite_array,
    sanitize_identifier,
    validate_identifier,
    validate_payload_size,
    validate_positive_float,
    validate_positive_int,
    validate_probability,
    validate_range,
    validate_tags,
)


class TestFinite:
    def test_ensure_finite_ok(self) -> None:
        assert ensure_finite(1.5) == 1.5

    def test_ensure_finite_nan(self) -> None:
        with pytest.raises(ValidationError):
            ensure_finite(float("nan"))

    def test_ensure_finite_inf(self) -> None:
        with pytest.raises(ValidationError):
            ensure_finite(float("inf"))

    def test_array_ok(self) -> None:
        arr = ensure_finite_array([1.0, 2.0, 3.0])
        assert arr.tolist() == [1.0, 2.0, 3.0]

    def test_array_rejects_nan(self) -> None:
        with pytest.raises(ValidationError):
            ensure_finite_array([1.0, float("nan")])

    def test_empty_array_allowed(self) -> None:
        assert ensure_finite_array([]).size == 0

    def test_as_float_array_contiguous(self) -> None:
        arr = as_float_array([1.0, 2.0])
        assert arr.dtype == np.float64
        assert arr.flags["C_CONTIGUOUS"]


class TestIdentifiers:
    def test_sanitize_trims_whitespace(self) -> None:
        assert sanitize_identifier("  pump-01  ") == "pump-01"

    def test_control_characters_are_rejected_not_mutated(self) -> None:
        """Identifiers are rejected, never silently rewritten (no hidden fallback)."""
        assert sanitize_identifier("a\x00b") == "a\x00b"
        with pytest.raises(ValidationError):
            validate_identifier("a\x00b")

    def test_validate_ok(self) -> None:
        assert validate_identifier("pump-01.vib_x") == "pump-01.vib_x"

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            validate_identifier("   ")

    def test_rejects_space(self) -> None:
        with pytest.raises(ValidationError):
            validate_identifier("pump 01")

    def test_rejects_newline(self) -> None:
        with pytest.raises(ValidationError):
            validate_identifier("pump\n01")

    def test_rejects_backslash(self) -> None:
        with pytest.raises(ValidationError):
            validate_identifier("..\\..\\etc")

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ValidationError):
            validate_identifier(42)  # type: ignore[arg-type]

    def test_rejects_overlong(self) -> None:
        with pytest.raises(ValidationError):
            validate_identifier("a" * 200)


class TestRanges:
    def test_within(self) -> None:
        assert validate_range(5.0, minimum=0.0, maximum=10.0) == 5.0

    def test_below(self) -> None:
        with pytest.raises(ValidationError):
            validate_range(-1.0, minimum=0.0)

    def test_above(self) -> None:
        with pytest.raises(ValidationError):
            validate_range(11.0, maximum=10.0)

    def test_probability(self) -> None:
        assert validate_probability(0.5) == 0.5
        with pytest.raises(ValidationError):
            validate_probability(1.5)

    def test_positive_int(self) -> None:
        assert validate_positive_int(3) == 3
        with pytest.raises(ValidationError):
            validate_positive_int(0)
        with pytest.raises(ValidationError):
            validate_positive_int(True)  # type: ignore[arg-type]

    def test_positive_float_bounded(self) -> None:
        assert validate_positive_float(2.0, maximum=5.0) == 2.0
        with pytest.raises(ValidationError):
            validate_positive_float(9.0, maximum=5.0)


class TestTags:
    def test_ok(self) -> None:
        assert validate_tags({"a": "b"}) == {"a": "b"}

    def test_none_becomes_empty(self) -> None:
        assert validate_tags(None) == {}

    def test_rejects_non_mapping(self) -> None:
        with pytest.raises(ValidationError):
            validate_tags(["a"])  # type: ignore[arg-type]

    def test_rejects_too_many(self) -> None:
        with pytest.raises(ValidationError):
            validate_tags({f"k{i}": "v" for i in range(100)})

    def test_strips_control_characters(self) -> None:
        assert validate_tags({"a": "b\x00c"}) == {"a": "bc"}


class TestPayload:
    def test_within_limit(self) -> None:
        assert validate_payload_size(1024) == 1024

    def test_over_limit(self) -> None:
        with pytest.raises(ValidationError):
            validate_payload_size(MAX_PAYLOAD_BYTES + 1)

    def test_negative(self) -> None:
        with pytest.raises(ValidationError):
            validate_payload_size(-1)


class TestSerialization:
    def test_encode_is_sorted(self) -> None:
        assert encode({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValidationError):
            encode({"a": float("nan")})

    def test_nested_nan_rejected(self) -> None:
        with pytest.raises(ValidationError):
            encode({"a": {"b": float("inf")}})

    def test_decode_rejects_nan_literal(self) -> None:
        with pytest.raises(ValidationError):
            decode('{"a": NaN}')

    def test_encoder_handles_enums(self) -> None:
        from sofia_ai.core.quality import DataQuality

        assert json.loads(json.dumps({"q": DataQuality.GOOD}, cls=SofiaJSONEncoder))["q"] == "GOOD"

    def test_encoder_handles_contract(self) -> None:
        from sofia_ai.core.contracts import TelemetrySample

        sample = TelemetrySample(timestamp=1.7e9, device_id="d", channel="c", value=1.0,
                                 unit="g")
        assert json.loads(json.dumps({"s": sample}, cls=SofiaJSONEncoder))["s"]["value"] == 1.0

    def test_json_roundtrip_file(self, tmp_path) -> None:
        path = tmp_path / "x.json"
        dump_json({"b": 1, "a": 2}, str(path))
        assert load_json(str(path)) == {"a": 2, "b": 1}

    def test_jsonl_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "x.jsonl"
        records = [{"i": i} for i in range(5)]
        assert write_jsonl(records, str(path)) == 5
        assert len(read_jsonl(str(path))) == 5

    def test_jsonl_skips_blank_lines(self, tmp_path) -> None:
        path = tmp_path / "x.jsonl"
        path.write_text('{"a":1}\n\n{"a":2}\n', encoding="utf-8")
        assert len(read_jsonl(str(path))) == 2

    def test_jsonl_rejects_non_object(self, tmp_path) -> None:
        path = tmp_path / "x.jsonl"
        path.write_text("[1,2]\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            read_jsonl(str(path))

    def test_jsonl_is_bounded(self, tmp_path) -> None:
        path = tmp_path / "x.jsonl"
        path.write_text('{"a":1}\n' * 100, encoding="utf-8")
        assert len(read_jsonl(str(path), max_lines=10)) == 10


class TestHashing:
    def test_bytes_digest(self) -> None:
        assert len(sha256_bytes(b"abc")) == 64

    def test_file_digest(self, tmp_path) -> None:
        path = tmp_path / "f.bin"
        path.write_bytes(b"abc")
        assert sha256_file(str(path)) == sha256_bytes(b"abc")

    def test_json_digest_is_stable(self) -> None:
        assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})

    def test_verify_checksum_ok(self, tmp_path) -> None:
        path = tmp_path / "f.bin"
        path.write_bytes(b"abc")
        assert verify_checksum(str(path), sha256_bytes(b"abc"))

    def test_verify_checksum_mismatch(self, tmp_path) -> None:
        path = tmp_path / "f.bin"
        path.write_bytes(b"abd")
        assert not verify_checksum(str(path), sha256_bytes(b"abc"))

    def test_verify_rejects_malformed_expected(self, tmp_path) -> None:
        path = tmp_path / "f.bin"
        path.write_bytes(b"abc")
        assert not verify_checksum(str(path), "not-a-digest")


def test_math_helpers_are_used() -> None:
    assert math.isfinite(1.0)
