"""Unit tests for security controls (SOFIA-SEC-*)."""

from __future__ import annotations

import base64
import os

import pytest

from sofia_ai.core.errors import (
    SecretMissingError,
    SignatureError,
    UnsafePathError,
    ValidationError,
)
from sofia_ai.core.hashing import sha256_bytes
from sofia_ai.security import safeio
from sofia_ai.security import secrets as secret_module
from sofia_ai.security.safeio import (
    atomic_write_bytes,
    resolve_within,
    safe_read_json,
    safe_write_json,
)
from sofia_ai.security.secrets import (
    env_name,
    has_secret,
    is_secret_key,
    require_secret,
    secret,
)
from sofia_ai.security.signing import (
    NullVerifier,
    SignatureVerifier,
    manifest_signing_payload,
    verify_detached,
)


class TestPathConfinement:
    def test_inside_root(self, tmp_path) -> None:
        resolved = resolve_within(tmp_path, "a/b.txt")
        assert str(resolved).startswith(str(tmp_path))

    def test_traversal_rejected(self, tmp_path) -> None:
        with pytest.raises(UnsafePathError):
            resolve_within(tmp_path, "../outside.txt")

    def test_absolute_outside_rejected(self, tmp_path) -> None:
        with pytest.raises(UnsafePathError):
            resolve_within(tmp_path, str(tmp_path.parent / "other"))

    def test_overlong_path(self, tmp_path) -> None:
        with pytest.raises(UnsafePathError):
            resolve_within(tmp_path, "a" * 5000)

    def test_symlink_escape_rejected(self, tmp_path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "f.txt"
        target.write_text("x", encoding="utf-8")
        inside = tmp_path / "root"
        inside.mkdir()
        try:
            os.symlink(str(target), str(inside / "link.txt"))
        except (OSError, NotImplementedError):  # pragma: no cover - Windows perms
            pytest.skip("symlinks unavailable")
        with pytest.raises(UnsafePathError):
            resolve_within(inside, "link.txt")


class TestSafeJson:
    def test_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "x.json"
        safe_write_json(str(path), {"b": 1, "a": 2})
        assert safe_read_json(str(path)) == {"a": 2, "b": 1}

    def test_size_ceiling(self, tmp_path) -> None:
        path = tmp_path / "big.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValidationError):
            safe_read_json(str(path), max_bytes=0)

    def test_invalid_json(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{", encoding="utf-8")
        with pytest.raises(ValidationError):
            safe_read_json(str(path))

    def test_non_object(self, tmp_path) -> None:
        path = tmp_path / "arr.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValidationError):
            safe_read_json(str(path))

    def test_root_confinement(self, tmp_path) -> None:
        with pytest.raises(UnsafePathError):
            safe_read_json("../escape.json", root=str(tmp_path))


class TestAtomicWrite:
    def test_no_temp_file_left(self, tmp_path) -> None:
        path = tmp_path / "f.bin"
        atomic_write_bytes(str(path), b"data")
        assert path.read_bytes() == b"data"
        assert not list(tmp_path.glob("*.tmp"))

    def test_creates_parent(self, tmp_path) -> None:
        path = tmp_path / "sub" / "f.bin"
        atomic_write_bytes(str(path), b"x")
        assert path.exists()


class TestSecrets:
    def test_env_prefix(self) -> None:
        assert env_name("broker_password") == "SOFIA_BROKER_PASSWORD"
        assert env_name("SOFIA_X") == "SOFIA_X"

    def test_missing_is_none(self, monkeypatch) -> None:
        monkeypatch.delenv("SOFIA_ABSENT", raising=False)
        assert secret("absent") is None
        assert not has_secret("absent")

    def test_present(self, monkeypatch) -> None:
        monkeypatch.setenv("SOFIA_PRESENT", "value")
        assert secret("present") == "value"
        assert require_secret("present") == "value"
        assert has_secret("present")

    def test_require_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("SOFIA_MISSING", raising=False)
        with pytest.raises(SecretMissingError):
            require_secret("missing")

    def test_blank_is_absent(self, monkeypatch) -> None:
        monkeypatch.setenv("SOFIA_BLANK", "   ")
        assert secret("blank") is None

    def test_empty_name(self) -> None:
        with pytest.raises(SecretMissingError):
            env_name("")
        with pytest.raises(SecretMissingError):
            require_secret("")

    def test_is_secret_key(self) -> None:
        for key in ("password", "api_key", "token", "authorization", "secret",
                    "private_key", "credential", "bearer", "cookie"):
            assert is_secret_key(key), key
        assert not is_secret_key("host")
        assert not is_secret_key("sample_rate")


class TestSigning:
    def test_signing_payload_excludes_signature(self) -> None:
        manifest = {"model_id": "m", "signature": "abc"}
        payload = manifest_signing_payload(manifest)
        assert b"abc" not in payload

    def test_null_verifier_declines_nothing(self) -> None:
        assert NullVerifier().verify(b"x", "")

    def test_verifier_rejects_bad_algorithm(self) -> None:
        with pytest.raises(SignatureError):
            SignatureVerifier(public_key_b64="a" * 44, algorithm="rsa")

    def test_verifier_rejects_empty_key(self) -> None:
        with pytest.raises(SignatureError):
            SignatureVerifier(public_key_b64="  ")

    def test_requires_signature_material(self) -> None:
        with pytest.raises(SignatureError):
            verify_detached(b"x", "", "a" * 44)

    def test_rejects_invalid_base64(self) -> None:
        with pytest.raises(SignatureError):
            verify_detached(b"x", "!!!", "a" * 44)

    def test_cryptography_absent_raises_typed_error(self) -> None:
        """A missing crypto backend must be a typed error, never a silent skip."""
        import importlib.util

        backend_available = importlib.util.find_spec("cryptography") is not None
        if not backend_available:
            with pytest.raises(SignatureError):
                verify_detached(b"x", base64.b64encode(b"0" * 64).decode(),
                                base64.b64encode(b"0" * 32).decode())
        else:  # pragma: no cover - depends on environment
            with pytest.raises(SignatureError):
                verify_detached(b"x", base64.b64encode(b"0" * 16).decode(),
                                base64.b64encode(b"0" * 32).decode())


class TestNoInsecureDefaults:
    def test_safeio_has_no_pickle(self) -> None:
        with open(safeio.__file__, encoding="utf-8") as handle:
            source = handle.read()
        assert "import pickle" not in source
        assert "pickle.load" not in source

    def test_digest_is_sha256(self) -> None:
        assert len(sha256_bytes(b"")) == 64

    def test_module_exports_are_explicit(self) -> None:
        assert secret_module.__all__
