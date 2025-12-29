"""Tests for the utils module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nyrag.utils import (
    DEFAULT_CLOUD_PORT,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LOCAL_PORT,
    chunks,
    get_vespa_tls_config,
    is_vespa_cloud,
    resolve_vespa_cloud_mtls_paths,
    resolve_vespa_port,
)


class TestIsVespaCloud:
    """Tests for is_vespa_cloud function."""

    def test_https_url_is_cloud(self):
        """Test that HTTPS URLs are detected as cloud."""
        assert is_vespa_cloud("https://example.vespa-app.cloud") is True

    def test_http_url_is_not_cloud(self):
        """Test that HTTP URLs are not detected as cloud."""
        assert is_vespa_cloud("http://localhost") is False

    def test_local_url_is_not_cloud(self):
        """Test that local URLs are not detected as cloud."""
        assert is_vespa_cloud("http://localhost:8080") is False

    @patch.dict(os.environ, {"NYRAG_LOCAL": "1"})
    def test_nyrag_local_env_override(self):
        """Test that NYRAG_LOCAL=1 forces local mode."""
        assert is_vespa_cloud("https://example.vespa-app.cloud") is False

    @patch.dict(os.environ, {"NYRAG_LOCAL": "true"})
    def test_nyrag_local_env_true(self):
        """Test that NYRAG_LOCAL=true forces local mode."""
        assert is_vespa_cloud("https://example.vespa-app.cloud") is False

    @patch.dict(os.environ, {"NYRAG_LOCAL": "0"})
    def test_nyrag_local_env_false(self):
        """Test that NYRAG_LOCAL=0 allows cloud detection."""
        assert is_vespa_cloud("https://example.vespa-app.cloud") is True

    @patch.dict(os.environ, {}, clear=True)
    def test_empty_url(self):
        """Test that empty URL defaults to local."""
        assert is_vespa_cloud("") is False


class TestResolveVespaPort:
    """Tests for resolve_vespa_port function."""

    @patch.dict(os.environ, {}, clear=True)
    def test_cloud_url_default_port(self):
        """Test that cloud URLs use default cloud port."""
        port = resolve_vespa_port("https://example.vespa-app.cloud")
        assert port == DEFAULT_CLOUD_PORT

    @patch.dict(os.environ, {}, clear=True)
    def test_local_url_default_port(self):
        """Test that local URLs use default local port."""
        port = resolve_vespa_port("http://localhost")
        assert port == DEFAULT_LOCAL_PORT

    @patch.dict(os.environ, {"VESPA_PORT": "9090"})
    def test_env_port_override(self):
        """Test that VESPA_PORT env var overrides defaults."""
        port = resolve_vespa_port("http://localhost")
        assert port == 9090

    @patch.dict(os.environ, {"VESPA_PORT": "9090"})
    def test_env_port_override_for_cloud(self):
        """Test that VESPA_PORT env var overrides cloud default."""
        port = resolve_vespa_port("https://example.vespa-app.cloud")
        assert port == 9090


class TestResolveVespaCloudMtlsPaths:
    """Tests for resolve_vespa_cloud_mtls_paths function."""

    def test_mtls_paths(self):
        """Test that mTLS paths are resolved correctly."""
        cert_path, key_path = resolve_vespa_cloud_mtls_paths("my-project")
        expected_base = Path.home() / ".vespa" / "devrel-public.my-project.default"
        assert cert_path == expected_base / "data-plane-public-cert.pem"
        assert key_path == expected_base / "data-plane-private-key.pem"


class TestGetVespaTlsConfig:
    """Tests for get_vespa_tls_config function."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_env_vars(self):
        """Test with no environment variables set."""
        cert, key, ca, verify = get_vespa_tls_config()
        assert cert is None
        assert key is None
        assert ca is None
        assert verify is None

    @patch.dict(
        os.environ,
        {
            "VESPA_CLIENT_CERT": "/path/to/cert.pem",
            "VESPA_CLIENT_KEY": "/path/to/key.pem",
        },
    )
    def test_with_cert_and_key(self):
        """Test with cert and key environment variables."""
        cert, key, ca, verify = get_vespa_tls_config()
        assert cert == "/path/to/cert.pem"
        assert key == "/path/to/key.pem"
        assert ca is None
        assert verify is None

    @patch.dict(os.environ, {"VESPA_CA_CERT": "/path/to/ca.pem"})
    def test_with_ca_cert(self):
        """Test with CA cert environment variable."""
        cert, key, ca, verify = get_vespa_tls_config()
        assert ca == "/path/to/ca.pem"

    @patch.dict(os.environ, {"VESPA_TLS_VERIFY": "false"})
    def test_verify_false(self):
        """Test with VESPA_TLS_VERIFY=false."""
        cert, key, ca, verify = get_vespa_tls_config()
        assert verify is False

    @patch.dict(os.environ, {"VESPA_TLS_VERIFY": "/path/to/ca-bundle.crt"})
    def test_verify_custom_path(self):
        """Test with custom CA bundle path."""
        cert, key, ca, verify = get_vespa_tls_config()
        assert verify == "/path/to/ca-bundle.crt"


class TestChunks:
    """Tests for chunks utility function."""

    def test_basic_chunking(self):
        """Test basic text chunking without overlap."""
        text = " ".join(["word"] * 100)  # 100 words
        result = list(chunks(text, chunk_size=10, overlap=0))
        assert len(result) == 10
        assert all(len(chunk.split()) == 10 for chunk in result)

    def test_chunking_with_overlap(self):
        """Test text chunking with overlap."""
        text = " ".join(["word"] * 50)  # 50 words
        result = list(chunks(text, chunk_size=10, overlap=5))
        assert len(result) > 5

    def test_text_shorter_than_chunk_size(self):
        """Test with text shorter than chunk size."""
        text = "short text"
        result = list(chunks(text, chunk_size=10, overlap=0))
        assert len(result) == 1
        assert result[0] == "short text"

    def test_empty_text(self):
        """Test with empty text."""
        text = ""
        result = list(chunks(text, chunk_size=10, overlap=0))
        assert len(result) == 1
        assert result[0] == ""

    def test_exact_chunk_size(self):
        """Test with text exactly matching chunk size."""
        text = " ".join(["word"] * 10)
        result = list(chunks(text, chunk_size=10, overlap=0))
        assert len(result) == 1

    def test_overlap_larger_than_chunk_size(self):
        """Test with overlap larger than chunk size (edge case)."""
        text = " ".join(["word"] * 100)
        # When overlap >= chunk_size, it should raise ValueError
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            list(chunks(text, chunk_size=10, overlap=15))

    def test_word_boundary_preservation(self):
        """Test that chunks work on word boundaries."""
        text = "hello world this is a test with more words"
        result = list(chunks(text, chunk_size=3, overlap=0))
        # Should split on word boundaries
        assert len(result) >= 2


class TestConstants:
    """Tests for module constants."""

    def test_default_embedding_model(self):
        """Test default embedding model constant."""
        assert DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"

    def test_default_embedding_dim(self):
        """Test default embedding dimension constant."""
        assert DEFAULT_EMBEDDING_DIM == 384

    def test_default_ports(self):
        """Test default port constants."""
        assert DEFAULT_LOCAL_PORT == 8080
        assert DEFAULT_CLOUD_PORT == 443
