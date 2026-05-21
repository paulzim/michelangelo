"""Tests for the pusher exceptions module."""

from __future__ import annotations

from unittest import TestCase

from michelangelo.workflow.schema.exceptions import ConfigurationError
from michelangelo.workflow.tasks.pusher.exceptions import (
    ArtifactNotFoundError,
    PusherError,
    PusherPluginError,
)


class TestPusherError(TestCase):
    """Tests for the PusherError base exception."""

    def test_is_exception(self):
        """It is a subclass of Exception."""
        self.assertTrue(issubclass(PusherError, Exception))

    def test_can_be_raised_and_caught(self):
        """It can be raised and caught as a PusherError."""
        with self.assertRaises(PusherError):
            raise PusherError("base error")


class TestArtifactNotFoundError(TestCase):
    """Tests for ArtifactNotFoundError."""

    def test_is_pusher_error(self):
        """It is a subclass of PusherError."""
        self.assertTrue(issubclass(ArtifactNotFoundError, PusherError))

    def test_message_includes_missing_name(self):
        """It includes the missing artifact name in the error message."""
        err = ArtifactNotFoundError("model", ["dataset"])
        self.assertIn("model", str(err))

    def test_message_includes_all_available_names(self):
        """It includes all available artifact names in the error message."""
        err = ArtifactNotFoundError("model", ["dataset", "report"])
        self.assertIn("dataset", str(err))
        self.assertIn("report", str(err))

    def test_empty_available_list(self):
        """It handles an empty available list without raising."""
        err = ArtifactNotFoundError("missing", [])
        self.assertIn("missing", str(err))

    def test_can_be_caught_as_pusher_error(self):
        """It can be caught via its PusherError base class."""
        with self.assertRaises(PusherError):
            raise ArtifactNotFoundError("x", [])


class TestConfigurationError(TestCase):
    """Tests for ConfigurationError (defined in workflow.schema.exceptions)."""

    def test_is_exception(self):
        """It is a subclass of Exception."""
        self.assertTrue(issubclass(ConfigurationError, Exception))

    def test_is_not_pusher_error(self):
        """It is a schema-layer exception, not a runtime PusherError subclass."""
        self.assertFalse(issubclass(ConfigurationError, PusherError))

    def test_message_is_preserved_exactly(self):
        """It preserves the provided message without modification."""
        msg = "No plugin specified for artifact 'model'."
        self.assertEqual(str(ConfigurationError(msg)), msg)

    def test_can_be_raised_and_caught(self):
        """It can be raised and caught as ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            raise ConfigurationError("bad config")


class TestPusherPluginError(TestCase):
    """Tests for PusherPluginError."""

    def test_is_pusher_error(self):
        """It is a subclass of PusherError."""
        self.assertTrue(issubclass(PusherPluginError, PusherError))

    def test_message_includes_plugin_name(self):
        """It includes the plugin name in the error message."""
        err = PusherPluginError("my_artifact", "model_plugin")
        self.assertIn("model_plugin", str(err))

    def test_message_includes_artifact_name(self):
        """It includes the artifact name in the error message."""
        err = PusherPluginError("my_artifact", "model_plugin")
        self.assertIn("my_artifact", str(err))

    def test_can_be_chained(self):
        """It can be chained from an original exception."""
        original = ValueError("upload failed")
        try:
            raise PusherPluginError("art", "plug") from original
        except PusherPluginError as e:
            self.assertIs(e.__cause__, original)


class TestExceptionsShimIdentity(TestCase):
    """Tests that pusher/exceptions.py re-exports ConfigurationError correctly."""

    def test_configuration_error_is_same_object_as_schema(self):
        """ConfigurationError re-exported from shim is identical to schema class."""
        from michelangelo.workflow.schema.exceptions import (
            ConfigurationError as SchemaConfigurationError,
        )
        from michelangelo.workflow.tasks.pusher.exceptions import (
            ConfigurationError as ShimConfigurationError,
        )

        self.assertIs(ShimConfigurationError, SchemaConfigurationError)
