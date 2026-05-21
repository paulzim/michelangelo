"""Tests for ModelPusherPlugin."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock, call

from michelangelo.workflow.schema.exceptions import ConfigurationError
from michelangelo.workflow.schema.pusher import ModelPluginConfig
from michelangelo.workflow.tasks.pusher.plugins.model_plugin import ModelPusherPlugin
from michelangelo.workflow.variables.metadata import ModelMetadata
from michelangelo.workflow.variables.types import AssembledModel, ModelArtifact


def _make_artifact(
    raw_path: str = "/tmp/raw",
    dep_path: str = "/tmp/dep",
    metadata: ModelMetadata | None = None,
) -> AssembledModel:
    """Return an AssembledModel with MagicMock-friendly paths."""
    meta = metadata or ModelMetadata(training_framework="xgboost", deployable=True)
    return AssembledModel(
        raw_model=ModelArtifact(path=raw_path, metadata=meta),
        deployable_model=ModelArtifact(path=dep_path, metadata=meta),
    )


def _make_registry(name: str = "test-model", version: str = "1") -> MagicMock:
    """Return a MagicMock registry client with a pre-configured return value."""
    rc = MagicMock()
    rc.register_model.return_value = MagicMock(name=name, version=version)
    return rc


def _make_plugin(
    model_name: str | None = "test-model",
    artifact: AssembledModel | None = None,
    storage_backend: object | None = None,
    registry_client: object | None = None,
) -> ModelPusherPlugin:
    """Return a ModelPusherPlugin with sensible defaults."""
    return ModelPusherPlugin(
        config=ModelPluginConfig(model_name=model_name),
        artifact=artifact or _make_artifact(),
        storage_backend=storage_backend or MagicMock(),
        registry_client=registry_client or _make_registry(name=model_name or "model-x"),
    )


class TestModelPusherPluginExecute(TestCase):
    """Tests for ModelPusherPlugin.execute()."""

    def test_uploads_both_artifacts_and_registers_once(self):
        """It calls upload() twice and register_model() once."""
        sb = MagicMock()
        rc = _make_registry()
        plugin = _make_plugin(storage_backend=sb, registry_client=rc)
        plugin.execute()
        self.assertEqual(sb.upload.call_count, 2)
        self.assertEqual(rc.register_model.call_count, 1)

    def test_returns_four_key_dict(self):
        """It returns a dict with exactly the four documented keys."""
        result = _make_plugin().execute()
        self.assertEqual(
            set(result.keys()),
            {"model_name", "version", "raw_artifact_uri", "deployable_artifact_uri"},
        )

    def test_uses_config_model_name(self):
        """It registers the model under the name from ModelPluginConfig.model_name."""
        rc = _make_registry(name="pricing-clf")
        plugin = _make_plugin(model_name="pricing-clf", registry_client=rc)
        plugin.execute()
        name_kwarg = rc.register_model.call_args.kwargs["name"]
        self.assertEqual(name_kwarg, "pricing-clf")

    def test_generates_name_when_none(self):
        """It generates a model name starting with 'model-' when model_name is None."""
        rc = MagicMock()
        rc.register_model.return_value = MagicMock(name="captured", version="1")
        plugin = _make_plugin(model_name=None, registry_client=rc)
        plugin.execute()
        generated = rc.register_model.call_args.kwargs["name"]
        self.assertTrue(generated.startswith("model-"))

    def test_merges_metadata_and_extra_metadata(self):
        """It merges ModelMetadata fields and extra_metadata into register_model()."""
        meta = ModelMetadata(training_framework="pytorch", deployable=True)
        rc = _make_registry()
        plugin = ModelPusherPlugin(
            config=ModelPluginConfig(model_name="m", extra_metadata={"team": "search"}),
            artifact=_make_artifact(metadata=meta),
            storage_backend=MagicMock(),
            registry_client=rc,
        )
        plugin.execute()
        metadata_kwarg = rc.register_model.call_args.kwargs["metadata"]
        self.assertEqual(metadata_kwarg["training_framework"], "pytorch")
        self.assertEqual(metadata_kwarg["team"], "search")
        self.assertIn("deployable", metadata_kwarg)

    def test_upload_order_raw_before_deployable(self):
        """It uploads the raw artifact before the deployable artifact."""
        sb = MagicMock()
        sb.upload.side_effect = ["/store/raw", "/store/dep"]
        plugin = ModelPusherPlugin(
            config=ModelPluginConfig(model_name="m"),
            artifact=_make_artifact(raw_path="/local/raw", dep_path="/local/dep"),
            storage_backend=sb,
            registry_client=_make_registry(name="m"),
        )
        plugin.execute()
        calls = sb.upload.call_args_list
        self.assertEqual(calls[0], call("/local/raw", "models/m/raw"))
        self.assertEqual(calls[1], call("/local/dep", "models/m/deployable"))

    def test_result_uris_match_upload_return_values(self):
        """It passes upload() return values to register_model() and result dict."""
        sb = MagicMock()
        sb.upload.side_effect = ["/store/raw-uri", "/store/dep-uri"]
        rc = _make_registry()
        plugin = _make_plugin(storage_backend=sb, registry_client=rc)
        result = plugin.execute()
        self.assertEqual(result["raw_artifact_uri"], "/store/raw-uri")
        self.assertEqual(result["deployable_artifact_uri"], "/store/dep-uri")
        self.assertEqual(
            rc.register_model.call_args.kwargs["artifact_uri"], "/store/raw-uri"
        )
        self.assertEqual(
            rc.register_model.call_args.kwargs["deployable_artifact_uri"],
            "/store/dep-uri",
        )


class TestModelPusherPluginInit(TestCase):
    """Tests for ModelPusherPlugin.__init__() validation."""

    def test_raises_when_storage_backend_none(self):
        """It raises ConfigurationError at __init__ when storage_backend is None."""
        with self.assertRaises(ConfigurationError) as ctx:
            ModelPusherPlugin(
                config=ModelPluginConfig(),
                artifact=_make_artifact(),
                storage_backend=None,
                registry_client=MagicMock(),
            )
        self.assertIn("storage_backend", str(ctx.exception))

    def test_raises_when_registry_client_none(self):
        """It raises ConfigurationError at __init__ when registry_client is None."""
        with self.assertRaises(ConfigurationError) as ctx:
            ModelPusherPlugin(
                config=ModelPluginConfig(),
                artifact=_make_artifact(),
                storage_backend=MagicMock(),
                registry_client=None,
            )
        self.assertIn("registry_client", str(ctx.exception))
