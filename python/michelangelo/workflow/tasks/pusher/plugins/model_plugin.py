"""ModelPusherPlugin — uploads and registers a trained model artifact."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from michelangelo.workflow.schema.exceptions import ConfigurationError
from michelangelo.workflow.tasks.pusher.plugins.base import PusherPluginBase

if TYPE_CHECKING:
    from michelangelo.lib.artifact_manager.storage_backend import StorageBackend
    from michelangelo.lib.model_manager.registry.client import ModelRegistryClient
    from michelangelo.workflow.schema.pusher import ModelPluginConfig
    from michelangelo.workflow.variables.types import AssembledModel

_logger = logging.getLogger(__name__)


class ModelPusherPlugin(PusherPluginBase):
    """Plugin that uploads a trained model and registers it in a model registry.

    Uploads both the raw model artifact and the deployable artifact to the
    configured storage backend, then registers the model in the registry.
    Artifacts must already be packaged before this plugin is invoked —
    packaging is the assembler's responsibility.

    Args:
        config: ``ModelPluginConfig`` with optional ``model_name``,
            ``description``, and ``extra_metadata``.
        artifact: An ``AssembledModel`` with pre-packaged ``raw_model``
            and ``deployable_model`` artifacts.
        storage_backend: Backend for uploading artifacts. Required — raises
            ``ConfigurationError`` at construction if ``None``.
        registry_client: Registry for registering the uploaded model. Required
            — raises ``ConfigurationError`` at construction if ``None``.

    Raises:
        ConfigurationError: If ``storage_backend`` or ``registry_client``
            is ``None`` at construction time.

    Example::

        plugin = ModelPusherPlugin(
            config=ModelPluginConfig(model_name="boston-xgb"),
            artifact=assembled_model,
            storage_backend=LocalStorageBackend("/tmp/store"),
            registry_client=my_registry,
        )
        result = plugin.execute()
        print(result["model_name"], result["version"])
    """

    def __init__(
        self,
        config: ModelPluginConfig,
        artifact: AssembledModel | None = None,
        storage_backend: StorageBackend | None = None,
        registry_client: ModelRegistryClient | None = None,
    ) -> None:
        """Validate required dependencies then store all as protected attributes."""
        super().__init__(config, artifact, storage_backend, registry_client)
        if storage_backend is None:
            raise ConfigurationError(
                "ModelPusherPlugin requires a storage_backend. "
                "Pass a StorageBackend implementation to push()."
            )
        if registry_client is None:
            raise ConfigurationError(
                "ModelPusherPlugin requires a registry_client. "
                "Pass a ModelRegistryClient implementation to push()."
            )

    def execute(self) -> dict[str, Any]:
        """Upload model artifacts and register the model in the registry.

        Uploads the raw artifact first, then the deployable artifact, then
        calls ``register_model()``. Upload order is observable via
        ``storage_backend.upload.call_args_list`` in tests.

        Returns:
            A dict with exactly four keys:

            - ``model_name``: Name under which the model was registered.
            - ``version``: Registry-assigned version string.
            - ``raw_artifact_uri``: URI of the uploaded raw model artifact.
            - ``deployable_artifact_uri``: URI of the uploaded deployable
              artifact.

        Raises:
            IOError: If the storage backend or registry cannot be reached.
        """
        model_name = self._config.model_name or self._generate_name()

        _logger.info("Uploading raw model artifact for '%s'.", model_name)
        raw_uri = self._storage_backend.upload(
            self._artifact.raw_model.path,
            f"models/{model_name}/raw",
        )

        _logger.info("Uploading deployable artifact for '%s'.", model_name)
        deployable_uri = self._storage_backend.upload(
            self._artifact.deployable_model.path,
            f"models/{model_name}/deployable",
        )

        metadata = self._build_metadata_dict()

        _logger.info("Registering '%s' in model registry.", model_name)
        registered = self._registry_client.register_model(
            name=model_name,
            artifact_uri=raw_uri,
            deployable_artifact_uri=deployable_uri,
            metadata=metadata,
        )

        return {
            "model_name": registered.name,
            "version": registered.version,
            "raw_artifact_uri": raw_uri,
            "deployable_artifact_uri": deployable_uri,
        }

    def _build_metadata_dict(self) -> dict[str, str]:
        """Flatten ModelMetadata public fields and merge with extra_metadata.

        Extracts only the public string/bool fields from ModelMetadata —
        skipping the private BytesIO payload fields (_schema, _sample_data,
        _hyperparameters) which are binary and unsuitable as registry tags.
        Overridable by provider subclasses that add proprietary metadata fields.

        Returns:
            A ``dict[str, str]`` of merged metadata suitable for the registry.
        """
        m = self._artifact.raw_model.metadata
        base: dict[str, str] = {}
        if m.training_framework is not None:
            base["training_framework"] = m.training_framework
        if m.model_class is not None:
            base["model_class"] = m.model_class
        base["assembled"] = str(m.assembled)
        base["deployable"] = str(m.deployable)
        return {**base, **self._config.extra_metadata}

    @staticmethod
    def _generate_name() -> str:
        """Generate a unique model name with a 'model-' prefix."""
        return f"model-{uuid.uuid4().hex[:8]}"
