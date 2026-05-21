"""Runtime exception hierarchy for the pusher module.

``ConfigurationError`` is defined in ``michelangelo.workflow.schema.exceptions``
(the schema layer) and re-exported here so that callers importing from this
module continue to work without modification.
"""

from __future__ import annotations

from michelangelo.workflow.schema.exceptions import ConfigurationError

__all__ = [
    "ArtifactNotFoundError",
    "ConfigurationError",
    "PusherError",
    "PusherPluginError",
]


class PusherError(Exception):
    """Base exception class for all pusher runtime errors.

    All exceptions raised by the pusher module at execution time inherit from
    this class, allowing callers to catch the full family with a single
    ``except PusherError`` clause.

    Note that ``ConfigurationError`` (raised at config-validation time) is
    intentionally *not* a subclass of ``PusherError`` — it originates from
    the schema layer, not the runtime layer.
    """


class ArtifactNotFoundError(PusherError):
    """Raised when an artifact named in config is absent from the artifacts dict.

    Args:
        name: The artifact name that was expected.
        available: The artifact names that are actually present in the dict.

    Example:
        >>> err = ArtifactNotFoundError("model", ["dataset", "report"])
        >>> "model" in str(err)
        True
    """

    def __init__(self, name: str, available: list[str]) -> None:
        """Initialize with the missing artifact name and available names."""
        super().__init__(f"Artifact '{name}' not found. Available: {available}")


class PusherPluginError(PusherError):
    """Raised when a plugin's ``execute()`` raises an unexpected exception.

    This exception is raised by ``push()`` when ``fail_fast=True`` and a
    plugin raises. The original exception is chained via the ``__cause__``
    attribute so the full stack trace is preserved.

    Args:
        artifact_name: The artifact name the plugin was processing.
        plugin_name: The name of the plugin that raised.

    Example:
        >>> err = PusherPluginError("model", "model_plugin")
        >>> "model_plugin" in str(err)
        True
    """

    def __init__(self, artifact_name: str, plugin_name: str) -> None:
        """Initialize with the artifact name and plugin name that failed."""
        super().__init__(
            f"Plugin '{plugin_name}' failed for artifact '{artifact_name}'."
        )
