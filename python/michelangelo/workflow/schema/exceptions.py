"""Schema-layer exceptions raised during workflow task configuration validation."""

from __future__ import annotations


class ConfigurationError(Exception):
    """Raised when a workflow task config dataclass fails validation.

    Raised by ``__post_init__`` methods on config dataclasses (e.g.
    ``PusherPluginConfig``, ``PusherConfig``) and by resolved-name/config
    helpers when the configuration is ambiguous or incomplete.

    This exception is intentionally defined at the schema layer rather than
    inside a specific task package so that ``workflow/schema/`` remains
    independent of any task implementation.

    Args:
        message: Human-readable description of the configuration problem.

    Example:
        >>> err = ConfigurationError("No plugin specified for artifact 'model'.")
        >>> str(err)
        "No plugin specified for artifact 'model'."
    """

    def __init__(self, message: str) -> None:
        """Initialize with a human-readable configuration error message."""
        super().__init__(message)
