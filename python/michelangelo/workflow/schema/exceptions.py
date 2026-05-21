"""Schema-layer exceptions raised during workflow task configuration validation."""

from __future__ import annotations

__all__ = ["ConfigurationError"]


class ConfigurationError(Exception):
    """Raised when a workflow task config dataclass fails validation.

    Raised by ``__post_init__`` methods on config dataclasses (e.g.
    ``PusherPluginConfig``, ``PusherConfig``) and by resolved-name/config
    helpers when the configuration is ambiguous or incomplete.

    This exception is defined at the schema layer and is intentionally *not*
    a subclass of ``PusherError``. Configuration errors occur at construction
    time, before any push execution begins. Callers that previously caught
    ``PusherError`` to handle all pusher-related failures should also catch
    ``ConfigurationError`` separately if they construct config objects inside
    the same ``try`` block.

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
