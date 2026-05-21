"""Schema package for workflow task configuration dataclasses.

Callers should import from the submodules directly:

    from michelangelo.workflow.schema.pusher import PusherConfig
    from michelangelo.workflow.schema.exceptions import ConfigurationError

This ``__init__.py`` intentionally exports nothing — import paths are kept
explicit so that adding future task schemas (trainer, evaluator, etc.) does
not create an ever-growing top-level namespace.
"""
