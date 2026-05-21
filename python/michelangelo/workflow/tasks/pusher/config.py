"""Configuration dataclasses for the pusher module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from michelangelo.workflow.tasks.pusher.exceptions import ConfigurationError


class DatasetFormat(Enum):
    """Supported output formats for ``DatasetPusherPlugin``.

    Attributes:
        CSV: Comma-separated values text format.
        PARQUET: Columnar binary format (recommended for large datasets).
        JSON: JSON lines format — one record per line.

    Example:
        >>> DatasetFormat.PARQUET.value
        'parquet'
    """

    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"


@dataclass
class ModelPluginConfig:
    """Configuration for ``ModelPusherPlugin``.

    Attributes:
        model_name: Name to register the model under in the registry. A
            unique name is generated automatically when ``None``.
        description: Optional human-readable description stored in the
            registry alongside the model.
        extra_metadata: Additional string key-value pairs forwarded to
            the registry at registration time.

    Example:
        >>> cfg = ModelPluginConfig(model_name="boston-xgb")
        >>> cfg.model_name
        'boston-xgb'
        >>> cfg.extra_metadata
        {}
    """

    model_name: str | None = None
    description: str | None = None
    extra_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetPluginConfig:
    """Configuration for ``DatasetPusherPlugin``.

    Attributes:
        destination_path: Local directory path where the output file is
            written. Created automatically if absent.
        format: Output format. Defaults to ``DatasetFormat.PARQUET``.
        partition_by: Optional list of column names to partition the
            output by. Unused by the built-in file writer; available
            for provider subclasses.

    Example:
        >>> cfg = DatasetPluginConfig(destination_path="/tmp/data")
        >>> cfg.format
        <DatasetFormat.PARQUET: 'parquet'>
    """

    destination_path: str = ""
    format: DatasetFormat = DatasetFormat.PARQUET
    partition_by: list[str] = field(default_factory=list)


@dataclass
class EvalReportPluginConfig:
    """Configuration for ``EvalReportPusherPlugin``.

    Attributes:
        report_name: Name assigned to the evaluation report. A unique
            name is generated automatically when ``None``.
        extra_fields: Additional key-value pairs merged into the report
            document before it is written.

    Example:
        >>> cfg = EvalReportPluginConfig(report_name="run-2026")
        >>> cfg.extra_fields
        {}
    """

    report_name: str | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class PusherPluginConfig:
    """Configuration for a single artifact push within a ``PusherConfig``.

    Exactly one of the typed plugin config fields or the extension pair
    (``plugin_name`` + ``plugin_config``) must be set. Setting zero or
    more than one raises ``ConfigurationError``.

    Attributes:
        name: Artifact identifier matching a key in the ``artifacts``
            dict passed to ``push()``.
        model_plugin: Config for ``ModelPusherPlugin``.
        dataset_plugin: Config for ``DatasetPusherPlugin``.
        eval_report_plugin: Config for ``EvalReportPusherPlugin``.
        plugin_name: Name of a provider-registered plugin. Use together
            with ``plugin_config`` when extending beyond the three
            built-in plugins.
        plugin_config: Raw configuration dict forwarded to the custom
            plugin named by ``plugin_name``.

    Example:
        >>> from michelangelo.workflow.tasks.pusher.config import (
        ...     ModelPluginConfig, PusherPluginConfig
        ... )
        >>> cfg = PusherPluginConfig(
        ...     name="clf",
        ...     model_plugin=ModelPluginConfig(model_name="my-clf"),
        ... )
        >>> cfg.resolved_plugin_name()
        'model_plugin'
    """

    name: str
    model_plugin: ModelPluginConfig | None = None
    dataset_plugin: DatasetPluginConfig | None = None
    eval_report_plugin: EvalReportPluginConfig | None = None
    plugin_name: str | None = None
    plugin_config: dict[str, Any] | None = None

    _BUILTIN_FIELDS: ClassVar[tuple[str, ...]] = (
        "model_plugin",
        "dataset_plugin",
        "eval_report_plugin",
    )

    def resolved_plugin_name(self) -> str:
        """Return the name of the active plugin.

        Returns:
            Plugin name string — one of the built-in field names
            (e.g. ``"model_plugin"``) or the value of ``plugin_name``
            for provider-registered plugins.

        Raises:
            ConfigurationError: If zero plugins are specified, more than
                one built-in field is set, or both a typed field and
                ``plugin_name`` are set simultaneously.

        Example:
            >>> cfg = PusherPluginConfig(
            ...     name="report",
            ...     eval_report_plugin=EvalReportPluginConfig(),
            ... )
            >>> cfg.resolved_plugin_name()
            'eval_report_plugin'
        """
        active = [f for f in self._BUILTIN_FIELDS if getattr(self, f) is not None]
        if len(active) > 1:
            raise ConfigurationError(
                f"Artifact '{self.name}' has multiple plugin configs set: {active}. "
                "Exactly one must be specified."
            )
        if len(active) == 1:
            if self.plugin_name is not None:
                raise ConfigurationError(
                    f"Artifact '{self.name}' sets both '{active[0]}' and "
                    "'plugin_name'. Use exactly one."
                )
            return active[0]
        if self.plugin_name:
            return self.plugin_name
        raise ConfigurationError(
            f"No plugin specified for artifact '{self.name}'. "
            f"Set one of: {list(self._BUILTIN_FIELDS)}, or set plugin_name."
        )

    def resolved_plugin_config(self) -> Any:
        """Return the typed config for built-in plugins or raw dict for custom ones.

        For built-in plugins, returns the typed config dataclass (e.g.
        ``ModelPluginConfig``). For provider-registered plugins where
        ``plugin_name`` is set to a name that is not a dataclass field,
        returns ``plugin_config`` (the raw dict).

        Returns:
            The active plugin config — a typed dataclass or a ``dict``.

        Raises:
            ConfigurationError: Propagated from ``resolved_plugin_name()``
                if the plugin specification is ambiguous or missing.

        Example:
            >>> raw = {"table": "ml_evals"}
            >>> cfg = PusherPluginConfig(
            ...     name="data",
            ...     plugin_name="hive_plugin",
            ...     plugin_config=raw,
            ... )
            >>> cfg.resolved_plugin_config() is raw
            True
        """
        plugin = self.resolved_plugin_name()
        typed = getattr(self, plugin, None)
        return typed if typed is not None else self.plugin_config


@dataclass
class PusherConfig:
    """Top-level configuration for a ``push()`` call.

    Attributes:
        items: Ordered list of artifact push configurations. Each item
            maps one artifact name to one plugin.

    Example:
        >>> cfg = PusherConfig(items=[
        ...     PusherPluginConfig(
        ...         name="model",
        ...         model_plugin=ModelPluginConfig(),
        ...     )
        ... ])
        >>> len(cfg.items)
        1
    """

    items: list[PusherPluginConfig] = field(default_factory=list)
