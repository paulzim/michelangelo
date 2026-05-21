"""Workflow variable types for artifact storage and push results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from michelangelo.workflow.variables.metadata import ModelMetadata

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class ModelArtifact:
    """A packaged model artifact ready for upload.

    Both the raw model package and the serving-ready deployable artifact are
    represented as ``ModelArtifact`` instances. Packaging must be complete
    before passing to the pusher — packaging is an assembler-time concern
    (e.g. a Ray worker with GPU access).

    Attributes:
        path: Absolute local filesystem path to the packaged artifact file or
            directory.
        metadata: Typed metadata forwarded to the model registry at
            registration time. Subclass ``ModelMetadata`` to add
            provider-specific fields.

    Example:
        >>> from michelangelo.workflow.variables.metadata import ModelMetadata
        >>> meta = ModelMetadata(training_framework="xgboost", deployable=True)
        >>> artifact = ModelArtifact(path="/tmp/model", metadata=meta)
        >>> artifact.metadata.training_framework
        'xgboost'
    """

    path: str
    metadata: ModelMetadata = field(default_factory=ModelMetadata)


@dataclass
class AssembledModel:
    """A trained model transmitted between workflow tasks.

    Both artifacts must be fully packaged before passing to the pusher.
    Packaging is the assembler's responsibility. The pusher only uploads and
    registers pre-packaged artifacts.

    Attributes:
        raw_model: Raw model package (weights + sample data) intended for
            offline validation and reproducibility.
        deployable_model: Serving-ready bundle (e.g. Triton config + weights)
            intended for deployment to a model server.

    Example:
        >>> artifact = ModelArtifact(path="/tmp/model.ubj")
        >>> assembled = AssembledModel(
        ...     raw_model=artifact,
        ...     deployable_model=artifact,
        ... )
        >>> assembled.raw_model.path
        '/tmp/model.ubj'
    """

    raw_model: ModelArtifact
    deployable_model: ModelArtifact


@dataclass
class PusherResult:
    """The outcome of a single plugin execution.

    Attributes:
        name: Artifact name from ``PusherPluginConfig.name``.
        plugin: Plugin name that was invoked (e.g. ``"model_plugin"``).
        success: ``True`` if the plugin completed without error.
        value: Plugin-specific return data. Empty dict when ``success`` is
            ``False``.
        error: Human-readable error description when ``success`` is ``False``.
            ``None`` when ``success`` is ``True``.

    Example:
        >>> result = PusherResult(
        ...     name="model",
        ...     plugin="model_plugin",
        ...     success=True,
        ...     value={"model_name": "clf-v1", "version": "1"},
        ... )
        >>> result.success
        True
        >>> result.error is None
        True
    """

    name: str
    plugin: str
    success: bool
    value: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class DatasetArtifact:
    """A dataset artifact flowing between workflow tasks.

    Wraps the underlying tabular data produced by an assembler or trainer task
    and consumed by the pusher. The ``value`` may be a ``pandas.DataFrame``
    (always available), ``pyspark.sql.DataFrame`` (optional), or
    ``ray.data.Dataset`` (optional) — the type depends on the runtime
    environment.

    ``DatasetPusherPlugin`` extracts the value via ``to_pandas()`` before
    handing it to a ``DataSink`` implementation. Provider sinks (e.g.
    ``UberHiveSink``) may bypass ``to_pandas()`` and consume the native
    Spark DataFrame directly for efficiency.

    Attributes:
        value: The underlying dataset — a ``pandas.DataFrame``, a
            ``pyspark.sql.DataFrame``, or a ``ray.data.Dataset``.
        metadata: Optional free-form key-value metadata describing the dataset
            (schema version, feature names, etc.).

    Example:
        >>> import pandas as pd
        >>> artifact = DatasetArtifact.from_pandas(pd.DataFrame([{"x": 1}]))
        >>> isinstance(artifact.value, pd.DataFrame)
        True
    """

    value: Any  # pd.DataFrame | pyspark.sql.DataFrame | ray.data.Dataset
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> DatasetArtifact:
        """Create a ``DatasetArtifact`` wrapping a pandas DataFrame.

        Args:
            df: A ``pandas.DataFrame`` containing the dataset records.

        Returns:
            A ``DatasetArtifact`` whose ``value`` is the provided DataFrame.

        Raises:
            TypeError: If ``df`` is not a ``pandas.DataFrame``.

        Example:
            >>> import pandas as pd
            >>> artifact = DatasetArtifact.from_pandas(pd.DataFrame([{"x": 1}]))
            >>> len(artifact.value)
            1
        """
        import pandas as pd_rt

        if not isinstance(df, pd_rt.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(df).__name__}. "
                "Use DatasetArtifact(value=...) directly for Spark or Ray datasets."
            )
        return cls(value=df)

    def to_pandas(self) -> pd.DataFrame:
        """Return the dataset as a ``pandas.DataFrame``.

        Converts from Spark or Ray if necessary. Requires pandas to be installed.

        Returns:
            A ``pandas.DataFrame`` with one row per record.

        Raises:
            ImportError: If pandas is not installed.
            TypeError: If the underlying ``value`` type is not supported.

        Example:
            >>> import pandas as pd
            >>> artifact = DatasetArtifact.from_pandas(pd.DataFrame([{"x": 1}]))
            >>> artifact.to_pandas().shape
            (1, 1)
        """
        import pandas as pd_rt

        if isinstance(self.value, pd_rt.DataFrame):
            return self.value

        # Lazy Spark conversion
        try:
            import pyspark.sql as _ps

            if isinstance(self.value, _ps.DataFrame):
                return self.value.toPandas()
        except ModuleNotFoundError:
            pass

        # Lazy Ray conversion
        try:
            import ray.data as _rd

            if isinstance(self.value, _rd.Dataset):
                return self.value.to_pandas()
        except ModuleNotFoundError:
            pass

        raise TypeError(
            f"Cannot convert {type(self.value).__name__} to pandas.DataFrame. "
            "Supported: pandas.DataFrame, pyspark.sql.DataFrame, ray.data.Dataset."
        )
