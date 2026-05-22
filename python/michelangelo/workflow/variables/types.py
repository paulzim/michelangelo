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

    Wraps a tabular dataset produced by an assembler or trainer task and
    consumed by the pusher. Pass the value directly — type is detected
    automatically via ``backend``:

    - ``pandas.DataFrame`` — single-machine or small datasets.
    - ``pyspark.sql.DataFrame`` — large-scale distributed datasets (Spark).
    - ``ray.data.Dataset`` — Ray-based ML pipelines.

    Each ``DataSink`` extracts data in the format most efficient for its backend:

    - ``LocalFileSink`` calls ``artifact.to_pandas()`` — fine for pandas;
      triggers ``toPandas()`` collection for Spark (avoid on large datasets).
    - ``HiveSink`` accesses ``artifact.value`` directly as a Spark DataFrame —
      no ``toPandas()`` collection to driver; safe for billion-row datasets.

    Attributes:
        value: The underlying dataset — a ``pandas.DataFrame``, a
            ``pyspark.sql.DataFrame``, or a ``ray.data.Dataset``.
        metadata: Optional free-form key-value metadata describing the dataset
            (schema version, feature names, etc.).

    Example:
        >>> import pandas as pd
        >>> artifact = DatasetArtifact(value=pd.DataFrame([{"x": 1}]))
        >>> artifact.backend
        'pandas'
        >>> # spark_df = SparkSession.builder.getOrCreate().createDataFrame(...)
        >>> # DatasetArtifact(value=spark_df).backend
        'spark'
    """

    value: Any  # pd.DataFrame | pyspark.sql.DataFrame | ray.data.Dataset
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def backend(self) -> str:
        """Return the name of the underlying data backend.

        Detects the type of ``value`` at runtime via ``isinstance``, mirroring
        the internal ``DatasetVariable.save()`` dispatch pattern.

        Returns:
            ``"pandas"``, ``"spark"``, ``"ray"``, or ``"unknown"``.
            Checked in order: pandas → spark → ray. The first match wins.

        Example:
            >>> import pandas as pd
            >>> DatasetArtifact(value=pd.DataFrame()).backend
            'pandas'
        """
        import pandas as pd_rt

        if isinstance(self.value, pd_rt.DataFrame):
            return "pandas"
        try:
            import pyspark.sql as _ps

            if isinstance(self.value, _ps.DataFrame):
                return "spark"
        except ImportError:
            pass
        try:
            import ray.data as _rd

            if isinstance(self.value, _rd.Dataset):
                return "ray"
        except ImportError:
            pass
        return "unknown"

    def to_pandas(self) -> pd.DataFrame:
        """Return the dataset as a ``pandas.DataFrame``.

        For Spark DataFrames, calls ``toPandas()`` which collects all data to
        the driver — avoid on large datasets; use ``HiveSink`` instead.
        For Ray Datasets, calls ``to_pandas()``.

        Returns:
            A ``pandas.DataFrame`` with one row per record.

        Raises:
            ImportError: If pandas is not installed.
            TypeError: If the underlying ``value`` type is not supported.

        Example:
            >>> import pandas as pd
            >>> artifact = DatasetArtifact(value=pd.DataFrame([{"x": 1}]))
            >>> artifact.to_pandas().shape
            (1, 1)
        """
        import pandas as pd_rt

        if isinstance(self.value, pd_rt.DataFrame):
            return self.value

        # Spark conversion — collects to driver; use HiveSink for large datasets
        try:
            import pyspark.sql as _ps

            if isinstance(self.value, _ps.DataFrame):
                return self.value.toPandas()
        except ImportError:
            pass

        # Ray conversion
        try:
            import ray.data as _rd

            if isinstance(self.value, _rd.Dataset):
                return self.value.to_pandas()
        except ImportError:
            pass

        raise TypeError(
            f"Cannot convert {type(self.value).__name__} to pandas.DataFrame. "
            "Supported: pandas.DataFrame, pyspark.sql.DataFrame, ray.data.Dataset."
        )
