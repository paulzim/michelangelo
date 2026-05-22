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
    consumed by the pusher. Three backends are supported as first-class citizens:

    - **pandas** — single-machine or small datasets; always available.
    - **Spark** — large-scale distributed datasets; requires pyspark.
    - **Ray** — Ray-based ML pipelines; requires ray[data].

    Each ``DataSink`` extracts data in the format most efficient for its backend:

    - ``LocalFileSink`` calls ``artifact.to_pandas()`` — triggers ``toPandas()``
      collection for Spark; fine for pandas and small datasets.
    - ``HiveSink`` accesses ``artifact.value`` directly as a Spark DataFrame —
      no ``toPandas()`` collection to driver; safe for billion-row datasets.

    Attributes:
        value: The underlying dataset — a ``pandas.DataFrame``, a
            ``pyspark.sql.DataFrame``, or a ``ray.data.Dataset``.
        metadata: Optional free-form key-value metadata describing the dataset
            (schema version, feature names, etc.).

    Example:
        >>> import pandas as pd
        >>> artifact = DatasetArtifact.from_pandas(pd.DataFrame([{"x": 1}]))
        >>> artifact.backend
        'pandas'
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
                "Use from_spark() for Spark DataFrames or from_ray() for Ray Datasets."
            )
        return cls(value=df)

    @classmethod
    def from_spark(cls, df: Any) -> DatasetArtifact:
        """Create a ``DatasetArtifact`` wrapping a Spark DataFrame.

        The artifact holds the native Spark DataFrame. Sinks such as ``HiveSink``
        access ``artifact.value`` directly without calling ``toPandas()``, avoiding
        full data collection to the driver on large datasets.

        Args:
            df: A ``pyspark.sql.DataFrame``.

        Returns:
            A ``DatasetArtifact`` whose ``value`` is the provided Spark DataFrame.

        Raises:
            ImportError: If pyspark is not installed.
            TypeError: If ``df`` is not a ``pyspark.sql.DataFrame``.

        Example:
            >>> # spark = SparkSession.builder.getOrCreate()
            >>> # df = spark.createDataFrame([{"x": 1}])
            >>> # artifact = DatasetArtifact.from_spark(df)
            >>> # artifact.backend
            'spark'
        """
        try:
            import pyspark.sql as _ps
        except ImportError as e:
            raise ImportError(
                "pyspark is required: pip install pyspark"
            ) from e
        if not isinstance(df, _ps.DataFrame):
            raise TypeError(
                f"Expected pyspark.sql.DataFrame, got {type(df).__name__}. "
                "Use from_pandas() for pandas DataFrames."
            )
        return cls(value=df)

    @classmethod
    def from_ray(cls, ds: Any) -> DatasetArtifact:
        """Create a ``DatasetArtifact`` wrapping a Ray Dataset.

        Args:
            ds: A ``ray.data.Dataset``.

        Returns:
            A ``DatasetArtifact`` whose ``value`` is the provided Ray Dataset.

        Raises:
            ImportError: If ray[data] is not installed.
            TypeError: If ``ds`` is not a ``ray.data.Dataset``.

        Example:
            >>> # import ray
            >>> # artifact = DatasetArtifact.from_ray(ray.data.from_items([{"x": 1}]))
            >>> # artifact.backend
            'ray'
        """
        try:
            import ray.data as _rd
        except ImportError as e:
            raise ImportError(
                "ray[data] is required: pip install 'ray[data]'"
            ) from e
        if not isinstance(ds, _rd.Dataset):
            raise TypeError(
                f"Expected ray.data.Dataset, got {type(ds).__name__}. "
                "Use from_pandas() for pandas DataFrames."
            )
        return cls(value=ds)

    @property
    def backend(self) -> str:
        """Return the name of the underlying data backend.

        Returns:
            ``"pandas"``, ``"spark"``, ``"ray"``, or ``"unknown"``.

        Example:
            >>> import pandas as pd
            >>> DatasetArtifact.from_pandas(pd.DataFrame()).backend
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
            >>> artifact = DatasetArtifact.from_pandas(pd.DataFrame([{"x": 1}]))
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
