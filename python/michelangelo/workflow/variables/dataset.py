"""DatasetArtifact — tabular dataset artifact for ML workflow tasks.

Mirrors the internal ``DatasetVariable`` design: wraps a storage path and a
transient in-memory value. Persistence is delegated to the uniflow IO registry
(``PandasIO``, ``SparkIO``, ``RayDatasetIO``), keeping this class free of any
direct dependency on pandas, pyspark, or ray at import time.

Open source ``_private/`` convention:
    Internal implementation details intended for subclassing by provider layers
    (Uber, cloud providers) belong in ``workflow/variables/_private/``. That
    directory is not imported by the public ``__init__.py`` and is excluded from
    the stable public API. ``DatasetArtifact`` itself *is* public and lives here
    in the top-level ``variables/`` package.
"""

from __future__ import annotations

from typing import Any


class DatasetArtifact:
    """A dataset artifact flowing between workflow tasks.

    Mirrors the internal ``DatasetVariable`` design: wraps a storage path and
    a transient in-memory value. The value is loaded lazily from storage on
    first access when not already present in memory.

    Three backends are supported as first-class citizens:

    - ``pandas.DataFrame`` — single-machine or small datasets.
    - ``pyspark.sql.DataFrame`` — large-scale distributed datasets (Spark).
    - ``ray.data.Dataset`` — Ray-based ML pipelines.

    Persistence is delegated to the IO registry:

    - ``PandasIO`` reads/writes Parquet via PyArrow (part-*.parquet directory).
    - ``SparkIO`` reads/writes Parquet via Spark.
    - ``RayDatasetIO`` reads/writes Parquet via Ray.

    Each ``DataSink`` operates on ``artifact.value`` in its native format:

    - ``LocalFileSink`` — accepts pandas only; raises ``TypeError`` for Spark/Ray.
    - ``HiveSink`` — accepts Spark only; accesses ``artifact.value`` natively.

    Example:
        >>> import pandas as pd
        >>> artifact = DatasetArtifact(value=pd.DataFrame([{"x": 1}]))
        >>> artifact.backend
        'pandas'
        >>> artifact.save()
        >>> artifact2 = DatasetArtifact(path=artifact.path)
        >>> artifact2.load_pandas_dataframe()
        >>> len(artifact2.value)
        1
    """

    def __init__(
        self,
        value: Any = None,
        path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialise with an optional in-memory value and/or storage path.

        Args:
            value: The in-memory dataset value. When provided, ``save()``
                persists it to ``path``. When ``None``, accessing ``value``
                triggers a load from ``path``.
            path: Storage path (local or fsspec URL). Auto-generated as a
                ``memory://`` UUID if not provided.
            metadata: Optional free-form key-value metadata.
        """
        import uuid

        self.path: str = path or f"memory://{uuid.uuid4().hex}"
        self.metadata: dict[str, Any] = metadata or {}
        self._value: Any = value
        self._saved: bool = False

    @classmethod
    def create(cls, value: Any, path: str | None = None) -> DatasetArtifact:
        """Create a ``DatasetArtifact`` holding an in-memory value.

        Equivalent to ``DatasetArtifact(value=value, path=path)``.

        Args:
            value: The dataset value — a pandas, Spark, or Ray object.
            path: Optional storage path. Auto-generated when ``None``.

        Returns:
            A new ``DatasetArtifact`` with ``value`` ready in memory.

        Example:
            >>> import pandas as pd
            >>> artifact = DatasetArtifact.create(pd.DataFrame([{"x": 1}]))
            >>> artifact.backend
            'pandas'
        """
        return cls(value=value, path=path)

    # ------------------------------------------------------------------
    # Value access
    # ------------------------------------------------------------------

    @property
    def value(self) -> Any:
        """Return the in-memory dataset, loading from storage if necessary.

        Triggers ``_load()`` on first access when the value is not already
        in memory (i.e. when constructed with only a ``path``).

        Returns:
            The underlying dataset — a pandas, Spark, or Ray object.
        """
        if self._value is None:
            self._load()
        return self._value

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

        if isinstance(self._value, pd_rt.DataFrame):
            return "pandas"
        try:
            import pyspark.sql as _ps

            if isinstance(self._value, _ps.DataFrame):
                return "spark"
        except ImportError:
            pass
        try:
            import ray.data as _rd

            if isinstance(self._value, _rd.Dataset):
                return "ray"
        except ImportError:
            pass
        return "unknown"

    # ------------------------------------------------------------------
    # Save — persist _value to self.path via the right IO
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the in-memory value to ``self.path`` using the right IO.

        Dispatches on the runtime type of ``value``:

        - ``pandas.DataFrame`` → ``PandasIO.write``
        - ``pyspark.sql.DataFrame`` → ``SparkIO.write``
        - ``ray.data.Dataset`` → ``RayDatasetIO.write``

        Raises:
            TypeError: If the value type is not supported.
        """
        import pandas as pd_rt

        if isinstance(self._value, pd_rt.DataFrame):
            self.save_pandas_dataframe()
            return
        try:
            import pyspark.sql as _ps

            if isinstance(self._value, _ps.DataFrame):
                self.save_spark_dataframe()
                return
        except ImportError:
            pass
        try:
            import ray.data as _rd

            if isinstance(self._value, _rd.Dataset):
                self.save_ray_dataset()
                return
        except ImportError:
            pass
        raise TypeError(
            f"Cannot save {type(self._value).__name__}. "
            "Supported: pandas.DataFrame, pyspark.sql.DataFrame, ray.data.Dataset."
        )

    def save_pandas_dataframe(self) -> None:
        """Persist the pandas DataFrame to ``self.path`` via ``PandasIO``."""
        from michelangelo.uniflow.plugins.pandas.io import PandasIO

        PandasIO().write(self.path, self._value)
        self._saved = True

    def save_spark_dataframe(self) -> None:
        """Persist the Spark DataFrame to ``self.path`` via ``SparkIO``."""
        from michelangelo.uniflow.plugins.spark.io import SparkIO

        SparkIO().write(self.path, self._value)
        self._saved = True

    def save_ray_dataset(self) -> None:
        """Persist the Ray Dataset to ``self.path`` via ``RayDatasetIO``."""
        from michelangelo.uniflow.plugins.ray.io import RayDatasetIO

        RayDatasetIO().write(self.path, self._value)
        self._saved = True

    # ------------------------------------------------------------------
    # Load — read _value from self.path via the right IO
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load ``_value`` from ``self.path``, detecting the active runtime.

        Runtime detection order (mirrors ``DatasetVariable._load()``):

        1. Active Spark session → ``load_spark_dataframe()``
        2. Ray initialized → ``load_ray_dataset()``
        3. Fallback → ``load_pandas_dataframe()``
        """
        try:
            import pyspark.sql as _ps

            spark = _ps.SparkSession.getActiveSession()
            if spark is not None:
                self.load_spark_dataframe()
                return
        except ImportError:
            pass
        try:
            import ray

            if ray.is_initialized():
                self.load_ray_dataset()
                return
        except ImportError:
            pass
        self.load_pandas_dataframe()

    def load_pandas_dataframe(self) -> None:
        """Load the dataset from ``self.path`` as a pandas DataFrame."""
        from michelangelo.uniflow.plugins.pandas.io import PandasIO

        self._value = PandasIO().read(self.path, None)
        self._saved = True

    def load_spark_dataframe(self) -> None:
        """Load the dataset from ``self.path`` as a Spark DataFrame."""
        from michelangelo.uniflow.plugins.spark.io import SparkIO

        self._value = SparkIO().read(self.path, None)
        self._saved = True

    def load_ray_dataset(self) -> None:
        """Load the dataset from ``self.path`` as a Ray Dataset."""
        from michelangelo.uniflow.plugins.ray.io import RayDatasetIO

        self._value = RayDatasetIO().read(self.path, None)
        self._saved = True
