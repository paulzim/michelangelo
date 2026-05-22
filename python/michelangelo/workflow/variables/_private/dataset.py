"""DatasetVariable — provider-layer subclass of DatasetArtifact.

Extends ``DatasetArtifact`` with the ``Variable`` base class infrastructure
(``_load_value_using_io`` / ``_save_value_using_io`` helpers, ``path`` from
environment, ``_io_metadata`` round-trip). Provider layers (e.g. Uber) subclass
``DatasetVariable`` here to add platform-specific IO or metadata handling.

Open source users: use ``DatasetArtifact`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from michelangelo.uniflow.plugins.pandas.io import PandasIO
from michelangelo.uniflow.plugins.ray.io import RayDatasetIO
from michelangelo.uniflow.plugins.spark.io import SparkIO
from michelangelo.workflow.variables._private.base import Variable
from michelangelo.workflow.variables.dataset import DatasetArtifact


@dataclass
class DatasetVariable(DatasetArtifact, Variable):
    """Dataset variable with Variable base infrastructure.

    Inherits the full IO dispatch from ``DatasetArtifact`` and the
    ``_load_value_using_io`` / ``_save_value_using_io`` helpers from
    ``Variable``. Provider subclasses override ``save()`` or ``_load()``
    to add platform-specific behaviour.
    """

    def save_pandas_dataframe(self) -> None:
        """Persist the pandas DataFrame via Variable IO infrastructure."""
        self._save_value_using_io(PandasIO)

    def save_spark_dataframe(self) -> None:
        """Persist the Spark DataFrame via Variable IO infrastructure."""
        self._save_value_using_io(SparkIO)

    def save_ray_dataset(self) -> None:
        """Persist the Ray Dataset via Variable IO infrastructure."""
        self._save_value_using_io(RayDatasetIO)

    def load_pandas_dataframe(self) -> None:
        """Load from path as a pandas DataFrame via Variable IO infrastructure."""
        self._load_value_using_io(PandasIO)

    def load_spark_dataframe(self) -> None:
        """Load from path as a Spark DataFrame via Variable IO infrastructure."""
        self._load_value_using_io(SparkIO)

    def load_ray_dataset(self) -> None:
        """Load from path as a Ray Dataset via Variable IO infrastructure."""
        self._load_value_using_io(RayDatasetIO)
