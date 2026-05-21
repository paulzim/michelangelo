"""DatasetPusherPlugin — writes a DatasetArtifact to one or more configured sinks.

Consumes a ``DatasetArtifact`` (wrapping a pandas, Spark, or Ray dataset) and
writes it to a destination via the configured output path. In PR4 this is
extended by a pluggable ``DataSink`` abstraction that allows provider layers
(Uber) to target ``UberHiveSink``/``UberTerrablobSink``, and the community to
add ``S3Sink``, ``GCSSink``, ``BigQuerySink``, etc. without modifying this plugin.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from michelangelo.workflow.schema.exceptions import ConfigurationError
from michelangelo.workflow.schema.pusher import DatasetFormat, DatasetPluginConfig
from michelangelo.workflow.tasks.pusher.plugins.base import PusherPluginBase

if TYPE_CHECKING:
    from michelangelo.workflow.variables.types import DatasetArtifact

_logger = logging.getLogger(__name__)


class DatasetPusherPlugin(PusherPluginBase):
    """Plugin that writes a dataset artifact to a file sink in a configurable format.

    Consumes a ``DatasetArtifact`` (wrapping pandas, Spark, or Ray data) and
    converts it to a ``pandas.DataFrame`` via ``artifact.to_pandas()`` before
    writing. Supported output formats: CSV, Parquet, and JSON Lines.

    **Extension path (PR4):** ``DatasetPluginConfig`` will gain a
    ``sinks: list[DataSink]`` field. Provider layers supply their own
    ``DataSink`` implementations — ``UberHiveSink`` passes the native Spark
    DataFrame directly to Hive (bypassing ``to_pandas()``); ``S3Sink``,
    ``GCSSink``, etc. write to remote storage — all without subclassing this
    plugin. The ``destination_path`` shorthand auto-creates a ``LocalFileSink``
    for backwards compatibility.

    Args:
        config: ``DatasetPluginConfig`` specifying ``destination_path``,
            ``format``, and optional ``partition_by`` columns.
        artifact: A ``DatasetArtifact`` wrapping the dataset to write.
        storage_backend: Unused by this built-in implementation. Available for
            provider sink subclasses that compose with a StorageBackend.
        registry_client: Unused by this built-in implementation.

    Raises:
        ConfigurationError: If neither ``destination_path`` nor ``sinks`` is
            set (PR4 will use ``sinks``; PR3 requires ``destination_path``).

    Example::

        import pandas as pd
        from michelangelo.workflow.variables.types import DatasetArtifact

        artifact = DatasetArtifact.from_pandas(
            pd.DataFrame([{"col1": 1, "col2": "a"}])
        )
        plugin = DatasetPusherPlugin(
            config=DatasetPluginConfig(
                destination_path="/tmp/eval_data",
                format=DatasetFormat.PARQUET,
            ),
            artifact=artifact,
        )
        result = plugin.execute()
        print(result["destination_path"], result["num_records"])
    """

    def __init__(
        self,
        config: DatasetPluginConfig,
        artifact: DatasetArtifact | None = None,
        storage_backend: Any = None,
        registry_client: Any = None,
    ) -> None:
        """Validate that a write destination is configured, then store dependencies."""
        super().__init__(config, artifact, storage_backend, registry_client)
        # PR4 will add config.sinks — validate the combined state so that
        # introducing sinks does not require changing this guard.
        has_sinks = bool(getattr(config, "sinks", []))
        if not has_sinks and config.destination_path is None:
            raise ConfigurationError(
                "DatasetPusherPlugin requires a destination. "
                "Set DatasetPluginConfig(destination_path=...) or, in PR4+, "
                "pass an explicit sinks=[LocalFileSink(...)] list."
            )

    def execute(self) -> dict[str, Any]:
        """Convert the artifact to pandas and write to the configured destination.

        Calls ``artifact.to_pandas()`` to obtain a ``pandas.DataFrame``, then
        writes it to disk. The output file is named ``data.<format>`` inside
        ``destination_path``, which is created if absent.

        Returns:
            A dict with exactly three keys:

            - ``destination_path``: Absolute path to the written output file.
            - ``format``: The format value string (``"csv"``, ``"parquet"``,
              or ``"json"``).
            - ``num_records``: Number of rows written.

        Raises:
            ImportError: If pandas or pyarrow is not installed (Parquet only).
            IOError: If the destination path is not writable.
            TypeError: If the artifact value cannot be converted to pandas.
        """
        df = self._artifact.to_pandas()
        dest = self._config.destination_path
        fmt = self._config.format

        os.makedirs(dest, exist_ok=True)
        output_path = os.path.join(dest, f"data.{fmt.value}")

        if fmt == DatasetFormat.CSV:
            self._write_csv(df, output_path)
        elif fmt == DatasetFormat.PARQUET:
            self._write_parquet(df, output_path)
        elif fmt == DatasetFormat.JSON:
            self._write_json(df, output_path)
        else:
            raise ValueError(f"Unsupported DatasetFormat: {fmt!r}")

        num_records = len(df)
        _logger.info(
            "Wrote %d records to '%s' (%s).", num_records, output_path, fmt.value
        )
        return {
            "destination_path": output_path,
            "format": fmt.value,
            "num_records": num_records,
        }

    @staticmethod
    def _write_csv(df: Any, path: str) -> None:
        """Write a DataFrame as CSV with a header row.

        Args:
            df: A ``pandas.DataFrame``.
            path: Absolute path to write the CSV file.
        """
        df.to_csv(path, index=False)

    @staticmethod
    def _write_parquet(df: Any, path: str) -> None:
        """Write a DataFrame as Parquet.

        Args:
            df: A ``pandas.DataFrame``.
            path: Absolute path to write the Parquet file.

        Raises:
            ImportError: If pyarrow is not installed.
        """
        df.to_parquet(path, index=False)

    @staticmethod
    def _write_json(df: Any, path: str) -> None:
        """Write a DataFrame as JSON Lines (one JSON object per line).

        Args:
            df: A ``pandas.DataFrame``.
            path: Absolute path to write the JSON Lines file.
        """
        df.to_json(path, orient="records", lines=True)
