"""DatasetPusherPlugin — writes a dataset to a local file sink."""

from __future__ import annotations

import logging
import os
from typing import Any

from michelangelo.workflow.schema.exceptions import ConfigurationError
from michelangelo.workflow.schema.pusher import DatasetFormat, DatasetPluginConfig
from michelangelo.workflow.tasks.pusher.plugins.base import PusherPluginBase

_logger = logging.getLogger(__name__)


class DatasetPusherPlugin(PusherPluginBase):
    """Plugin that writes a dataset to a file sink in a configurable format.

    Accepts a list of records (``list[dict[str, Any]]``) and writes them to
    the ``destination_path`` specified in the config. Supported formats: CSV,
    Parquet, and JSON Lines.

    ``destination_path`` must not be ``None`` — it is validated at construction
    time so that missing config is caught with an actionable error rather than
    a ``FileNotFoundError`` deep in the write path.

    Provider layers that need Spark or remote sinks should subclass this plugin
    and override ``execute()`` to call their own sink function.

    Args:
        config: ``DatasetPluginConfig`` specifying ``destination_path``,
            ``format``, and optional ``partition_by`` columns.
        artifact: A ``list[dict[str, Any]]`` representing the dataset records.
        storage_backend: Unused by this built-in implementation. Available for
            subclasses that add a remote upload step after writing.
        registry_client: Unused by this built-in implementation.

    Raises:
        ConfigurationError: If ``config.destination_path`` is ``None``.

    Example::

        plugin = DatasetPusherPlugin(
            config=DatasetPluginConfig(
                destination_path="/tmp/eval_data",
                format=DatasetFormat.PARQUET,
            ),
            artifact=[{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}],
        )
        result = plugin.execute()
        print(result["destination_path"], result["num_records"])
    """

    def __init__(
        self,
        config: DatasetPluginConfig,
        artifact: list[dict[str, Any]] | None = None,
        storage_backend: Any = None,
        registry_client: Any = None,
    ) -> None:
        """Validate destination_path then store dependencies."""
        super().__init__(config, artifact, storage_backend, registry_client)
        if config.destination_path is None:
            raise ConfigurationError(
                "DatasetPusherPlugin requires destination_path to be set. "
                "Pass a local directory path in DatasetPluginConfig.destination_path."
            )

    def execute(self) -> dict[str, Any]:
        """Write the dataset artifact to the configured destination path.

        Creates the destination directory if it does not exist. The output
        file is named ``data.<format>`` (e.g. ``data.parquet``).

        Returns:
            A dict with exactly three keys:

            - ``destination_path``: Absolute path to the written output file.
            - ``format``: The format value string (``"csv"``, ``"parquet"``,
              or ``"json"``).
            - ``num_records``: Number of records written.

        Raises:
            ImportError: If pandas or pyarrow is not installed (Parquet only).
            IOError: If the destination path is not writable.
        """
        dest = self._config.destination_path
        fmt = self._config.format
        records: list[dict[str, Any]] = self._artifact

        os.makedirs(dest, exist_ok=True)
        output_path = os.path.join(dest, f"data.{fmt.value}")

        if fmt == DatasetFormat.CSV:
            self._write_csv(records, output_path)
        elif fmt == DatasetFormat.PARQUET:
            self._write_parquet(records, output_path)
        elif fmt == DatasetFormat.JSON:
            self._write_json(records, output_path)

        _logger.info(
            "Wrote %d records to '%s' (%s).", len(records), output_path, fmt.value
        )
        return {
            "destination_path": output_path,
            "format": fmt.value,
            "num_records": len(records),
        }

    @staticmethod
    def _write_csv(records: list[dict[str, Any]], path: str) -> None:
        """Write records as CSV with a header row.

        Args:
            records: List of dicts. An empty list produces an empty file
                (no header — no column names to infer).
            path: Absolute path to write the CSV file.
        """
        import csv

        if not records:
            open(path, "w").close()
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def _write_parquet(records: list[dict[str, Any]], path: str) -> None:
        """Write records as Parquet using pandas and pyarrow.

        Args:
            records: List of dicts. An empty list produces a valid zero-row
                Parquet file.
            path: Absolute path to write the Parquet file.

        Raises:
            ImportError: If pandas or pyarrow is not installed.
        """
        import pandas as pd  # lazy — optional dep

        pd.DataFrame(records).to_parquet(path, index=False)

    @staticmethod
    def _write_json(records: list[dict[str, Any]], path: str) -> None:
        """Write records as JSON Lines (one JSON object per line).

        Args:
            records: List of dicts.
            path: Absolute path to write the JSON Lines file.
        """
        import json

        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
