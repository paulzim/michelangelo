"""Tests for DatasetPusherPlugin."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from unittest import TestCase

from michelangelo.workflow.schema.exceptions import ConfigurationError
from michelangelo.workflow.schema.pusher import DatasetFormat, DatasetPluginConfig
from michelangelo.workflow.tasks.pusher.plugins.dataset_plugin import (
    DatasetPusherPlugin,
)

_RECORDS = [
    {"name": "alice", "score": 0.92},
    {"name": "bob", "score": 0.88},
    {"name": "carol", "score": 0.95},
]


def _make_plugin(
    records: list | None = None,
    fmt: DatasetFormat = DatasetFormat.CSV,
    dest: str | None = None,
) -> DatasetPusherPlugin:
    """Return a DatasetPusherPlugin with a temp destination and sensible defaults."""
    return DatasetPusherPlugin(
        config=DatasetPluginConfig(
            destination_path=dest or tempfile.mkdtemp(),
            format=fmt,
        ),
        artifact=records if records is not None else _RECORDS.copy(),
    )


class TestDatasetPusherPluginInit(TestCase):
    """Tests for DatasetPusherPlugin.__init__() validation."""

    def test_raises_when_destination_path_none(self):
        """It raises ConfigurationError when neither destination_path nor sinks is set."""
        with self.assertRaises(ConfigurationError) as ctx:
            DatasetPusherPlugin(
                config=DatasetPluginConfig(destination_path=None),
                artifact=[],
            )
        self.assertIn("destination", str(ctx.exception))


class TestDatasetPusherPluginExecute(TestCase):
    """Tests for DatasetPusherPlugin.execute()."""

    def test_writes_csv_file_with_header_and_rows(self):
        """It writes a CSV file with a header row and one data row per record."""
        dest = tempfile.mkdtemp()
        result = _make_plugin(fmt=DatasetFormat.CSV, dest=dest).execute()
        self.assertTrue(os.path.exists(result["destination_path"]))
        with open(result["destination_path"], newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), len(_RECORDS))
        self.assertEqual(rows[0]["name"], "alice")

    def test_writes_parquet_file_with_correct_shape(self):
        """It writes a Parquet file readable by pandas with the correct shape."""
        import pandas as pd

        dest = tempfile.mkdtemp()
        result = _make_plugin(fmt=DatasetFormat.PARQUET, dest=dest).execute()
        self.assertTrue(os.path.exists(result["destination_path"]))
        df = pd.read_parquet(result["destination_path"])
        self.assertEqual(df.shape[0], len(_RECORDS))
        self.assertIn("name", df.columns)
        self.assertIn("score", df.columns)

    def test_writes_json_lines_file(self):
        """It writes a JSON Lines file where each line is a valid JSON object."""
        dest = tempfile.mkdtemp()
        result = _make_plugin(fmt=DatasetFormat.JSON, dest=dest).execute()
        self.assertTrue(os.path.exists(result["destination_path"]))
        with open(result["destination_path"]) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), len(_RECORDS))
        for line in lines:
            obj = json.loads(line)
            self.assertIn("name", obj)

    def test_output_file_written_under_destination_path(self):
        """It writes the output file under the configured destination_path."""
        dest = tempfile.mkdtemp()
        result = _make_plugin(fmt=DatasetFormat.CSV, dest=dest).execute()
        self.assertTrue(result["destination_path"].startswith(dest))

    def test_returns_three_key_dict(self):
        """It returns a dict with exactly the three documented keys."""
        result = _make_plugin().execute()
        self.assertEqual(
            set(result.keys()), {"destination_path", "format", "num_records"}
        )

    def test_num_records_matches_artifact_length(self):
        """It sets num_records equal to the number of input records."""
        records = [{"x": i} for i in range(7)]
        result = _make_plugin(records=records).execute()
        self.assertEqual(result["num_records"], 7)

    def test_creates_destination_directory_if_absent(self):
        """It creates the destination directory automatically when it does not exist."""
        base = tempfile.mkdtemp()
        dest = os.path.join(base, "new_subdir", "nested")
        DatasetPusherPlugin(
            config=DatasetPluginConfig(destination_path=dest, format=DatasetFormat.CSV),
            artifact=_RECORDS.copy(),
        ).execute()
        self.assertTrue(os.path.isdir(dest))

    def test_empty_artifact_writes_valid_zero_row_parquet(self):
        """It writes a valid zero-row Parquet file when artifact is an empty list."""
        import pandas as pd

        dest = tempfile.mkdtemp()
        plugin = DatasetPusherPlugin(
            config=DatasetPluginConfig(
                destination_path=dest, format=DatasetFormat.PARQUET
            ),
            artifact=[],
        )
        result = plugin.execute()
        self.assertEqual(result["num_records"], 0)
        df = pd.read_parquet(result["destination_path"])
        self.assertEqual(len(df), 0)
