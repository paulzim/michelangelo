"""Tests for workflow/schema/data_sink.py — DataSink, LocalFileSink, InMemorySink."""

from __future__ import annotations

import os
import tempfile
from unittest import TestCase

import pandas as pd

from michelangelo.workflow.schema.data_sink import (
    DataSink,
    InMemorySink,
    LocalFileSink,
    SinkResult,
)
from michelangelo.workflow.schema.exceptions import ConfigurationError
from michelangelo.workflow.schema.pusher import DatasetFormat, DatasetPluginConfig
from michelangelo.workflow.variables.types import DatasetArtifact

_DF = pd.DataFrame([{"name": "alice", "score": 0.92}, {"name": "bob", "score": 0.88}])


def _artifact(df: pd.DataFrame | None = None) -> DatasetArtifact:
    return DatasetArtifact.from_pandas(df if df is not None else _DF.copy())


class TestDataSinkABC(TestCase):
    """Tests for the DataSink abstract base class."""

    def test_cannot_be_instantiated_directly(self):
        """It raises TypeError when instantiated without implementing write()."""
        with self.assertRaises(TypeError):
            DataSink()  # type: ignore[abstract]


class TestSinkResult(TestCase):
    """Tests for the SinkResult frozen dataclass."""

    def test_stores_uri_and_num_records(self):
        """It stores uri and num_records fields."""
        r = SinkResult(uri="/tmp/data.parquet", num_records=3)
        self.assertEqual(r.uri, "/tmp/data.parquet")
        self.assertEqual(r.num_records, 3)

    def test_is_frozen(self):
        """It raises FrozenInstanceError on attribute assignment."""
        r = SinkResult(uri="/tmp/x", num_records=1)
        with self.assertRaises(AttributeError):
            r.uri = "/tmp/other"  # type: ignore[misc]


class TestLocalFileSinkCSV(TestCase):
    """Tests for LocalFileSink CSV output."""

    def test_writes_csv_file(self):
        """It writes a valid CSV file with header and data rows."""
        dest = tempfile.mkdtemp()
        sink = LocalFileSink(dest, format=DatasetFormat.CSV)
        result = sink.write(_artifact())
        self.assertTrue(os.path.exists(result.uri))
        df_out = pd.read_csv(result.uri)
        self.assertEqual(len(df_out), len(_DF))
        self.assertIn("name", df_out.columns)

    def test_returns_correct_sink_result(self):
        """It returns a SinkResult with the correct uri and num_records."""
        dest = tempfile.mkdtemp()
        sink = LocalFileSink(dest, format=DatasetFormat.CSV)
        result = sink.write(_artifact())
        self.assertEqual(result.num_records, len(_DF))
        self.assertTrue(result.uri.startswith(dest))


class TestLocalFileSinkParquet(TestCase):
    """Tests for LocalFileSink Parquet output."""

    def test_writes_parquet_file(self):
        """It writes a valid Parquet file with correct shape."""
        dest = tempfile.mkdtemp()
        sink = LocalFileSink(dest, format=DatasetFormat.PARQUET)
        result = sink.write(_artifact())
        df_out = pd.read_parquet(result.uri)
        self.assertEqual(df_out.shape[0], len(_DF))

    def test_empty_dataframe_writes_zero_row_parquet(self):
        """It writes a valid zero-row Parquet file for an empty artifact."""
        dest = tempfile.mkdtemp()
        sink = LocalFileSink(dest, format=DatasetFormat.PARQUET)
        result = sink.write(_artifact(pd.DataFrame()))
        self.assertEqual(result.num_records, 0)
        df_out = pd.read_parquet(result.uri)
        self.assertEqual(len(df_out), 0)


class TestLocalFileSinkJSON(TestCase):
    """Tests for LocalFileSink JSON Lines output."""

    def test_writes_json_lines_file(self):
        """It writes a JSON Lines file with one object per row."""
        import json

        dest = tempfile.mkdtemp()
        sink = LocalFileSink(dest, format=DatasetFormat.JSON)
        result = sink.write(_artifact())
        with open(result.uri) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), len(_DF))
        for line in lines:
            obj = json.loads(line)
            self.assertIn("name", obj)


class TestLocalFileSinkDirectory(TestCase):
    """Tests for LocalFileSink directory creation."""

    def test_creates_destination_directory_if_absent(self):
        """It creates the destination directory automatically."""
        base = tempfile.mkdtemp()
        dest = os.path.join(base, "new_subdir", "nested")
        sink = LocalFileSink(dest, format=DatasetFormat.CSV)
        sink.write(_artifact())
        self.assertTrue(os.path.isdir(dest))


class TestInMemorySink(TestCase):
    """Tests for InMemorySink."""

    def test_write_stores_records(self):
        """It stores written records accessible via .records property."""
        sink = InMemorySink()
        sink.write(_artifact())
        self.assertEqual(len(sink.records), len(_DF))
        self.assertEqual(sink.records[0]["name"], "alice")

    def test_records_empty_before_write(self):
        """It returns an empty list before any write() call."""
        sink = InMemorySink()
        self.assertEqual(sink.records, [])

    def test_returns_memory_uri(self):
        """It returns a SinkResult with a memory:// URI."""
        sink = InMemorySink()
        result = sink.write(_artifact())
        self.assertTrue(result.uri.startswith("memory://"))
        self.assertEqual(result.num_records, len(_DF))


class TestDatasetPluginConfigPostInit(TestCase):
    """Tests for DatasetPluginConfig.__post_init__ LocalFileSink auto-creation."""

    def test_auto_creates_local_file_sink_from_destination_path(self):
        """It creates a LocalFileSink when destination_path is set."""
        cfg = DatasetPluginConfig(destination_path="/tmp/out")
        self.assertEqual(len(cfg.sinks), 1)
        self.assertIsInstance(cfg.sinks[0], LocalFileSink)

    def test_explicit_sinks_not_overridden(self):
        """It does not auto-create a sink when sinks is already populated."""
        sink = InMemorySink()
        cfg = DatasetPluginConfig(sinks=[sink])
        self.assertIs(cfg.sinks[0], sink)

    def test_no_sinks_and_no_destination_path_leaves_sinks_empty(self):
        """It leaves sinks empty when neither sinks nor destination_path is set."""
        cfg = DatasetPluginConfig()
        self.assertEqual(cfg.sinks, [])

    def test_explicit_empty_sinks_not_overridden_by_destination_path(self):
        """It does not auto-create a sink when sinks=[] is passed explicitly."""
        cfg = DatasetPluginConfig(sinks=[], destination_path="/tmp/out")
        self.assertEqual(cfg.sinks, [])

    def test_raises_config_error_when_plugin_has_no_sinks(self):
        """DatasetPusherPlugin raises ConfigurationError when sinks is empty."""
        from michelangelo.workflow.tasks.pusher.plugins.dataset_plugin import (
            DatasetPusherPlugin,
        )

        with self.assertRaises(ConfigurationError):
            DatasetPusherPlugin(
                config=DatasetPluginConfig(),
                artifact=_artifact(),
            )
