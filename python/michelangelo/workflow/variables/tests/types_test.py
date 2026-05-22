"""Tests for workflow variable types: ModelArtifact, AssembledModel, PusherResult."""

from __future__ import annotations

import sys
import types as _types
from dataclasses import dataclass
from io import BytesIO
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from michelangelo.workflow.variables.metadata import ModelMetadata
from michelangelo.workflow.variables.types import (
    AssembledModel,
    DatasetArtifact,
    ModelArtifact,
    PusherResult,
)


class TestModelMetadata(TestCase):
    """Tests for ModelMetadata."""

    def test_defaults_all_none_or_false(self):
        """It initialises with all fields at their defaults."""
        meta = ModelMetadata()
        self.assertIsNone(meta.training_framework)
        self.assertIsNone(meta.model_class)
        self.assertFalse(meta.assembled)
        self.assertFalse(meta.deployable)
        self.assertIsNone(meta._schema)
        self.assertIsNone(meta._sample_data)
        self.assertIsNone(meta._hyperparameters)

    def test_stores_training_framework(self):
        """It stores the provided training_framework."""
        meta = ModelMetadata(training_framework="xgboost")
        self.assertEqual(meta.training_framework, "xgboost")

    def test_stores_model_class(self):
        """It stores the provided model_class import path."""
        meta = ModelMetadata(model_class="mypackage.models.Clf")
        self.assertEqual(meta.model_class, "mypackage.models.Clf")

    def test_stores_assembled_and_deployable_flags(self):
        """It stores assembled and deployable boolean flags."""
        meta = ModelMetadata(assembled=True, deployable=True)
        self.assertTrue(meta.assembled)
        self.assertTrue(meta.deployable)

    def test_stores_binary_payloads(self):
        """It stores schema, sample_data, and hyperparameters as BytesIO."""
        schema = BytesIO(b"schema-bytes")
        sample = BytesIO(b"sample-bytes")
        hparams = BytesIO(b"hparam-bytes")
        meta = ModelMetadata(
            _schema=schema,
            _sample_data=sample,
            _hyperparameters=hparams,
        )
        self.assertEqual(meta._schema.read(), b"schema-bytes")
        self.assertEqual(meta._sample_data.read(), b"sample-bytes")
        self.assertEqual(meta._hyperparameters.read(), b"hparam-bytes")

    def test_is_subclassable(self):
        """A subclass can add provider-specific fields."""

        @dataclass
        class UberModelMetadata(ModelMetadata):
            training_job_id: str | None = None

        uber_meta = UberModelMetadata(
            training_framework="pytorch",
            training_job_id="job-1",
        )
        self.assertEqual(uber_meta.training_framework, "pytorch")
        self.assertEqual(uber_meta.training_job_id, "job-1")
        self.assertIsInstance(uber_meta, ModelMetadata)


class TestModelArtifact(TestCase):
    """Tests for ModelArtifact."""

    def test_stores_path(self):
        """It stores the provided path."""
        artifact = ModelArtifact(path="/tmp/model")
        self.assertEqual(artifact.path, "/tmp/model")

    def test_metadata_defaults_to_empty_model_metadata(self):
        """It defaults metadata to a ModelMetadata instance with defaults."""
        artifact = ModelArtifact(path="/tmp/model")
        self.assertIsInstance(artifact.metadata, ModelMetadata)
        self.assertIsNone(artifact.metadata.training_framework)

    def test_metadata_instances_are_independent(self):
        """It creates a separate ModelMetadata instance for each artifact."""
        a = ModelArtifact(path="/tmp/a")
        b = ModelArtifact(path="/tmp/b")
        a.metadata.training_framework = "pytorch"
        self.assertIsNone(b.metadata.training_framework)

    def test_metadata_can_be_provided(self):
        """It stores an explicitly provided ModelMetadata."""
        meta = ModelMetadata(training_framework="xgboost", deployable=True)
        artifact = ModelArtifact(path="/tmp/m", metadata=meta)
        self.assertEqual(artifact.metadata.training_framework, "xgboost")
        self.assertTrue(artifact.metadata.deployable)

    def test_metadata_accepts_subclass(self):
        """It stores a ModelMetadata subclass without modification."""

        @dataclass
        class UberModelMetadata(ModelMetadata):
            training_job_id: str | None = None

        uber_meta = UberModelMetadata(
            training_framework="huggingface",
            training_job_id="j-42",
        )
        artifact = ModelArtifact(path="/tmp/m", metadata=uber_meta)
        self.assertIsInstance(artifact.metadata, ModelMetadata)
        # type: ignore[attr-defined]
        self.assertEqual(artifact.metadata.training_job_id, "j-42")  # type: ignore[attr-defined]


class TestAssembledModel(TestCase):
    """Tests for AssembledModel."""

    def _make_artifact(self, path: str = "/tmp/model") -> ModelArtifact:
        return ModelArtifact(path=path)

    def test_stores_raw_and_deployable_models(self):
        """It stores both raw_model and deployable_model."""
        raw = self._make_artifact("/tmp/raw")
        deployable = self._make_artifact("/tmp/deployable")
        model = AssembledModel(raw_model=raw, deployable_model=deployable)
        self.assertEqual(model.raw_model.path, "/tmp/raw")
        self.assertEqual(model.deployable_model.path, "/tmp/deployable")


class TestPusherResult(TestCase):
    """Tests for PusherResult."""

    def test_successful_result_fields(self):
        """It stores name, plugin, success, and value for a successful result."""
        result = PusherResult(
            name="model",
            plugin="model_plugin",
            success=True,
            value={"model_name": "clf", "version": "1"},
        )
        self.assertEqual(result.name, "model")
        self.assertEqual(result.plugin, "model_plugin")
        self.assertTrue(result.success)
        self.assertEqual(result.value["version"], "1")

    def test_failed_result_fields(self):
        """It stores error message and empty value for a failed result."""
        result = PusherResult(
            name="model",
            plugin="model_plugin",
            success=False,
            value={},
            error="Upload failed.",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Upload failed.")

    def test_value_defaults_to_empty_dict(self):
        """It defaults value to an empty dict."""
        result = PusherResult(name="r", plugin="p", success=True)
        self.assertEqual(result.value, {})

    def test_value_instances_are_independent(self):
        """It creates a separate value dict for each instance."""
        a = PusherResult(name="a", plugin="p", success=True)
        b = PusherResult(name="b", plugin="p", success=True)
        a.value["key"] = "v"
        self.assertEqual(b.value, {})

    def test_error_defaults_to_none(self):
        """It defaults error to None."""
        result = PusherResult(name="r", plugin="p", success=True)
        self.assertIsNone(result.error)


# ---------------------------------------------------------------------------
# Helpers for mocking optional backends (pyspark / ray not installed)
# ---------------------------------------------------------------------------

def _mock_pyspark(df_instance=None):
    """Return (mock_pyspark_sql_module, mock_spark_df) for sys.modules patching."""
    mock_df_class = type("DataFrame", (), {})
    mock_sql = _types.SimpleNamespace(DataFrame=mock_df_class)
    spark_df = df_instance if df_instance is not None else mock_df_class()
    return mock_sql, spark_df


def _mock_ray(ds_instance=None):
    """Return (mock_ray_data_module, mock_ray_dataset) for sys.modules patching."""
    mock_dataset_class = type("Dataset", (), {})
    mock_data = _types.SimpleNamespace(Dataset=mock_dataset_class)
    ray_ds = ds_instance if ds_instance is not None else mock_dataset_class()
    return mock_data, ray_ds


def _spark_mods(mock_sql):
    """sys.modules patch dict for pyspark."""
    return {"pyspark": _types.SimpleNamespace(sql=mock_sql), "pyspark.sql": mock_sql}


def _ray_mods(mock_data):
    """sys.modules patch dict for ray.data."""
    return {"ray": _types.SimpleNamespace(data=mock_data), "ray.data": mock_data}


class TestDatasetArtifactFromPandas(TestCase):
    """Tests for DatasetArtifact.from_pandas()."""

    def test_wraps_dataframe(self):
        """It stores the DataFrame in .value."""
        df = pd.DataFrame([{"x": 1}])
        artifact = DatasetArtifact.from_pandas(df)
        self.assertIs(artifact.value, df)

    def test_raises_type_error_for_non_dataframe(self):
        """It raises TypeError when passed a non-DataFrame."""
        with self.assertRaises(TypeError):
            DatasetArtifact.from_pandas([{"x": 1}])  # type: ignore[arg-type]

    def test_backend_is_pandas(self):
        """It reports 'pandas' as the backend."""
        artifact = DatasetArtifact.from_pandas(pd.DataFrame())
        self.assertEqual(artifact.backend, "pandas")


class TestDatasetArtifactFromSpark(TestCase):
    """Tests for DatasetArtifact.from_spark()."""

    def test_wraps_spark_dataframe(self):
        """It stores the Spark DataFrame in .value."""
        mock_sql, spark_df = _mock_pyspark()
        mods = _spark_mods(mock_sql)
        with patch.dict(sys.modules, mods):
            artifact = DatasetArtifact.from_spark(spark_df)
        self.assertIs(artifact.value, spark_df)

    def test_raises_import_error_when_pyspark_missing(self):
        """It raises ImportError when pyspark is not installed."""
        with patch.dict(sys.modules, {"pyspark": None, "pyspark.sql": None}), \
                self.assertRaises(ImportError):
            DatasetArtifact.from_spark(MagicMock())

    def test_raises_type_error_for_wrong_type(self):
        """It raises TypeError when value is not a Spark DataFrame."""
        mock_sql, _ = _mock_pyspark()
        mods = _spark_mods(mock_sql)
        with patch.dict(sys.modules, mods), self.assertRaises(TypeError):
            DatasetArtifact.from_spark("not-a-spark-df")

    def test_backend_is_spark(self):
        """It reports 'spark' as the backend."""
        mock_sql, spark_df = _mock_pyspark()
        with patch.dict(sys.modules, _spark_mods(mock_sql)):
            artifact = DatasetArtifact.from_spark(spark_df)
            self.assertEqual(artifact.backend, "spark")


class TestDatasetArtifactFromRay(TestCase):
    """Tests for DatasetArtifact.from_ray()."""

    def test_wraps_ray_dataset(self):
        """It stores the Ray Dataset in .value."""
        mock_data, ray_ds = _mock_ray()
        with patch.dict(sys.modules, _ray_mods(mock_data)):
            artifact = DatasetArtifact.from_ray(ray_ds)
        self.assertIs(artifact.value, ray_ds)

    def test_raises_import_error_when_ray_missing(self):
        """It raises ImportError when ray is not installed."""
        with patch.dict(sys.modules, {"ray": None, "ray.data": None}), \
                self.assertRaises(ImportError):
            DatasetArtifact.from_ray(MagicMock())

    def test_raises_type_error_for_wrong_type(self):
        """It raises TypeError when value is not a Ray Dataset."""
        mock_data, _ = _mock_ray()
        with patch.dict(sys.modules, _ray_mods(mock_data)), \
                self.assertRaises(TypeError):
            DatasetArtifact.from_ray("not-a-ray-dataset")

    def test_backend_is_ray(self):
        """It reports 'ray' as the backend."""
        mock_data, ray_ds = _mock_ray()
        with patch.dict(sys.modules, _ray_mods(mock_data)):
            artifact = DatasetArtifact.from_ray(ray_ds)
            self.assertEqual(artifact.backend, "ray")


class TestDatasetArtifactToPandas(TestCase):
    """Tests for DatasetArtifact.to_pandas() cross-backend conversion."""

    def test_returns_pandas_directly(self):
        """It returns the stored DataFrame for a pandas artifact."""
        df = pd.DataFrame([{"x": 1}])
        artifact = DatasetArtifact.from_pandas(df)
        self.assertIs(artifact.to_pandas(), df)

    def test_calls_to_pandas_on_spark_df(self):
        """It calls toPandas() on a Spark DataFrame."""
        expected = pd.DataFrame([{"x": 1}])
        mock_sql, spark_df = _mock_pyspark()
        spark_df.toPandas = MagicMock(return_value=expected)
        with patch.dict(sys.modules, _spark_mods(mock_sql)):
            artifact = DatasetArtifact.from_spark(spark_df)
            result = artifact.to_pandas()
        self.assertIs(result, expected)
        spark_df.toPandas.assert_called_once()

    def test_calls_to_pandas_on_ray_dataset(self):
        """It calls to_pandas() on a Ray Dataset."""
        expected = pd.DataFrame([{"x": 2}])
        mock_data, ray_ds = _mock_ray()
        ray_ds.to_pandas = MagicMock(return_value=expected)
        with patch.dict(sys.modules, _ray_mods(mock_data)):
            artifact = DatasetArtifact.from_ray(ray_ds)
            result = artifact.to_pandas()
        self.assertIs(result, expected)
        ray_ds.to_pandas.assert_called_once()

    def test_raises_type_error_for_unsupported_value(self):
        """It raises TypeError for an unrecognised value type."""
        artifact = DatasetArtifact(value={"not": "a dataframe"})
        with self.assertRaises(TypeError):
            artifact.to_pandas()
