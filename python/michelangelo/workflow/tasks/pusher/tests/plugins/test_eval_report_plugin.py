"""Tests for EvalReportPusherPlugin."""

from __future__ import annotations

import json
import os
from unittest import TestCase

from michelangelo.workflow.schema.pusher import EvalReportPluginConfig
from michelangelo.workflow.tasks.pusher.plugins.eval_report_plugin import (
    EvalReportPusherPlugin,
)

_ARTIFACT = {"accuracy": 0.93, "f1": 0.91, "num_samples": 1200}


def _make_plugin(
    config: EvalReportPluginConfig | None = None,
    artifact: dict | None = None,
) -> EvalReportPusherPlugin:
    """Return an EvalReportPusherPlugin with sensible defaults."""
    return EvalReportPusherPlugin(
        config=config or EvalReportPluginConfig(),
        artifact=artifact if artifact is not None else _ARTIFACT.copy(),
    )


class TestEvalReportPusherPluginExecute(TestCase):
    """Tests for EvalReportPusherPlugin.execute()."""

    def test_writes_valid_json_file_with_artifact_keys(self):
        """It writes a valid JSON file that contains the original artifact keys."""
        result = _make_plugin().execute()
        self.assertTrue(os.path.exists(result["output_path"]))
        with open(result["output_path"]) as f:
            doc = json.load(f)
        for key in _ARTIFACT:
            self.assertIn(key, doc)

    def test_uses_config_report_name(self):
        """It names the output file '<report_name>.json' when report_name is set."""
        result = EvalReportPusherPlugin(
            config=EvalReportPluginConfig(report_name="my-report"),
            artifact={"x": 1},
        ).execute()
        self.assertEqual(os.path.basename(result["output_path"]), "my-report.json")
        self.assertEqual(result["report_name"], "my-report")

    def test_generates_name_when_none(self):
        """It generates a report name starting with 'eval-report-' when None."""
        result = EvalReportPusherPlugin(
            config=EvalReportPluginConfig(report_name=None),
            artifact={"x": 1},
        ).execute()
        self.assertTrue(result["report_name"].startswith("eval-report-"))
        self.assertTrue(
            os.path.basename(result["output_path"]).startswith("eval-report-")
        )

    def test_merges_extra_fields_into_document(self):
        """It merges extra_fields from config into the written JSON document."""
        result = EvalReportPusherPlugin(
            config=EvalReportPluginConfig(extra_fields={"run_id": "abc123"}),
            artifact={"accuracy": 0.9},
        ).execute()
        with open(result["output_path"]) as f:
            doc = json.load(f)
        self.assertEqual(doc["run_id"], "abc123")
        self.assertEqual(doc["accuracy"], 0.9)

    def test_returns_three_key_dict(self):
        """It returns a dict with exactly the three documented keys."""
        result = _make_plugin().execute()
        self.assertEqual(set(result.keys()), {"report_name", "output_path", "num_keys"})

    def test_num_keys_counts_artifact_keys_only(self):
        """It counts only the original artifact's top-level keys in num_keys."""
        artifact = {"a": 1, "b": 2}
        result = EvalReportPusherPlugin(
            config=EvalReportPluginConfig(
                extra_fields={"extra_1": "x", "extra_2": "y"}
            ),
            artifact=artifact,
        ).execute()
        self.assertEqual(result["num_keys"], 2)

    def test_report_name_key_written_to_document(self):
        """It adds a '_report_name' key to the written JSON document."""
        result = EvalReportPusherPlugin(
            config=EvalReportPluginConfig(report_name="audit-run"),
            artifact={"loss": 0.12},
        ).execute()
        with open(result["output_path"]) as f:
            doc = json.load(f)
        self.assertEqual(doc["_report_name"], "audit-run")

    def test_output_written_to_temp_dir_with_prefix(self):
        """It creates the output file inside a michelangelo_reports_ temp dir."""
        result = _make_plugin().execute()
        parent = os.path.dirname(result["output_path"])
        self.assertIn("michelangelo_reports_", os.path.basename(parent))
