"""EvalReportPusherPlugin — persists a structured evaluation report as JSON."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from typing import Any

from michelangelo.workflow.tasks.pusher.plugins.base import PusherPluginBase

_logger = logging.getLogger(__name__)


class EvalReportPusherPlugin(PusherPluginBase):
    """Plugin that persists a structured evaluation report as a JSON file.

    Accepts a plain ``dict`` as the artifact and writes it to a temporary
    directory as ``<report_name>.json``. Provider layers can subclass this
    and override ``execute()`` to post the report to a database or API
    instead of writing to disk.

    Args:
        config: ``EvalReportPluginConfig`` with optional ``report_name`` and
            ``extra_fields``.
        artifact: A ``dict`` representing the evaluation report document.
            Top-level keys are counted for ``num_keys`` in the return value.
        storage_backend: Unused by this built-in implementation.
        registry_client: Unused by this built-in implementation.

    Example::

        plugin = EvalReportPusherPlugin(
            config=EvalReportPluginConfig(report_name="run-2026"),
            artifact={"accuracy": 0.93, "f1": 0.91, "num_samples": 1200},
        )
        result = plugin.execute()
        print(result["report_name"], result["output_path"])
    """

    def execute(self) -> dict[str, Any]:
        """Write the evaluation report to a JSON file in a temp directory.

        Merges ``extra_fields`` from config into the document and adds an
        internal ``_report_name`` key. ``num_keys`` is counted from the
        **original artifact** before merging — it does not count
        ``extra_fields`` keys or the ``_report_name`` key added internally.

        The temp directory is created with
        ``tempfile.mkdtemp(prefix="michelangelo_reports_")`` so it is safe
        for concurrent runs. The caller owns cleanup of the temp directory.

        Returns:
            A dict with exactly three keys:

            - ``report_name``: Name assigned to this report.
            - ``output_path``: Absolute path to the written JSON file.
            - ``num_keys``: Number of top-level keys in the original artifact
              dict (not counting ``extra_fields`` or ``_report_name``).

        Raises:
            IOError: If the temp directory cannot be created or the file
                cannot be written.
        """
        report_name = self._config.report_name or self._generate_name()
        num_keys = len(self._artifact)  # counted BEFORE merge

        document = {
            **self._artifact,
            **self._config.extra_fields,
            "_report_name": report_name,
        }

        output_dir = tempfile.mkdtemp(prefix="michelangelo_reports_")
        output_path = os.path.join(output_dir, f"{report_name}.json")
        with open(output_path, "w") as f:
            json.dump(document, f, indent=2)

        _logger.info("Wrote evaluation report '%s' to '%s'.", report_name, output_path)
        return {
            "report_name": report_name,
            "output_path": output_path,
            "num_keys": num_keys,
        }

    @staticmethod
    def _generate_name() -> str:
        """Generate a unique report name with an 'eval-report-' prefix."""
        return f"eval-report-{uuid.uuid4().hex[:8]}"
