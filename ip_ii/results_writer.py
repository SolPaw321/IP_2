"""Stateful CSV diagnostics writer (original column order and formatting)."""

from __future__ import annotations

import csv
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from ip_ii.reporting import write_decision_rows, write_burst_rows, channel_summary

if TYPE_CHECKING:
    from config import ProjectConfig


class ResultsWriter:
    """Own three diagnostic CSV streams throughout one channel-processing run."""

    def __init__(self, staged: dict[Path, Path], config: ProjectConfig) -> None:
        self.staged = staged
        self.config = config
        self._files: ExitStack | None = None

    def __enter__(self) -> "ResultsWriter":
        stack = ExitStack()
        try:
            c = self.config
            self.parameters = csv.writer(stack.enter_context(
                self.staged[c.EWLS_PARAMETERS_FILE].open("w", newline="", encoding="utf-8")))
            self.impulses = csv.writer(stack.enter_context(
                self.staged[c.DETECTED_IMPULSES_FILE].open("w", newline="", encoding="utf-8")))
            self.bursts = csv.writer(stack.enter_context(
                self.staged[c.DETECTED_BURSTS_FILE].open("w", newline="", encoding="utf-8")))
            self.parameters.writerow(
                ["channel", "sample_index", "time_s"]
                + [f"a{i}" for i in range(1, c.AR_ORDER + 1)]
                + ["prediction", "error", "local_residual_variance", "local_residual_std",
                   "residual_history_count", "detection_threshold", "normalized_error",
                   "impulse_detected", "ewls_updated"]
            )
            self.impulses.writerow([
                "channel", "sample_index", "time_s", "observed", "prediction", "error",
                "local_residual_std", "detection_threshold", "normalized_error",
            ])
            self.bursts.writerow([
                "channel", "start_index", "end_index", "length", "start_time_s",
                "end_time_s", "within_max_length", "status", "skip_reason",
            ])
            self._files = stack
        except BaseException:
            stack.close()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._files is not None
        self._files.__exit__(exc_type, exc, tb)

    def observation(self, obs, *, channel_id: int, sample_rate: int) -> None:
        write_decision_rows(obs, channel_id=channel_id, sample_rate=sample_rate,
                            parameters_writer=self.parameters, impulses_writer=self.impulses)

    def burst_rows(self, result, *, channel_id: int, sample_rate: int) -> None:
        write_burst_rows(result, channel_id=channel_id, sample_rate=sample_rate,
                         writer=self.bursts)

    @staticmethod
    def summary(result, *, channel_id: int) -> dict:
        return channel_summary(result, channel_id=channel_id)
