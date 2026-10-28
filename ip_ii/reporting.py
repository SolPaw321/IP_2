"""Shared diagnostic-file readers and provenance checks for plotting scripts.

Analysis never alters the audio. Every recorded sample remains untouched.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd
import numpy as np
from collections.abc import Iterable
from typing import Protocol

from ip_ii.declick import DeclickObservation, DeclickResult


class CsvRowWriter(Protocol):
    """Minimal protocol of the stdlib csv.writer used in reports."""

    def writerow(self, row: Iterable[object]) -> object: ...


PARAMETER_COLUMNS = (
    "channel", "sample_index", "time_s", "a1", "a2", "a3", "a4",
    "error", "detection_threshold",
)
IMPULSE_COLUMNS = ("channel", "sample_index", "time_s", "error")
BURST_COLUMNS = ("channel", "start_index", "end_index", "status")


def read_diagnostic_csv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    """Read CSV, validate required columns and numeric columns.

    Header-only event tables are valid: an empty table means no detections.
    """
    table = pd.read_csv(path)
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")
    for column in required:
        if column != "status":
            table[column] = pd.to_numeric(table[column], errors="raise")
    return table


def check_input_identity(input_path: Path, diagnostics: dict, cleaned_path: Path | None = None) -> None:
    """Reject stale runs when SHA-256 fingerprints are available.

    Results created by older versions (without hashes) remain readable.
    """
    expected = diagnostics.get("input_sha256")
    if expected and sha256(input_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Input WAV differs from the recording used to generate results. Run python run.py again.")
    if cleaned_path is not None:
        expected_clean = diagnostics.get("cleaned_sha256")
        if expected_clean and sha256(cleaned_path.read_bytes()).hexdigest() != expected_clean:
            raise ValueError("Cleaned WAV differs from the one recorded in diagnostics.json. Run python run.py again.")


def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def write_decision_rows(
    obs: DeclickObservation, *, channel_id: int, sample_rate: int,
    parameters_writer: CsvRowWriter, impulses_writer: CsvRowWriter,
) -> None:
    """Write unchanged parameter snapshots and detector flags to existing CSV schemas."""
    dec = obs.decision
    stats = obs.statistics
    if obs.save_parameters:
        parameters_writer.writerow(
            [channel_id, obs.index, f"{obs.index / sample_rate:.9f}"]
            + [_number(v) for v in obs.coefficients]
            + [_number(obs.prediction), _number(obs.error)]
            + (["", ""] if stats is None else
               [_number(stats.variance), _number(stats.standard_deviation)])
            + [obs.residual_history_count, _number(dec.threshold),
               _number(dec.normalized_error),
               "" if dec.detected is None else int(dec.detected),
               int(obs.updated)]
        )
    if dec.detected is True:
        impulses_writer.writerow([
            channel_id, obs.index, f"{obs.index / sample_rate:.9f}",
            _number(obs.observed), _number(obs.prediction), _number(obs.error),
            _number(stats.standard_deviation) if stats else "",
            _number(dec.threshold), _number(dec.normalized_error),
        ])



def write_burst_rows(
    result: DeclickResult, *, channel_id: int, sample_rate: int,
    writer: CsvRowWriter,
) -> None:
    """Report each candidate run and its reconstruction status."""
    successful = {
        (b.start_index, b.end_index)
        for b in result.interpolation.reconstructed_bursts
    }
    skipped = {
        (b.start_index, b.end_index): b.reason
        for b in result.interpolation.skipped_bursts
    }
    for burst in result.bursts:
        key = (burst.start_index, burst.end_index)
        reason = skipped.get(key)
        writer.writerow([
            channel_id, burst.start_index, burst.end_index, burst.length,
            f"{burst.start_index / sample_rate:.9f}",
            f"{burst.end_index / sample_rate:.9f}", int(burst.within_limit),
            "reconstructed" if key in successful else "skipped", reason or "",
        ])



def channel_summary(result: DeclickResult, *, channel_id: int) -> dict:
    """Create the unchanged per-channel JSON summary fields."""
    stats = result.final_residual_statistics
    return {
        "channel": channel_id,
        "ewls_updates": result.ewls_updates,
        "initialization_segment": (
            list(result.initialization_segment)
            if result.initialization_segment is not None else None
        ),
        "rechecked_anchor_count": result.rechecked_anchor_count,
        "residual_calibration_restarts": result.residual_restarts,
        "coefficients": result.coefficients.tolist(),
        "accepted_prediction_rmse_after_0_1s": (
            result.prediction_rmse if np.isfinite(result.prediction_rmse) else None
        ),
        "classified_samples": result.classified,
        "unclassified_samples": result.unclassified,
        "flagged_samples": result.detected,
        "candidate_bursts": len(result.bursts),
        "overlong_bursts": sum(not b.within_limit for b in result.bursts),
        "reconstructed_bursts": len(result.interpolation.reconstructed_bursts),
        "reconstructed_samples": result.interpolation.reconstructed_sample_count,
        "skipped_bursts": len(result.interpolation.skipped_bursts),
        "skipped_reasons": {
            reason: sum(b.reason == reason for b in result.interpolation.skipped_bursts)
            for reason in sorted({b.reason for b in result.interpolation.skipped_bursts})
        },
        "final_residual_std": stats.standard_deviation if stats else None,
    }
