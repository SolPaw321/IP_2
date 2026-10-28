"""Descriptive comparison of WAV reconstructions (project 1, step 12).

These measurements describe the *amount* of editing and the detector's
behavior. They DO NOT measure audio quality or signal-to-noise improvement:
no clean reference recording is supplied in the assignment.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ip_ii.declick import DeclickResult


def channel_metrics(original: np.ndarray, cleaned: np.ndarray, result: DeclickResult) -> dict:
    """Summarize one channel using the actual, quantized output PCM samples.

    Both inputs must be one-dimensional, normalized float arrays read from
    PCM16 WAV files. Counts of *changed* samples reflect PCM16 quantization;
    counts of *reconstructed* samples reflect the interpolation decisions.
    """
    source = np.asarray(original, dtype=np.float64)
    output = np.asarray(cleaned, dtype=np.float64)
    if source.ndim != 1 or output.shape != source.shape:
        raise ValueError("Original and cleaned channels must have identical 1D shapes.")
    if source.size == 0 or not np.all(np.isfinite(source)) or not np.all(np.isfinite(output)):
        raise ValueError("Audio channels must be nonempty and finite.")
    differences = output - source
    altered = output != source
    changed = int(np.count_nonzero(altered))
    permitted = np.zeros(source.shape[0], dtype=np.bool_)
    for burst in result.interpolation.reconstructed_bursts:
        permitted[burst.start_index:burst.end_index + 1] = True
    if np.any(altered & ~permitted):
        raise RuntimeError("Output changed a sample outside reconstructed bursts.")
    return {
        "frames": int(source.size),
        "flagged_samples": int(result.detected),
        "classified_samples": int(result.classified),
        "unclassified_samples": int(result.unclassified),
        "candidate_bursts": len(result.bursts),
        "reconstructed_bursts": len(result.interpolation.reconstructed_bursts),
        "reconstructed_samples": result.interpolation.reconstructed_sample_count,
        "changed_pcm_samples": changed,
        "skipped_bursts": len(result.interpolation.skipped_bursts),
        "overlong_bursts": sum(not b.within_limit for b in result.bursts),
        "residual_restarts": result.residual_restarts,
        "ewls_updates": result.ewls_updates,
        "accepted_prediction_rmse_after_0_1s": (
            result.prediction_rmse if math.isfinite(result.prediction_rmse) else None
        ),
        "difference_rms": float(np.sqrt(np.mean(np.square(differences)))),
        "difference_peak": float(np.max(np.abs(differences))),
        "changed_percent": 100.0 * changed / source.size,
    }


def summarize_channels(channel_rows: list[dict]) -> dict:
    """Aggregate additive counts; report RMS of *edits*, not quality."""
    if not channel_rows:
        raise ValueError("At least one channel is required.")
    rows = channel_rows
    count_fields = (
        "flagged_samples", "classified_samples", "unclassified_samples",
        "candidate_bursts", "reconstructed_bursts", "reconstructed_samples",
        "changed_pcm_samples", "skipped_bursts", "overlong_bursts",
        "residual_restarts", "ewls_updates",
    )
    summary = {key: sum(row[key] for row in rows) for key in count_fields}
    total_samples = sum(row["frames"] for row in rows)
    summary["changed_percent"] = 100.0 * summary["changed_pcm_samples"] / total_samples
    summary["difference_rms"] = math.sqrt(sum(
        row["difference_rms"] ** 2 * row["frames"] for row in rows
    ) / total_samples)
    summary["difference_peak"] = max(row["difference_peak"] for row in rows)
    rmses = [row["accepted_prediction_rmse_after_0_1s"] for row in rows]
    valid = [value for value in rmses if value is not None]
    # A simple mean of channel RMSEs; NEVER interpreted as a quality score.
    summary["mean_channel_prediction_rmse"] = sum(valid) / len(valid) if valid else None
    return summary


def plot_experiment_comparison(summary_rows: list[dict], destination: Path) -> None:
    """Create descriptive comparison charts for a set of experiments."""
    if not summary_rows:
        return
    names = [row["case_id"] for row in summary_rows]
    labels = np.arange(len(names), dtype=float)
    fig, (ax_counts, ax_diff) = plt.subplots(2, 1, figsize=(max(9, len(names) * 1.6), 8))
    width = 0.25
    for offset, key, caption in (
        (-width, "flagged_samples", "Detected"),
        (0, "reconstructed_samples", "Reconstructed"),
        (width, "changed_pcm_samples", "Changed in PCM16"),
    ):
        ax_counts.bar(labels + offset, [row[key] for row in summary_rows], width, label=caption)
    ax_counts.set_ylabel("Sample count, all channels")
    ax_counts.set_title("Candidate detections and actual editing (not an audio-quality score)")
    ax_counts.legend(loc="best")
    ax_counts.grid(axis="y", alpha=0.2)
    ax_diff.bar(labels, [row["difference_rms"] for row in summary_rows])
    ax_diff.set_ylabel("RMS(cleaned - original)")
    ax_diff.set_title("Strength of editing, NOT improvement vs pristine audio")
    ax_diff.grid(axis="y", alpha=0.2)
    for ax in (ax_counts, ax_diff):
        ax.set_xticks(labels, names, rotation=25, ha="right")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
