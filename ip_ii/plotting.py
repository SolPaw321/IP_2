"""Project 1 waveform and model-diagnostics plotting utilities.

Step 3: original-waveform inspection and amplitude-jump exploration.
Step 11: original/cleaned comparisons, detector candidate marks,
reconstructed/skipped runs, EW-LS parameter and residual plots.
Plotting never edits audio or changes saved diagnostic data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _channel_matrix(samples: np.ndarray) -> np.ndarray:
    """Return audio with shape (frames, channels), including mono recordings."""
    values = np.asarray(samples)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("Audio must have at least two frames and one channel.")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("Plots expect floating-point, normalized audio samples.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Audio samples cannot contain NaN or infinity.")
    return values


def _validate_plot_args(sample_rate: int, max_bins: int | None = None) -> None:
    if sample_rate <= 0:
        raise ValueError("Sampling rate must be positive.")
    if max_bins is not None and max_bins < 1:
        raise ValueError("Maximum number of overview bins must be positive.")


def _make_axes(channel_count: int, *, title: str) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        channel_count,
        1,
        figsize=(13, 2.8 * channel_count + 1.2),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title)
    for channel, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(f"Channel {channel + 1}\nAmplitude")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Time [s]")
    return fig, axes[:, 0]


def _save_figure(fig: plt.Figure, path: str | Path, *, keep_open: bool) -> Path:
    """Save a figure; optionally leave it open for interactive inspection."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=130)
    if not keep_open:
        plt.close(fig)
    return destination


def plot_waveform_overview(
    samples: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
    *,
    max_bins: int = 3500,
    keep_open: bool = False,
) -> Path:
    """Save a complete waveform overview without hiding short spikes.

    A naive plot of every N-th sample can skip a one-sample click. Here
    each time bin contributes BOTH its minimum and maximum sample.
    The resulting vertical segments preserve the observed amplitude
    extrema of the raw recording even when there are many samples.
    """
    data = _channel_matrix(samples)
    _validate_plot_args(sample_rate, max_bins)
    frame_count, channel_count = data.shape
    duration = frame_count / sample_rate
    fig, axes = _make_axes(channel_count, title="Original recording - overview")

    if frame_count <= max_bins:
        times = np.arange(frame_count) / sample_rate
        for channel, ax in enumerate(axes):
            ax.plot(times, data[:, channel], lw=0.8)
    else:
        # Edges partition all frames into nonempty consecutive bins.
        edges = np.linspace(0, frame_count, max_bins + 1, dtype=np.int64)
        starts = edges[:-1]
        bin_centers = (edges[:-1] + edges[1:] - 1) / (2 * sample_rate)
        for channel, ax in enumerate(axes):
            minimum = np.minimum.reduceat(data[:, channel], starts)
            maximum = np.maximum.reduceat(data[:, channel], starts)
            ax.vlines(bin_centers, minimum, maximum, lw=0.65)

    for ax in axes:
        ax.set_xlim(0, duration)
        ax.set_ylim(-1.05, 1.05)
    return _save_figure(fig, output_path, keep_open=keep_open)


def plot_waveform_zoom(
    samples: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    highlight_frame: int | None = None,
    title: str = "Original recording - detailed view",
    keep_open: bool = False,
) -> Path:
    """Save original samples from a short interval, with no decimation.

    If highlight_frame is specified, a dashed line marks the boundary
    immediately AFTER that frame (e.g. between samples k and k+1).
    """
    data = _channel_matrix(samples)
    _validate_plot_args(sample_rate)
    if not np.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("Zoom start time must be finite and nonnegative.")
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Zoom duration must be finite and positive.")

    frame_count, channel_count = data.shape
    start = min(int(start_seconds * sample_rate), frame_count - 1)
    length = max(2, int(round(duration_seconds * sample_rate)))
    end = min(frame_count, start + length)
    time = np.arange(start, end) / sample_rate
    fig, axes = _make_axes(channel_count, title=title)

    for channel, ax in enumerate(axes):
        ax.plot(time, data[start:end, channel], lw=0.9)
        # Mark individual samples when the interval is short enough.
        if end - start <= 600:
            ax.plot(time, data[start:end, channel], linestyle="None", marker=".", ms=2.5)
        if highlight_frame is not None and start <= highlight_frame < end - 1:
            ax.axvline((highlight_frame + 0.5) / sample_rate, ls="--", lw=1.2)
        ax.set_xlim(time[0], time[-1] if len(time) > 1 else time[0] + 1 / sample_rate)

    return _save_figure(fig, output_path, keep_open=keep_open)


def find_largest_sample_jump(samples: np.ndarray) -> tuple[int, int, float]:
    """Find the largest |y[k+1]-y[k]| in the RAW recording.

    Returns (k, channel_index, jump_amplitude). This is just an
    exploratory discontinuity measure, NOT an impulse-noise detector:
    musical attacks, percussion and other valid sounds can be abrupt.
    """
    data = _channel_matrix(samples)
    differences = np.abs(np.diff(data, axis=0))
    before_frame, channel = np.unravel_index(
        int(np.argmax(differences)), differences.shape
    )
    return int(before_frame), int(channel), float(differences[before_frame, channel])

# Step 11: comparison and diagnostic figures. These functions ONLY render
# supplied data; no WAV samples or analysis CSV files are modified.


def _checked_pair(original: np.ndarray, cleaned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate matched original/processed audio (mono or multi-channel)."""
    raw = _channel_matrix(original)
    result = _channel_matrix(cleaned)
    if raw.shape != result.shape:
        raise ValueError(f"Original and cleaned WAV shapes differ: {raw.shape} != {result.shape}.")
    return raw, result


def _draw_overview(ax: plt.Axes, samples: np.ndarray, rate: int, max_bins: int) -> None:
    """Render all extrema, rather than arbitrarily subsampling individual points."""
    frames = samples.size
    if frames <= max_bins:
        ax.plot(np.arange(frames) / rate, samples, linewidth=0.7)
    else:
        edges = np.linspace(0, frames, max_bins + 1, dtype=np.int64)
        centers = (edges[:-1] + edges[1:] - 1) / (2 * rate)
        ax.vlines(
            centers, np.minimum.reduceat(samples, edges[:-1]),
            np.maximum.reduceat(samples, edges[:-1]), linewidth=0.65,
        )


def plot_comparison_overview(
    original: np.ndarray,
    cleaned: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
    *,
    max_bins: int = 3500,
    keep_open: bool = False,
) -> Path:
    """Full original/cleaned waveforms on separate aligned axes per channel."""
    raw, result = _checked_pair(original, cleaned)
    _validate_plot_args(sample_rate, max_bins)
    frames, channels = raw.shape
    fig, axes = plt.subplots(
        2 * channels, 1, figsize=(13, 3.3 * channels + 1.6),
        squeeze=False, sharex=True, constrained_layout=True,
    )
    fig.suptitle("Original vs. cleaned — complete recording (min/max envelopes)")
    for ch in range(channels):
        for row, data, label in (
            (2 * ch, raw[:, ch], "Original"),
            (2 * ch + 1, result[:, ch], "Cleaned"),
        ):
            ax = axes[row, 0]
            _draw_overview(ax, data, sample_rate, max_bins)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlim(0, frames / sample_rate)
            ax.set_ylabel(f"Ch {ch + 1} {label}\nAmplitude")
            ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Time [s]")
    return _save_figure(fig, output_path, keep_open=keep_open)


def plot_comparison_zoom(
    original: np.ndarray,
    cleaned: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    impulses: list[dict],
    bursts: list[dict],
    title: str,
    keep_open: bool = False,
) -> Path:
    """Show both waveforms and detector/reconstruction locations at sample scale.

    `impulses` and `bursts` are parsed records from the COMPLETE run.py CSVs.
    Flags are *candidates*, not independently verified real clicks.
    """
    raw, result = _checked_pair(original, cleaned)
    _validate_plot_args(sample_rate)
    if not np.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("Zoom start must be finite and nonnegative.")
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Zoom duration must be finite and positive.")
    frames, channels = raw.shape
    start = min(int(start_seconds * sample_rate), frames - 1)
    length = max(2, int(round(duration_seconds * sample_rate)))
    end = min(frames, start + length)
    times = np.arange(start, end) / sample_rate
    fig, axes = _make_axes(channels, title=title)

    for ch, ax in enumerate(axes, start=1):
        ax.plot(times, raw[start:end, ch - 1], color="tab:gray", lw=0.9, label="Original")
        ax.plot(times, result[start:end, ch - 1], color="tab:blue", lw=1.0,
                label="Cleaned")
        if end - start <= 600:
            ax.plot(times, result[start:end, ch - 1], linestyle="none", marker=".",
                    color="tab:blue", markersize=2.5)

        # Mark only the detected points inside the visible window.
        local_indices = [
            int(item["sample_index"]) for item in impulses
            if int(item["channel"]) == ch and start <= int(item["sample_index"]) < end
        ]
        if local_indices:
            visible = np.asarray(local_indices, dtype=np.int64)
            ax.scatter(visible / sample_rate, raw[visible, ch - 1],
                       color="tab:red", marker="x", s=32, zorder=5,
                       label="Flagged by detector")

        # Reconstructed/skipped regions are shown separately: flag != repair.
        statuses = set()
        for burst in bursts:
            if int(burst["channel"]) != ch:
                continue
            first, last = int(burst["start_index"]), int(burst["end_index"])
            if first >= end or last < start:
                continue
            status = burst["status"]
            color = "tab:green" if status == "reconstructed" else "tab:orange"
            label = ("Reconstructed burst" if status == "reconstructed"
                     else "Skipped burst") if status not in statuses else None
            ax.axvspan((first - 0.5) / sample_rate, (last + 0.5) / sample_rate,
                       color=color, alpha=0.14, label=label)
            statuses.add(status)
        ax.legend(loc="best", fontsize=8)
        ax.set_xlim(times[0], times[-1] if len(times) > 1 else times[0] + 1 / sample_rate)
    return _save_figure(fig, output_path, keep_open=keep_open)


def plot_ar_parameters(parameters, output_path: str | Path, *, keep_open: bool = False) -> Path:
    """Four AR(4) coefficient trajectories, separately for each recorded channel.

    `parameters` is the sparsely logged DataFrame produced by run.py.
    Coefficients are the POST-update estimates at logged sample indices.
    """
    channel_ids = sorted(int(c) for c in parameters["channel"].unique())
    if not channel_ids:
        raise ValueError("No parameter snapshots to plot.")
    fig, axes = plt.subplots(
        4, len(channel_ids), figsize=(6.7 * len(channel_ids), 10.5),
        squeeze=False, sharex="col", constrained_layout=True,
    )
    fig.suptitle("AR(4) — EW-LS coefficient trajectories (logged snapshots)")
    for column, ch in enumerate(channel_ids):
        rows = parameters.loc[parameters["channel"] == ch].sort_values("sample_index")
        for i in range(4):
            ax = axes[i, column]
            ax.plot(rows["time_s"], rows[f"a{i + 1}"], lw=0.85)
            ax.set_ylabel(f"a{i + 1}")
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title(f"Channel {ch}")
        axes[-1, column].set_xlabel("Time [s]")
    return _save_figure(fig, output_path, keep_open=keep_open)


def plot_prediction_errors(
    parameters,
    impulses: list[dict],
    output_path: str | Path,
    *,
    keep_open: bool = False,
) -> Path:
    """Plot sampled absolute errors and thresholds, plus *all* flagged errors.

    Because run.py logs EW-LS snapshots with a stride, the lines do not
    contain every residual. The markers come from detected_impulses.csv
    without subsampling; avoid interpreting line segments as full history.
    """
    channel_ids = sorted(int(c) for c in parameters["channel"].unique())
    if not channel_ids:
        raise ValueError("No residual snapshots to plot.")
    fig, axes = _make_axes(
        len(channel_ids),
        title="Prediction residuals — sampled history and every detector flag",
    )
    for ax, ch in zip(axes, channel_ids):
        rows = parameters.loc[parameters["channel"] == ch].sort_values("sample_index")
        ax.plot(rows["time_s"], np.abs(rows["error"]), lw=0.7,
                label="|error| at saved snapshots")
        ax.plot(rows["time_s"], rows["detection_threshold"], lw=0.9,
                label="3σ at saved snapshots")
        flagged = [item for item in impulses if int(item["channel"]) == ch]
        if flagged:
            ax.scatter([item["time_s"] for item in flagged],
                       [abs(item["error"]) for item in flagged], color="tab:red",
                       marker=".", s=7, alpha=0.45, label="All detector flags")
        ax.set_title(f"Channel {ch} | {len(flagged)} flagged samples", fontsize=10)
        ax.set_ylabel("Magnitude (normalized)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(fig._suptitle.get_text() + "\nConnected lines are sparse snapshots, not every sample")
    return _save_figure(fig, output_path, keep_open=keep_open)


def decimate_extrema_for_panorama(
    times: np.ndarray,
    signal: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep each time bin's actual local minimum and maximum chronologically.

    The panorama and other plots can share this pixel-budget reduction.
    Unlike naive stride sampling, a one-sample click remains visible.
    """
    n = signal.size
    if n <= max_points:
        return times, signal

    bins = max(1, max_points // 2)
    edges = np.linspace(0, n, bins + 1, dtype=int)
    x_out: list[float] = []
    y_out: list[float] = []
    for i in range(bins):
        start = edges[i]
        end = edges[i + 1]
        if end <= start:
            continue
        segment = signal[start:end]
        local_min_idx = int(np.argmin(segment))
        local_max_idx = int(np.argmax(segment))
        idx_a = start + local_min_idx
        idx_b = start + local_max_idx
        indices = [idx_a, idx_b] if idx_a <= idx_b else [idx_b, idx_a]
        for idx in dict.fromkeys(indices):
            x_out.append(times[idx])
            y_out.append(signal[idx])
    return np.asarray(x_out), np.asarray(y_out)
