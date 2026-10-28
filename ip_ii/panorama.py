
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from config import EXPECTED_SAMPLE_RATE
from ip_ii.audio import read_wav
from ip_ii.plotting import decimate_extrema_for_panorama
from ip_ii.reporting import (
    BURST_COLUMNS, IMPULSE_COLUMNS, check_input_identity, read_diagnostic_csv,
)


def read_wav_float(path: Path) -> tuple[int, np.ndarray]:
    """Load PCM16 using the same validation and normalization as run.py."""
    sample_rate, data = read_wav(path, expected_sample_rate=EXPECTED_SAMPLE_RATE)
    if data.ndim == 1:
        data = data[:, None]
    return sample_rate, data


def load_optional_csv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    """An absent event table is empty; present tables must be well-formed."""
    if not path.is_file():
        return pd.DataFrame()
    return read_diagnostic_csv(path, required)


def _filter_channel(
    df: pd.DataFrame,
    channel_index_zero_based: int,
) -> pd.DataFrame:

    if df.empty or "channel" not in df.columns:
        return pd.DataFrame()

    # Support 0-based and 1-based channel numbering.
    channel_values = pd.to_numeric(
        df["channel"], errors="coerce"
    )

    if (channel_values == 0).any():
        channel_label = channel_index_zero_based
    else:
        channel_label = channel_index_zero_based + 1

    filtered = df[
        channel_values == channel_label
    ].copy()

    if filtered.empty:
        return filtered

    sort_cols = [
        col
        for col in [
            "sample_index",
            "start_index",
            "time_s",
            "start_time_s",
        ]
        if col in filtered.columns
    ]

    if sort_cols:
        filtered = filtered.sort_values(sort_cols)

    return filtered


def _status_column_name(
    df: pd.DataFrame,
) -> Optional[str]:

    for name in [
        "reconstruction_status",
        "status",
        "burst_status",
    ]:
        if name in df.columns:
            return name

    return None


class PanoramaPlotter:
    """Create the same long, single-row panorama PNGs as the original CLI.

    One output per audio channel; output names and sample markers are unchanged.
    """

    def __init__(
        self, *, recording_id: int = 5, root_dir: Path | None = None,
        inches_per_second: float = 20.0, height: float = 4.8, dpi: int = 160,
    ) -> None:
        self.recording_id = recording_id
        self.root_dir = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[1]
        self.inches_per_second = inches_per_second
        self.height = height
        self.dpi = dpi

    def plot_channel(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        sample_rate: int,
        recording_id: int,
        channel_index: int,
        impulses: pd.DataFrame,
        bursts: pd.DataFrame,
        inches_per_second: float,
        height_in: float,
        dpi: int,
        output_path: Path,
    ) -> None:

        n_samples = original.shape[0]

        total_duration_s = n_samples / sample_rate

        times = (
            np.arange(n_samples, dtype=np.float64)
            / sample_rate
        )

        # --------------------------------------------------
        # Panorama dimensions
        # --------------------------------------------------

        # Horizontal stretching is controlled here.
        width_in = max(
            18.0,
            total_duration_s * inches_per_second,
        )

        width_px = width_in * dpi

        if width_px >= 65536:
            raise ValueError(
                f"Panorama would be {width_px:.0f} pixels wide. "
                "Reduce --inches-per-second or --dpi "
                "to stay below the Matplotlib image limit."
            )

        max_points = int(width_in * dpi * 1.5)

        x_original, y_original = decimate_extrema_for_panorama(
            times,
            original,
            max_points=max_points,
        )

        x_cleaned, y_cleaned = decimate_extrema_for_panorama(
            times,
            cleaned,
            max_points=max_points,
        )

        fig, ax = plt.subplots(
            figsize=(width_in, height_in)
        )

        status_column = _status_column_name(bursts)

        # --------------------------------------------------
        # Reconstructed and skipped bursts
        # --------------------------------------------------

        if (
            not bursts.empty
            and "start_time_s" in bursts.columns
            and "end_time_s" in bursts.columns
        ):

            for _, burst in bursts.iterrows():

                burst_start = float(
                    burst["start_time_s"]
                )

                burst_end = float(
                    burst["end_time_s"]
                )

                status = None

                if status_column is not None:
                    status = str(
                        burst[status_column]
                    ).strip().lower()

                elif "within_max_length" in bursts.columns:
                    status = (
                        "reconstructed"
                        if bool(burst["within_max_length"])
                        else "skipped"
                    )

                if status in {
                    "reconstructed",
                    "accepted",
                    "restored",
                }:
                    color = "green"
                    alpha = 0.12
                    label = "Reconstructed burst"

                else:
                    color = "orange"
                    alpha = 0.18
                    label = "Skipped burst"

                # Sample centers are half a sample from the visible boundaries.
                ax.axvspan(
                    max(0.0, burst_start - 0.5 / sample_rate),
                    min(total_duration_s, burst_end + 0.5 / sample_rate),
                    color=color,
                    alpha=alpha,
                    label=label,
                    zorder=0,
                )

        # --------------------------------------------------
        # Original signal
        # --------------------------------------------------

        ax.plot(
            x_original,
            y_original,
            label="Original",
            linewidth=0.8,
            alpha=0.75,
            color="gray",
            zorder=2,
        )

        # --------------------------------------------------
        # Cleaned signal
        # --------------------------------------------------

        ax.plot(
            x_cleaned,
            y_cleaned,
            label="Cleaned",
            linewidth=0.9,
            color="tab:blue",
            zorder=3,
        )

        # --------------------------------------------------
        # Detected impulses
        # --------------------------------------------------

        if (
            not impulses.empty
            and "time_s" in impulses.columns
        ):

            impulse_times = impulses[
                "time_s"
            ].to_numpy(dtype=np.float64)

            if "observed" in impulses.columns:

                y_imp = impulses[
                    "observed"
                ].to_numpy(dtype=np.float64)

            else:

                y_imp = np.interp(
                    impulse_times,
                    times,
                    original,
                )

            ax.scatter(
                impulse_times,
                y_imp,
                marker="x",
                s=18,
                linewidths=1.2,
                color="red",
                label="Flagged by detector",
                zorder=4,
            )

        # --------------------------------------------------
        # Axes
        # --------------------------------------------------

        ax.set_xlim(
            0.0,
            total_duration_s,
        )

        ax.set_xlabel(
            "Time [s]"
        )

        ax.set_ylabel(
            f"Channel {channel_index + 1}\nAmplitude"
        )

        ax.set_title(
            "Original / cleaned panorama — "
            f"recording {recording_id:02d}, "
            f"channel {channel_index + 1}"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        # --------------------------------------------------
        # Legend
        # --------------------------------------------------

        handles, labels = ax.get_legend_handles_labels()

        unique_handles = []
        unique_labels = []
        seen = set()

        for handle, label in zip(handles, labels):

            if label and label not in seen:

                seen.add(label)

                unique_handles.append(handle)
                unique_labels.append(label)

        ax.legend(
            unique_handles,
            unique_labels,
            loc="upper left",
        )

        # --------------------------------------------------
        # Save panorama
        # --------------------------------------------------

        fig.tight_layout()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=dpi,
        )

        plt.close(fig)

    def generate(self) -> None:
        """Read existing WAV/CSV/JSON diagnostics and render panorama(s)."""
        root = self.root_dir

        data_dir = root / "data"

        results_dir = (
            root
            / "results"
            / f"{self.recording_id:02d}"
        )

        plots_dir = results_dir / "plots"

        original_path = (
            data_dir
            / f"{self.recording_id:02d}.wav"
        )

        cleaned_path = (
            results_dir
            / "cleaned.wav"
        )

        impulses_path = (
            results_dir
            / "detected_impulses.csv"
        )

        bursts_path = (
            results_dir
            / "detected_bursts.csv"
        )

        # --------------------------------------------------
        # Validate input files
        # --------------------------------------------------

        if not original_path.exists():

            raise FileNotFoundError(
                f"Missing original file: {original_path}"
            )

        if not cleaned_path.exists():

            raise FileNotFoundError(
                f"Missing cleaned file: {cleaned_path}. "
                "Run python run.py first."
            )

        # --------------------------------------------------
        # Read WAV files
        # --------------------------------------------------

        sr_original, original = read_wav_float(
            original_path
        )

        sr_cleaned, cleaned = read_wav_float(
            cleaned_path
        )

        if sr_original != sr_cleaned:

            raise ValueError(
                "Sampling-rate mismatch: "
                f"original={sr_original}, "
                f"cleaned={sr_cleaned}"
            )

        if original.shape != cleaned.shape:

            raise ValueError(
                "Shape mismatch: "
                f"original={original.shape}, "
                f"cleaned={cleaned.shape}"
            )

        # --------------------------------------------------
        # Read detection results
        # --------------------------------------------------

        impulses = load_optional_csv(
            impulses_path, IMPULSE_COLUMNS
        )

        bursts = load_optional_csv(
            bursts_path, BURST_COLUMNS
        )

        # --------------------------------------------------
        # Generate panorama for each channel
        # --------------------------------------------------

        # The recording ID is explicitly chosen by --recording-id (default 05).
        # Check the metadata if available; old diagnostic sets remain readable.
        diagnostics_path = results_dir / "diagnostics.json"
        if diagnostics_path.is_file():
            with diagnostics_path.open("r", encoding="utf-8") as handle:
                diagnostics = json.load(handle)
            if diagnostics.get("recording") != f"{self.recording_id:02d}.wav":
                raise ValueError("Diagnostic metadata belongs to a different recording.")
            if (diagnostics.get("sample_rate_hz") != sr_original
                    or diagnostics.get("frames_per_channel") != original.shape[0]
                    or diagnostics.get("channels") != original.shape[1]):
                raise ValueError("Diagnostic metadata does not match current WAV dimensions.")
            check_input_identity(original_path, diagnostics, cleaned_path)

        n_channels = original.shape[1]

        for channel_index in range(n_channels):

            channel_impulses = _filter_channel(
                impulses,
                channel_index,
            )

            channel_bursts = _filter_channel(
                bursts,
                channel_index,
            )

            suffix = (
                f"_channel_{channel_index + 1}"
                if n_channels > 1
                else ""
            )

            output_path = (
                plots_dir
                / f"comparison_mosaic{suffix}.png"
            )

            self.plot_channel(
                original=original[:, channel_index],
                cleaned=cleaned[:, channel_index],
                sample_rate=sr_original,
                recording_id=self.recording_id,
                channel_index=channel_index,
                impulses=channel_impulses,
                bursts=channel_bursts,
                inches_per_second=self.inches_per_second,
                height_in=self.height,
                dpi=self.dpi,
                output_path=output_path,
            )

            print(f"Saved: {output_path}")
