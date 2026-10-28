"""Read completed processing results and generate the eight original PNG plots."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ProjectConfig
from ip_ii.audio import read_wav
from ip_ii.reporting import (
    BURST_COLUMNS, IMPULSE_COLUMNS, PARAMETER_COLUMNS,
    read_diagnostic_csv, check_input_identity,
)
from ip_ii.plotting import (
    find_largest_sample_jump, plot_ar_parameters, plot_comparison_overview,
    plot_comparison_zoom, plot_prediction_errors, plot_waveform_overview,
    plot_waveform_zoom,
)


class ResultAnalyzer:
    """Analyze one recording selected by ProjectConfig; never modify WAV/CSV/JSON."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def _records(self, table: pd.DataFrame) -> list[dict]:
        """Convert an optional small event table into plotting-friendly records."""
        return table.to_dict(orient="records")


    def _pick_auto_center(self, 
        original: np.ndarray, cleaned: np.ndarray, bursts: list[dict], rate: int,
    ) -> tuple[float, str]:
        """Prefer the strongest actual repair; otherwise inspect a skipped burst/jump."""
        original_2d = original[:, None] if original.ndim == 1 else original
        cleaned_2d = cleaned[:, None] if cleaned.ndim == 1 else cleaned
        candidates = []
        for burst in bursts:
            if burst["status"] != "reconstructed":
                continue
            ch = int(burst["channel"]) - 1
            start = int(burst["start_index"])
            stop = int(burst["end_index"]) + 1
            delta = np.abs(original_2d[start:stop, ch] - cleaned_2d[start:stop, ch])
            if delta.size:
                relative_index = int(np.argmax(delta))
                candidates.append((float(delta[relative_index]), start + relative_index, ch + 1))
        if candidates:
            magnitude, index, channel = max(candidates)
            return index / rate, (
                f"Largest reconstructed difference — ch {channel}, "
                f"sample {index}, |Δ|={magnitude:.5f}"
            )
        if bursts:
            burst = bursts[0]
            middle = (int(burst["start_index"]) + int(burst["end_index"])) / 2
            return middle / rate, "First flagged burst (not reconstructed)"
        frame, channel, _ = find_largest_sample_jump(original)
        return (frame + 0.5) / rate, (
            f"No flagged bursts — largest adjacent-sample jump, channel {channel + 1}"
        )


    def _centered_start(self, time_s: float, width_s: float, duration_s: float) -> float:
        return max(0.0, min(time_s - width_s / 2, max(0.0, duration_s - width_s)))


    def _validate_outputs(self, 
        original: np.ndarray, cleaned: np.ndarray, rate: int,
        diagnostics: dict, parameters: pd.DataFrame,
        impulses: pd.DataFrame, bursts: pd.DataFrame,
    ) -> int:
        """Reject mismatched/stale file sets instead of plotting misleading results."""
        channels = 1 if original.ndim == 1 else original.shape[1]
        if cleaned.shape != original.shape:
            raise ValueError("cleaned.wav differs in length or number of channels.")
        if diagnostics.get("recording") != f"{self.config.RECORDING_ID:02d}.wav":
            raise ValueError("diagnostics.json is from a different recording; rerun run.py.")
        if (diagnostics.get("sample_rate_hz") != rate
                or diagnostics.get("frames_per_channel") != len(original)
                or diagnostics.get("channels") != channels):
            raise ValueError("diagnostics.json does not match the WAV; rerun run.py.")
        if parameters.empty:
            raise ValueError("parameters.csv has no snapshots; rerun run.py.")
        for name, table in (
            ("parameters.csv", parameters),
            ("detected_impulses.csv", impulses),
            ("detected_bursts.csv", bursts),
        ):
            if table.empty:
                continue
            if (table["channel"].isna().any()
                    or not table["channel"].between(1, channels).all()
                    or (table["channel"] % 1 != 0).any()):
                raise ValueError(f"Invalid channel values in {name}.")
            index_columns = (
                ["start_index", "end_index"] if name == "detected_bursts.csv"
                else ["sample_index"]
            )
            for key in index_columns:
                if (table[key].isna().any() or not table[key].between(0, len(original)-1).all()
                        or (table[key] % 1 != 0).any()):
                    raise ValueError(f"Invalid {key} values in {name}.")
        for ch in range(1, channels + 1):
            if not (parameters["channel"] == ch).any():
                raise ValueError(f"Missing EW-LS snapshots for channel {ch}.")
        if not bursts.empty:
            if (bursts["end_index"] < bursts["start_index"]).any():
                raise ValueError("Invalid interval in detected_bursts.csv.")
            if not bursts["status"].isin(["reconstructed", "skipped"]).all():
                raise ValueError("Unknown burst status in detected_bursts.csv.")
        return channels


    def analyze(self, ) -> None:
        """Create comparison plots and print objective, non-auditory diagnostics."""
        self.config.validate()
        print("IP_II_lab - Project 1: recording and model analysis")
        print(f"Recording: {self.config.RECORDING_ID:02d}.wav")
        required_files = (
            self.config.INPUT_FILE, self.config.CLEANED_FILE, self.config.EWLS_PARAMETERS_FILE, self.config.DETECTED_IMPULSES_FILE,
            self.config.DETECTED_BURSTS_FILE, self.config.DIAGNOSTICS_FILE,
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise SystemExit(
                "Missing input/processing results:\n  " + "\n  ".join(missing)
                + "\nPlace the original WAV in data/ and run 'python run.py' "
                  "before 'python analyze.py'."
            )

        sample_rate, original = read_wav(
            self.config.INPUT_FILE, expected_sample_rate=self.config.EXPECTED_SAMPLE_RATE,
        )
        clean_rate, cleaned = read_wav(
            self.config.CLEANED_FILE, expected_sample_rate=self.config.EXPECTED_SAMPLE_RATE,
        )
        if clean_rate != sample_rate:
            raise ValueError("Sample rates of original and cleaned WAV differ.")
        with self.config.DIAGNOSTICS_FILE.open("r", encoding="utf-8") as handle:
            diagnostics = json.load(handle)
        check_input_identity(self.config.INPUT_FILE, diagnostics, self.config.CLEANED_FILE)
        parameters = read_diagnostic_csv(self.config.EWLS_PARAMETERS_FILE, PARAMETER_COLUMNS)
        impulses_df = read_diagnostic_csv(self.config.DETECTED_IMPULSES_FILE, IMPULSE_COLUMNS)
        bursts_df = read_diagnostic_csv(self.config.DETECTED_BURSTS_FILE, BURST_COLUMNS)
        channels = self._validate_outputs(
            original, cleaned, sample_rate, diagnostics,
            parameters, impulses_df, bursts_df,
        )
        impulses = self._records(impulses_df)
        bursts = self._records(bursts_df)
        duration = len(original) / sample_rate

        raw_data = original[:, None] if original.ndim == 1 else original
        cleaned_data = cleaned[:, None] if cleaned.ndim == 1 else cleaned
        difference = cleaned_data.astype(np.float64) - raw_data.astype(np.float64)
        print(f"Duration: {duration:.3f} s, sample rate: {sample_rate} Hz")
        for ch in range(channels):
            ch_diff = difference[:, ch]
            raw_rms = float(np.sqrt(np.mean(raw_data[:, ch] ** 2)))
            clean_rms = float(np.sqrt(np.mean(cleaned_data[:, ch] ** 2)))
            raw_peak = float(np.max(np.abs(raw_data[:, ch])))
            clean_peak = float(np.max(np.abs(cleaned_data[:, ch])))
            changed = int(np.count_nonzero(ch_diff))
            candidate_count = sum(int(item["channel"]) == ch + 1 for item in impulses)
            successful = sum(item["channel"] == ch + 1 and item["status"] == "reconstructed"
                             for item in bursts)
            skipped = sum(item["channel"] == ch + 1 and item["status"] == "skipped"
                          for item in bursts)
            print(
                f"Channel {ch+1}: RMS raw/clean={raw_rms:.6f}/{clean_rms:.6f}, "
                f"peak raw/clean={raw_peak:.6f}/{clean_peak:.6f}; "
                f"detector flags={candidate_count}, "
                f"reconstructed bursts={successful}, skipped bursts={skipped}, "
                f"actually changed PCM samples={changed}, "
                f"difference RMS={np.sqrt(np.mean(ch_diff ** 2)):.8f}, "
                f"maximum |difference|={np.max(np.abs(ch_diff)):.8f}"
            )
        print("The reported difference quantifies the edits, NOT subjective audio quality.")

        output = []
        # Retain the three exploratory plots from step 3 (about ORIGINAL audio).
        output.append(plot_waveform_overview(
            original, sample_rate, self.config.PLOTS_DIR / "original_overview.png",
            max_bins=self.config.PLOT_MAX_BINS, keep_open=self.config.PLOT_SHOW_INTERACTIVE,
        ))
        output.append(plot_waveform_zoom(
            original, sample_rate, self.config.PLOTS_DIR / "original_zoom.png",
            start_seconds=self.config.PLOT_ZOOM_START_S,
            duration_seconds=self.config.PLOT_ZOOM_DURATION_S,
            keep_open=self.config.PLOT_SHOW_INTERACTIVE,
        ))
        jump_frame, jump_channel, _ = find_largest_sample_jump(original)
        output.append(plot_waveform_zoom(
            original, sample_rate, self.config.PLOTS_DIR / "original_max_jump_zoom.png",
            start_seconds=self._centered_start(
                (jump_frame + 0.5) / sample_rate, self.config.PLOT_AUTO_ZOOM_DURATION_S, duration,
            ),
            duration_seconds=self.config.PLOT_AUTO_ZOOM_DURATION_S,
            highlight_frame=jump_frame,
            title=f"Original - largest adjacent-sample jump (channel {jump_channel + 1}; not a confirmed click)",
            keep_open=self.config.PLOT_SHOW_INTERACTIVE,
        ))

        output.append(plot_comparison_overview(
            original, cleaned, sample_rate,
            self.config.PLOTS_DIR / "comparison_overview.png",
            max_bins=self.config.PLOT_MAX_BINS, keep_open=self.config.PLOT_SHOW_INTERACTIVE,
        ))
        output.append(plot_comparison_zoom(
            original, cleaned, sample_rate, self.config.PLOTS_DIR / "comparison_zoom.png",
            start_seconds=self.config.PLOT_ZOOM_START_S,
            duration_seconds=self.config.PLOT_ZOOM_DURATION_S,
            impulses=impulses, bursts=bursts,
            title="Original / cleaned — manually selected detail; markers are detector candidates",
            keep_open=self.config.PLOT_SHOW_INTERACTIVE,
        ))
        focus_time, focus_reason = self._pick_auto_center(
            original, cleaned, bursts, sample_rate,
        )
        output.append(plot_comparison_zoom(
            original, cleaned, sample_rate, self.config.PLOTS_DIR / "comparison_auto_zoom.png",
            start_seconds=self._centered_start(focus_time, self.config.PLOT_AUTO_ZOOM_DURATION_S, duration),
            duration_seconds=self.config.PLOT_AUTO_ZOOM_DURATION_S,
            impulses=impulses, bursts=bursts,
            title=f"Original / cleaned — {focus_reason}",
            keep_open=self.config.PLOT_SHOW_INTERACTIVE,
        ))
        output.append(plot_ar_parameters(
            parameters, self.config.PLOTS_DIR / "ar_parameters.png",
            keep_open=self.config.PLOT_SHOW_INTERACTIVE,
        ))
        output.append(plot_prediction_errors(
            parameters, impulses, self.config.PLOTS_DIR / "prediction_errors.png",
            keep_open=self.config.PLOT_SHOW_INTERACTIVE,
        ))

        print("Saved analysis plots:")
        for path in output:
            print(f"  {path}")
        print("Analysis generates PNG files only; WAV, CSV and JSON files remain unchanged.")
        print("The assignment requires a listening-based evaluation of cleaned.wav.")
        if self.config.PLOT_SHOW_INTERACTIVE:
            plt.show()
