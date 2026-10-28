"""Object-oriented orchestration of project WAV processing and diagnostics."""

from __future__ import annotations

from pathlib import Path
import json
from hashlib import sha256
import numpy as np

from config import ProjectConfig
from ip_ii.audio import PCM16_SCALE, read_wav, write_wav
from ip_ii.ar import ARModel
from ip_ii.declick import ALGORITHM_REVISION, DeclickObservation, DeclickProcessor
from ip_ii.output_files import staged_output_files
from ip_ii.results_writer import ResultsWriter
from ip_ii.analysis import ResultAnalyzer


class ProjectRunner:
    """High-level application: reconstruct once, then generate eight plots.

    Each channel gets its own fresh DeclickProcessor and EW-LS state. All
    publication paths remain those of the legacy entry point.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.analyzer = ResultAnalyzer(config)

    def run(self, *, analyze_only: bool = False) -> None:
        if not analyze_only:
            self.process()
        self.analyzer.analyze()

    def process(self) -> None:
        """Publish complete diagnostics/WAV only if processing succeeds.

        Original file names are unchanged. Each artifact is verified before
        its temporary sibling replaces the previously saved artifact.
        """
        published_paths = (
            self.config.ROUNDTRIP_FILE, self.config.EWLS_PARAMETERS_FILE, self.config.DETECTED_IMPULSES_FILE,
            self.config.DETECTED_BURSTS_FILE, self.config.CLEANED_FILE, self.config.DIAGNOSTICS_FILE,
        )
        with staged_output_files(published_paths) as staged:
            self._execute(staged)

    def _execute(self, staged: dict[Path, Path]) -> None:
        """Run the original numerical pipeline with staged output destinations."""
        self.config.validate()
        print("IP_II_lab - Project 1: impulse removal")
        print(f"Recording: {self.config.RECORDING_ID:02d}.wav")
        if not self.config.INPUT_FILE.is_file():
            raise SystemExit(f"Missing recording: {self.config.INPUT_FILE}\nPlace it in data/ first.")

        sample_rate, samples = read_wav(self.config.INPUT_FILE, expected_sample_rate=self.config.EXPECTED_SAMPLE_RATE)
        if samples.shape[0] <= self.config.AR_ORDER:
            raise ValueError("The WAV is too short to fit AR(4).")
        channel_count = 1 if samples.ndim == 1 else samples.shape[1]
        print(f"Input: {self.config.INPUT_FILE} | {sample_rate} Hz | {channel_count} channel(s) | "
              f"{samples.shape[0] / sample_rate:.3f} s")
        print(f"Settings: AR({self.config.AR_ORDER}), lambda={self.config.EWLS_FORGETTING_FACTOR}, "
              f"M={self.config.RESIDUAL_WINDOW_SIZE}, threshold={self.config.DETECTION_THRESHOLD} sigma, "
              f"max burst={self.config.MAX_IMPULSE_LENGTH}")

        # Preserve the step-2 PCM round-trip check and unchanged copy.
        write_wav(staged[self.config.ROUNDTRIP_FILE], sample_rate, samples)
        saved_rate, saved_samples = read_wav(
            staged[self.config.ROUNDTRIP_FILE], expected_sample_rate=self.config.EXPECTED_SAMPLE_RATE,
        )
        if saved_rate != sample_rate or not np.array_equal(saved_samples, samples):
            raise RuntimeError("PCM16 round-trip failed: samples changed without processing.")
        print(f"PCM16 round-trip OK: {self.config.ROUNDTRIP_FILE}")

        restored = np.empty_like(samples)
        diagnostics_channels = []
        for path in (self.config.EWLS_PARAMETERS_FILE, self.config.DETECTED_IMPULSES_FILE,
                     self.config.DETECTED_BURSTS_FILE, self.config.DIAGNOSTICS_FILE, self.config.CLEANED_FILE):
            path.parent.mkdir(parents=True, exist_ok=True)

        with ResultsWriter(staged, self.config) as writer:
            for ch in range(channel_count):
                channel_id = ch + 1
                audio_channel = samples if samples.ndim == 1 else samples[:, ch]
                phi0 = ARModel(self.config.AR_ORDER).regressor(audio_channel, self.config.AR_ORDER)
                print(f"Channel {channel_id}: first phi = {np.array2string(phi0, precision=6)}")

                def save_observation(obs: DeclickObservation) -> None:
                    writer.observation(obs, channel_id=channel_id, sample_rate=sample_rate)

                processor = DeclickProcessor(
                    order=self.config.AR_ORDER,
                    forgetting_factor=self.config.EWLS_FORGETTING_FACTOR,
                    initial_inverse_correlation=self.config.EWLS_INITIAL_INVERSE_CORRELATION,
                    residual_window_size=self.config.RESIDUAL_WINDOW_SIZE,
                    detection_threshold=self.config.DETECTION_THRESHOLD,
                    max_burst_length=self.config.MAX_IMPULSE_LENGTH,
                    log_stride=self.config.EWLS_LOG_STRIDE,
                )
                result = processor.process(
                    audio_channel, sample_rate=sample_rate,
                    on_observation=save_observation,
                )
                if samples.ndim == 1:
                    restored[:] = result.samples
                else:
                    restored[:, ch] = result.samples

                writer.burst_rows(result, channel_id=channel_id, sample_rate=sample_rate)
                row = writer.summary(result, channel_id=channel_id)
                diagnostics_channels.append(row)
                print(f"  Detected: {row['flagged_samples']} samples / {row['candidate_bursts']} bursts; "
                      f"restored: {row['reconstructed_samples']} samples "
                      f"({row['reconstructed_bursts']} bursts); "
                      f"skipped: {row['skipped_bursts']} bursts; "
                      f"EW-LS updates: {row['ewls_updates']}; "
                      f"detector resets: {row['residual_calibration_restarts']}; "
                      f"anchors rechecked: {row['rechecked_anchor_count']}; "
                      f"initialization window: {row['initialization_segment']}")

        write_wav(staged[self.config.CLEANED_FILE], sample_rate, restored)
        saved_rate, saved_clean = read_wav(
            staged[self.config.CLEANED_FILE], expected_sample_rate=self.config.EXPECTED_SAMPLE_RATE,
        )
        if saved_rate != sample_rate or saved_clean.shape != samples.shape:
            raise RuntimeError("Saved cleaned.wav does not preserve format/shape.")
        # Linear interpolation generally produces values between PCM16 levels.
        # The recorded result may therefore differ by at most half a PCM level.
        maximum_quantization_error = float(np.max(np.abs(saved_clean - restored)))
        if maximum_quantization_error > 0.5000001 / PCM16_SCALE:
            raise RuntimeError("WAV export differs by more than half a PCM16 level.")
        print(f"Maximum PCM16 quantization difference: "
              f"{maximum_quantization_error * PCM16_SCALE:.3f} LSB")

        diagnostics = {
            "algorithm_revision": ALGORITHM_REVISION,
            "input_sha256": sha256(self.config.INPUT_FILE.read_bytes()).hexdigest(),
            "cleaned_sha256": sha256(staged[self.config.CLEANED_FILE].read_bytes()).hexdigest(),
            "recording": f"{self.config.RECORDING_ID:02d}.wav",
            "sample_rate_hz": sample_rate,
            "frames_per_channel": int(samples.shape[0]),
            "channels": channel_count,
            "settings": {
                "ar_order": self.config.AR_ORDER,
                "forgetting_factor": self.config.EWLS_FORGETTING_FACTOR,
                "initial_inverse_correlation": self.config.EWLS_INITIAL_INVERSE_CORRELATION,
                "residual_window_size": self.config.RESIDUAL_WINDOW_SIZE,
                "detection_threshold_sigma": self.config.DETECTION_THRESHOLD,
                "max_impulse_length": self.config.MAX_IMPULSE_LENGTH,
                "parameter_log_stride": self.config.EWLS_LOG_STRIDE,
                "ewls_pilot_initialization": True,
                "recompute_right_anchor_after_interpolation": True,
            },
            "channels_summary": diagnostics_channels,
            "caveats": [
                "Detection flags are candidates, not verified click positions.",
                "The first M residuals of the MAIN processing pass cannot be classified; candidate runs >4 are not interpolated.",
                "Offline pilot EW-LS calibration uses up to the first 0.5 s to seed theta(0); no future data is added to the main residual window.",
                "The main estimator retains its configured lambda and P(0), and samples are processed from the recording start.",
                "Candidate right anchors after short bursts are checked using an interpolated AR context before an EW-LS update.",
                "Only accepted errors update the residual history; only accepted samples update EW-LS.",
                "Skipped-burst raw output remains unchanged, while internal AR context uses bounded predictions.",
                "After >4 consecutive detections, residual history is restarted to avoid permanent lockout (additional heuristic).",
            ],
        }
        with staged[self.config.DIAGNOSTICS_FILE].open("w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(f"Cleaned WAV:    {self.config.CLEANED_FILE}")
        print(f"Parameters:     {self.config.EWLS_PARAMETERS_FILE}")
        print(f"Flagged points: {self.config.DETECTED_IMPULSES_FILE}")
        print(f"Bursts/status:  {self.config.DETECTED_BURSTS_FILE}")
        print(f"Diagnostics:    {self.config.DIAGNOSTICS_FILE}")
        print("Inspect and listen to cleaned.wav: detection may also flag genuine musical attacks.")

