"""Central configuration for IP_II_lab: single WAV, AR(4), EW-LS and experiments."""

from pathlib import Path


# All paths are relative to the directory containing this file.
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

# One recording
RECORDING_ID = 6
INPUT_FILE = DATA_DIR / f"{RECORDING_ID:02d}.wav"
OUTPUT_DIR = RESULTS_DIR / f"{RECORDING_ID:02d}"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Step 2: unmodified PCM16 copy (not the later declicked output).
EXPECTED_SAMPLE_RATE = 22_050
ROUNDTRIP_FILE = OUTPUT_DIR / "roundtrip.wav"

# Step 3: visualization of the original (still noisy) recording.
# These settings control plots only: they do NOT affect processing.
PLOT_MAX_BINS = 3500                # Min/max pairs for the full overview.
PLOT_ZOOM_START_S = 2.0             # Start of user-selected detail [s].
PLOT_ZOOM_DURATION_S = 0.03        # Detail window duration [s].
PLOT_AUTO_ZOOM_DURATION_S = 0.03  # Around the largest one-step amplitude jump.
PLOT_SHOW_INTERACTIVE = False     # True: also show plots with the GUI toolbar.

# Fixed requirements from proj_1_pl.pdf.
AR_ORDER = 4
DETECTION_THRESHOLD = 3  # Prediction error threshold: 3 local std deviations.
MAX_IMPULSE_LENGTH = 4      # Maximum length of a corrupted sample sequence.

# To be chosen and justified in later steps (NOT assignment defaults).
# Step 5: preliminary numerical choices (NOT prescribed by the project PDF).
# lambda=0.999 gives effective weight sum 1/(1-lambda) ~= 1000 samples
# (~45 ms at 22050 Hz). The trade-off will be explored in step 12.
EWLS_FORGETTING_FACTOR = 0.999
EWLS_INITIAL_INVERSE_CORRELATION = 100.0  # P(0) = 100*I (initialization).
EWLS_LOG_STRIDE = 220                      # Approx. one CSV row per 10 ms.
EWLS_PARAMETERS_FILE = OUTPUT_DIR / "parameters.csv"

# Step 6: use M previous errors for local mean-squared residuals.
# M=20 is an example from wyklad_1.pdf, p. 49, NOT a fixed project demand.
RESIDUAL_WINDOW_SIZE = 20

# Step 7: a diagnostic CSV with every flagged sample (not reconstructed yet).
DETECTED_IMPULSES_FILE = OUTPUT_DIR / "detected_impulses.csv"

# Step 8/10: candidate bursts with reconstruction status.
DETECTED_BURSTS_FILE = OUTPUT_DIR / "detected_bursts.csv"

# Step 10: restored WAV and machine-readable processing summary.
CLEANED_FILE = OUTPUT_DIR / "cleaned.wav"
DIAGNOSTICS_FILE = OUTPUT_DIR / "diagnostics.json"


def validate_config() -> None:
    """Check values that must be valid already in project step 1."""
    if not 1 <= RECORDING_ID <= 25:
        raise ValueError("RECORDING_ID must be an integer between 1 and 25.")
    if EXPECTED_SAMPLE_RATE != 22_050:
        raise ValueError("Project 1 specifies a 22050 Hz WAV sampling rate.")
    if AR_ORDER != 4:
        raise ValueError("Project 1 explicitly requires the AR(4) model.")
    if DETECTION_THRESHOLD != 3.0:
        raise ValueError("Project 1 explicitly requires a 3-sigma threshold.")
    if MAX_IMPULSE_LENGTH != 4:
        raise ValueError("Project 1 assumes up to 4 consecutive corrupted samples.")
    if PLOT_MAX_BINS < 1:
        raise ValueError("PLOT_MAX_BINS must be positive.")
    if not (0 <= PLOT_ZOOM_START_S < float("inf")):
        raise ValueError("PLOT_ZOOM_START_S must be finite and nonnegative.")
    if not (0 < PLOT_ZOOM_DURATION_S < float("inf")):
        raise ValueError("PLOT_ZOOM_DURATION_S must be finite and positive.")
    if not (0 < PLOT_AUTO_ZOOM_DURATION_S < float("inf")):
        raise ValueError("PLOT_AUTO_ZOOM_DURATION_S must be finite and positive.")
    if not 0 < EWLS_FORGETTING_FACTOR <= 1:
        raise ValueError("EWLS_FORGETTING_FACTOR must be within (0, 1].")
    if not 0 < EWLS_INITIAL_INVERSE_CORRELATION < float("inf"):
        raise ValueError("EWLS_INITIAL_INVERSE_CORRELATION must be finite and positive.")
    if isinstance(EWLS_LOG_STRIDE, bool) or not isinstance(EWLS_LOG_STRIDE, int) or EWLS_LOG_STRIDE < 1:
        raise ValueError("EWLS_LOG_STRIDE must be a positive integer.")
    if (isinstance(RESIDUAL_WINDOW_SIZE, bool)
            or not isinstance(RESIDUAL_WINDOW_SIZE, int)
            or RESIDUAL_WINDOW_SIZE < 1):
        raise ValueError("RESIDUAL_WINDOW_SIZE must be a positive integer.")


from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ProjectConfig:
    """Immutable snapshot of the existing project parameters and paths.

    All defaults come from the exact legacy constants above. This preserves
    the current config.py editing workflow without introducing a second set
    of numerical defaults. A derived recording may be selected explicitly.
    """
    ROOT_DIR: Path = ROOT_DIR
    DATA_DIR: Path = DATA_DIR
    RESULTS_DIR: Path = RESULTS_DIR
    RECORDING_ID: int = RECORDING_ID
    INPUT_FILE: Path = INPUT_FILE
    OUTPUT_DIR: Path = OUTPUT_DIR
    PLOTS_DIR: Path = PLOTS_DIR
    EXPECTED_SAMPLE_RATE: int = EXPECTED_SAMPLE_RATE
    ROUNDTRIP_FILE: Path = ROUNDTRIP_FILE
    PLOT_MAX_BINS: int = PLOT_MAX_BINS
    PLOT_ZOOM_START_S: float = PLOT_ZOOM_START_S
    PLOT_ZOOM_DURATION_S: float = PLOT_ZOOM_DURATION_S
    PLOT_AUTO_ZOOM_DURATION_S: float = PLOT_AUTO_ZOOM_DURATION_S
    PLOT_SHOW_INTERACTIVE: bool = PLOT_SHOW_INTERACTIVE
    AR_ORDER: int = AR_ORDER
    DETECTION_THRESHOLD: float = DETECTION_THRESHOLD
    MAX_IMPULSE_LENGTH: int = MAX_IMPULSE_LENGTH
    EWLS_FORGETTING_FACTOR: float = EWLS_FORGETTING_FACTOR
    EWLS_INITIAL_INVERSE_CORRELATION: float = EWLS_INITIAL_INVERSE_CORRELATION
    EWLS_LOG_STRIDE: int = EWLS_LOG_STRIDE
    EWLS_PARAMETERS_FILE: Path = EWLS_PARAMETERS_FILE
    RESIDUAL_WINDOW_SIZE: int = RESIDUAL_WINDOW_SIZE
    DETECTED_IMPULSES_FILE: Path = DETECTED_IMPULSES_FILE
    DETECTED_BURSTS_FILE: Path = DETECTED_BURSTS_FILE
    CLEANED_FILE: Path = CLEANED_FILE
    DIAGNOSTICS_FILE: Path = DIAGNOSTICS_FILE

    def validate(self) -> None:
        """Check values that must be valid already in project step 1."""
        if not 1 <= self.RECORDING_ID <= 25:
            raise ValueError("RECORDING_ID must be an integer between 1 and 25.")
        if self.EXPECTED_SAMPLE_RATE != 22_050:
            raise ValueError("Project 1 specifies a 22050 Hz WAV sampling rate.")
        if self.AR_ORDER != 4:
            raise ValueError("Project 1 explicitly requires the AR(4) model.")
        if self.MAX_IMPULSE_LENGTH != 4:
            raise ValueError("Project 1 assumes up to 4 consecutive corrupted samples.")
        if self.PLOT_MAX_BINS < 1:
            raise ValueError("PLOT_MAX_BINS must be positive.")
        if not (0 <= self.PLOT_ZOOM_START_S < float("inf")):
            raise ValueError("PLOT_ZOOM_START_S must be finite and nonnegative.")
        if not (0 < self.PLOT_ZOOM_DURATION_S < float("inf")):
            raise ValueError("PLOT_ZOOM_DURATION_S must be finite and positive.")
        if not (0 < self.PLOT_AUTO_ZOOM_DURATION_S < float("inf")):
            raise ValueError("PLOT_AUTO_ZOOM_DURATION_S must be finite and positive.")
        if not 0 < self.EWLS_FORGETTING_FACTOR <= 1:
            raise ValueError("EWLS_FORGETTING_FACTOR must be within (0, 1].")
        if not 0 < self.EWLS_INITIAL_INVERSE_CORRELATION < float("inf"):
            raise ValueError("EWLS_INITIAL_INVERSE_CORRELATION must be finite and positive.")
        if isinstance(self.EWLS_LOG_STRIDE, bool) or not isinstance(self.EWLS_LOG_STRIDE, int) or self.EWLS_LOG_STRIDE < 1:
            raise ValueError("EWLS_LOG_STRIDE must be a positive integer.")
        if (isinstance(self.RESIDUAL_WINDOW_SIZE, bool)
                or not isinstance(self.RESIDUAL_WINDOW_SIZE, int)
                or self.RESIDUAL_WINDOW_SIZE < 1):
            raise ValueError("RESIDUAL_WINDOW_SIZE must be a positive integer.")


    def for_recording(self, recording_id: int) -> "ProjectConfig":
        """Return equivalent paths for another recording, without mutating self."""
        output = self.RESULTS_DIR / f"{recording_id:02d}"
        result = replace(
            self, RECORDING_ID=recording_id,
            INPUT_FILE=self.DATA_DIR / f"{recording_id:02d}.wav",
            OUTPUT_DIR=output, PLOTS_DIR=output / "plots",
            ROUNDTRIP_FILE=output / "roundtrip.wav",
            EWLS_PARAMETERS_FILE=output / "parameters.csv",
            DETECTED_IMPULSES_FILE=output / "detected_impulses.csv",
            DETECTED_BURSTS_FILE=output / "detected_bursts.csv",
            CLEANED_FILE=output / "cleaned.wav",
            DIAGNOSTICS_FILE=output / "diagnostics.json",
        )
        result.validate()
        return result


# Step 12: compare a small, deliberately non-exhaustive set of candidate settings.
# The first case mirrors the normal ``python run.py`` settings.
# These are EXPERIMENTAL hypotheses, not mandated defaults or proven optima.
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
EXPERIMENT_CASES = (
    {
        "name": "baseline",
        "forgetting_factor": EWLS_FORGETTING_FACTOR,
        "residual_window_size": RESIDUAL_WINDOW_SIZE,
        "initial_inverse_correlation": EWLS_INITIAL_INVERSE_CORRELATION,
    },
    {
        "name": "faster_ewls",
        "forgetting_factor": 0.995,
        "residual_window_size": RESIDUAL_WINDOW_SIZE,
        "initial_inverse_correlation": EWLS_INITIAL_INVERSE_CORRELATION,
    },
    {
        "name": "slower_ewls",
        "forgetting_factor": 0.9995,
        "residual_window_size": RESIDUAL_WINDOW_SIZE,
        "initial_inverse_correlation": EWLS_INITIAL_INVERSE_CORRELATION,
    },
    {
        "name": "longer_window",
        "forgetting_factor": EWLS_FORGETTING_FACTOR,
        "residual_window_size": 80,
        "initial_inverse_correlation": EWLS_INITIAL_INVERSE_CORRELATION,
    },
    {
        "name": "fast_and_long",
        "forgetting_factor": 0.995,
        "residual_window_size": 80,
        "initial_inverse_correlation": EWLS_INITIAL_INVERSE_CORRELATION,
    },
    {
        "name": "higher_initial_p",
        "forgetting_factor": EWLS_FORGETTING_FACTOR,
        "residual_window_size": RESIDUAL_WINDOW_SIZE,
        "initial_inverse_correlation": 1000.0,
    },
)
