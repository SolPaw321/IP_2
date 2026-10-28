"""Entry point for AR(4)/EW-LS impulse restoration and diagnostic plots.

    python run.py                 # full processing + eight standard PNGs
    python run.py --analyze-only  # regenerate standard PNGs from existing data
"""

import argparse

from config import ProjectConfig
from ip_ii.pipeline import ProjectRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AR(4) EW-LS impulse removal and result analysis."
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Regenerate plots without changing WAV, CSV, or JSON outputs.",
    )
    args = parser.parse_args()
    ProjectRunner(ProjectConfig()).run(analyze_only=args.analyze_only)


if __name__ == "__main__":
    main()
