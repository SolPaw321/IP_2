"""CLI for the class-based panoramic comparison plot."""

import argparse
from pathlib import Path
from ip_ii.panorama import PanoramaPlotter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a very long panoramic PNG comparing original and cleaned audio."
    )
    parser.add_argument("--recording-id", type=int, default=4)
    parser.add_argument("--root-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--inches-per-second", type=float, default=20.0,
                        help="Larger value makes a longer panorama.")
    parser.add_argument("--height", type=float, default=4.8)
    parser.add_argument("--dpi", type=int, default=160)
    options = parser.parse_args()
    PanoramaPlotter(
        recording_id=options.recording_id, root_dir=options.root_dir,
        inches_per_second=options.inches_per_second,
        height=options.height, dpi=options.dpi,
    ).generate()


if __name__ == "__main__":
    main()
