"""Stage outputs next to their original paths and publish on successful completion.

Data are written to sibling .tmp files first; exceptions leave the previously
published output set intact. Final filenames and file formats do not change.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator, Iterable


@contextmanager
def staged_output_files(paths: Iterable[Path]) -> Iterator[dict[Path, Path]]:
    """Yield {final_path: temporary_path}, then replace published files.

    Each individual file replacement is atomic on the same filesystem.
    Publishing a multi-file set is not a single atomic transaction.
    """
    staged: dict[Path, Path] = {}
    for final in paths:
        final.parent.mkdir(parents=True, exist_ok=True)
        staged[final] = final.with_name(final.name + ".tmp")
    try:
        yield staged
        for final, temporary in staged.items():
            if not temporary.is_file():
                raise RuntimeError(f"Expected completed staged output: {temporary}")
        for final, temporary in staged.items():
            temporary.replace(final)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
