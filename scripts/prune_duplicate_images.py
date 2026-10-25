#!/usr/bin/env python3
"""Center-crop a dataset and remove duplicate cropped images.

The default input is the cap/seal image directory in this repository. Each
image is cropped to 50% of its original width and height, centered on the
image, and written to the repository's writable ``images/pruned_images``
directory. Duplicate
images are detected from the cropped RGB pixels, so different file metadata
does not prevent duplicate detection.

Example:
    python scripts/prune_duplicate_images.py

The input and output can be overridden for use on another machine:
    python scripts/prune_duplicate_images.py /path/to/images \
        --output /path/to/pruned_images
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "shared/data/csi_vial_fill_dataset/cap_seal/images"
DEFAULT_CROP_FRACTION = 0.50

# Pillow's registered extensions include formats such as JPEG, PNG, TIFF,
# BMP, WEBP, and GIF. GIFs are handled as their first frame below.
IMAGE_EXTENSIONS = {
    extension.lower()
    for extension in Image.registered_extensions()
    if extension.lower() in {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
}


def image_paths(input_dir: Path, recursive: bool) -> Iterator[Path]:
    """Yield supported image files in deterministic order."""

    paths = input_dir.rglob("*") if recursive else input_dir.glob("*")
    for path in sorted(paths, key=lambda item: item.relative_to(input_dir).as_posix().lower()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def center_crop(image: Image.Image, fraction: float) -> Image.Image:
    """Return the centered crop retaining ``fraction`` of each dimension."""

    width, height = image.size
    crop_width = max(1, int(round(width * fraction)))
    crop_height = max(1, int(round(height * fraction)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def crop_digest(image: Image.Image) -> str:
    """Hash normalized cropped pixels, ignoring metadata and image mode."""

    normalized = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{normalized.width}x{normalized.height}:RGB\0".encode())
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def save_crop(image: Image.Image, output_path: Path, source_format: str | None) -> None:
    """Save a crop while retaining the source file format where possible."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {}

    # JPEG cannot store alpha or palette modes. Converting all JPEG output to
    # RGB also ensures the output can be opened consistently by training tools.
    if (source_format or "").upper() in {"JPEG", "JPG"}:
        image = image.convert("RGB")
        save_kwargs.update(quality=95, subsampling=0, optimize=True)

    image.save(output_path, format=source_format, **save_kwargs)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create an empty output directory, refusing accidental mixed results."""

    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {output_dir}")
        existing = list(output_dir.iterdir())
        if existing and not overwrite:
            raise ValueError(
                f"Output directory is not empty: {output_dir}. "
                "Choose another output path or pass --overwrite."
            )
        if overwrite:
            for path in existing:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    else:
        output_dir.mkdir(parents=True)


def prune_images(
    input_dir: Path,
    output_dir: Path,
    crop_fraction: float,
    recursive: bool,
    overwrite: bool,
) -> tuple[int, int, int]:
    """Process images and return ``(written, duplicates, failures)``."""

    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output directories must be different")

    prepare_output_dir(output_dir, overwrite)
    seen: dict[str, Path] = {}
    written = duplicates = failures = 0

    for source_path in image_paths(input_dir, recursive):
        relative_path = source_path.relative_to(input_dir)
        try:
            with Image.open(source_path) as source_image:
                # Apply EXIF orientation before cropping so "center" refers to
                # the displayed image rather than its unrotated storage layout.
                oriented = ImageOps.exif_transpose(source_image)
                cropped = center_crop(oriented, crop_fraction)
                digest = crop_digest(cropped)

                if digest in seen:
                    duplicates += 1
                    print(
                        f"duplicate: {source_path} (same as {seen[digest]})",
                        file=sys.stderr,
                    )
                    continue

                output_path = output_dir / relative_path
                save_crop(cropped, output_path, source_image.format)
                seen[digest] = source_path
                written += 1
        except Exception as exc:  # Keep one bad image from stopping the run.
            failures += 1
            print(f"skip: {source_path}: {exc}", file=sys.stderr)

    return written, duplicates, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Center-crop images to 50%% of width/height and remove duplicate "
            "cropped images."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input image directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Output directory (default: {REPO_ROOT / 'images/pruned_images'})",
    )
    parser.add_argument(
        "--crop-fraction",
        type=float,
        default=DEFAULT_CROP_FRACTION,
        metavar="FRACTION",
        help="Fraction retained from each dimension (default: 0.50)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process images in subdirectories and preserve their relative paths",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear an existing output directory before writing results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.crop_fraction <= 1:
        print("error: --crop-fraction must be greater than 0 and at most 1", file=sys.stderr)
        return 2

    input_dir = args.input.expanduser().resolve()
    output_dir = (
        args.output.expanduser().resolve()
        if args.output
        else REPO_ROOT / "images/pruned_images"
    )

    try:
        written, duplicates, failures = prune_images(
            input_dir=input_dir,
            output_dir=output_dir,
            crop_fraction=args.crop_fraction,
            recursive=args.recursive,
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Output: {output_dir}")
    print(f"Images written: {written}")
    print(f"Duplicates removed: {duplicates}")
    print(f"Images skipped: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
