#!/usr/bin/env python3
"""Summarize repository storage, source code, images, and video runtime.

The scan is read-only. It follows the repository's ``shared`` symlink while
ignoring dependency, VCS, and bytecode directories.
"""

from __future__ import annotations

import argparse
from collections import Counter
import heapq
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time


IMAGE_EXTENSIONS = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
}
VIDEO_EXTENSIONS = {
    ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"
}
CODE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp",
    ".java", ".js", ".jsx", ".m", ".mm", ".py", ".rs", ".sh", ".slurm",
    ".ts", ".tsx",
}
CODE_ROOT_NAMES = {"src", "scripts", "tests"}
SKIP_DIRECTORY_NAMES = {
    ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".svn",
    "__pycache__", "node_modules",
}
ROOT_SKIP_NAMES = {".conda_env", ".venv", "venv"}
SHARED_SKIP_NAMES = {"cache", "isaacsim_6.1"}
MODEL_EXTENSIONS = {".engine", ".onnx", ".pt", ".pth", ".safetensors", ".tflite"}
DOCUMENT_EXTENSIONS = {".md", ".rst"}
IMAGE_SAMPLE_SIZE = 500
TOP_COUNT = 15


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(size) < 1024.0 or unit == "PiB":
            return f"{size:,.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def human_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours >= 1:
        return f"{int(hours):,}h {int(minutes):02d}m {secs:04.1f}s"
    return f"{int(minutes):,}m {secs:04.1f}s"


def video_duration_ffprobe(path: Path, ffprobe: str) -> float | None:
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )
        duration = float(result.stdout.strip())
        return duration if duration >= 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def video_metadata_opencv(path: Path) -> dict[str, float] | None:
    try:
        import cv2  # Imported lazily so all non-video stats work without it.

        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if fps > 0 and frames >= 0:
            return {
                "seconds": frames / fps, "fps": fps, "frames": frames,
                "width": width, "height": height,
            }
    except Exception:
        pass
    return None


def count_lines(path: Path) -> tuple[int, int, int] | None:
    try:
        total = nonblank = todo = 0
        with path.open("rb") as handle:
            for line in handle:
                total += 1
                if line.strip():
                    nonblank += 1
                upper = line.upper()
                todo += upper.count(b"TODO") + upper.count(b"FIXME")
        return total, nonblank, todo
    except OSError:
        return None


def iter_files(root: Path):
    """Yield files, following directory links once and avoiding link loops."""
    stack = [(root, 0)]
    visited: set[tuple[int, int]] = set()
    while stack:
        directory, depth = stack.pop()
        try:
            stat = directory.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in visited:
                continue
            visited.add(identity)
            entries = os.scandir(directory)
        except OSError as exc:
            print(f"warning: cannot scan {directory}: {exc}", file=sys.stderr)
            continue
        with entries:
            for entry in entries:
                path = Path(entry.path)
                try:
                    if entry.is_dir(follow_symlinks=True):
                        if entry.name in SKIP_DIRECTORY_NAMES:
                            continue
                        if depth == 0 and entry.name in ROOT_SKIP_NAMES:
                            continue
                        if depth == 1 and directory.name == "shared" and entry.name in SHARED_SKIP_NAMES:
                            continue
                        stack.append((path, depth + 1))
                    elif entry.is_file(follow_symlinks=True):
                        yield path
                except OSError:
                    continue


def render_markdown(data: dict) -> str:
    """Render scan results as a standalone Markdown report."""
    total = data["total"]
    areas = data["areas"]
    outcomes = data["report_outcomes"]
    lines = [
        "# Project Statistics",
        "",
        f"> Generated by `scripts/tools/project_stats.py` in {data['scan_seconds']:,.1f} seconds.",
        "",
        "## At a glance",
        "",
        "| Metric | Total |",
        "|---|---:|",
        f"| Project data | **{human_bytes(total.get('bytes', 0))}** |",
        f"| Files | **{total.get('files', 0):,}** |",
        f"| Images | **{total.get('images', 0):,}** ({human_bytes(total.get('image_bytes', 0))}) |",
        f"| Videos | **{total.get('videos', 0):,}** ({human_bytes(total.get('video_bytes', 0))}) |",
        f"| Combined video runtime | **{human_duration(data['video_seconds'])}** |",
        f"| Experiment/run directories | **{data['run_count']:,}** |",
        f"| Source code | **{total.get('code_lines', 0):,} lines** |",
        f"| Tests | **{total.get('test_functions', 0):,} functions** |",
        f"| YOLO annotations | **{total.get('yolo_objects', 0):,} objects** |",
        f"| Model files | **{total.get('model_files', 0):,}** |",
        f"| USD scenes | **{total.get('scene_files', 0):,}** |",
        "",
        "## Storage by area",
        "",
        "| Area | Size | Files | Images | Videos |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("repository", "Repository, excluding `shared/`"),
        ("shared", "Shared assets"),
        ("shared_runs", "Shared runs"),
    ):
        values = areas.get(key, {})
        lines.append(
            f"| {label} | {human_bytes(values.get('bytes', 0))} | "
            f"{values.get('files', 0):,} | {values.get('images', 0):,} | "
            f"{values.get('videos', 0):,} |"
        )

    lines += [
        "",
        "## Source code and documentation",
        "",
        "| Language | Files | Lines | Nonblank lines |",
        "|---|---:|---:|---:|",
    ]
    for extension, values in sorted(
        data["code_by_extension"].items(),
        key=lambda item: item[1].get("lines", 0), reverse=True,
    ):
        lines.append(
            f"| `.{extension}` | {values.get('files', 0):,} | "
            f"{values.get('lines', 0):,} | {values.get('nonblank', 0):,} |"
        )
    lines += [
        "",
        f"- TODO/FIXME markers: **{total.get('todo_fixme', 0):,}**",
        f"- Test files: **{total.get('test_files', 0):,}**",
        f"- Test functions: **{total.get('test_functions', 0):,}**",
        f"- Documentation: **{total.get('document_words', 0):,} words** in "
        f"**{total.get('document_files', 0):,} files**",
        "",
        "## Machine-learning and simulation assets",
        "",
        f"- Models: **{total.get('model_files', 0):,} files** consuming "
        f"**{human_bytes(total.get('model_bytes', 0))}**",
        f"- USD scenes: **{total.get('scene_files', 0):,} files** consuming "
        f"**{human_bytes(total.get('scene_bytes', 0))}**",
        f"- YOLO labels: **{total.get('yolo_objects', 0):,} objects** across "
        f"**{total.get('yolo_label_files', 0):,} files**",
        "",
        "### YOLO objects by class ID",
        "",
        "| Class ID | Objects | Share |",
        "|---:|---:|---:|",
    ]
    object_total = max(1, total.get("yolo_objects", 0))
    for class_id, count in data["yolo_objects_by_class_id"].items():
        lines.append(f"| {class_id} | {count:,} | {100 * count / object_total:.1f}% |")

    lines += [
        "",
        "## Media",
        "",
        f"- Generated frames: **{total.get('frame_images', 0):,}** consuming "
        f"**{human_bytes(total.get('frame_bytes', 0))}**",
        f"- Videos: **{total.get('videos', 0):,}**, totaling "
        f"**{human_duration(data['video_seconds'])}**",
        f"- Unreadable videos: **{len(data['unreadable_videos']):,}**",
        f"- Image resolution sample: **{data['sampled_images']:,} images**",
        "",
        "### Sampled image resolutions",
        "",
        "| Resolution | Images | Share of sample |",
        "|---|---:|---:|",
    ]
    sample_total = max(1, data["sampled_images"])
    for resolution, count in data["image_resolution_sample"].items():
        lines.append(f"| {resolution} | {count:,} | {100 * count / sample_total:.1f}% |")

    lines += [
        "",
        "### Video formats",
        "",
        "| Resolution | Videos |",
        "|---|---:|",
    ]
    for resolution, count in data["video_resolutions"].items():
        lines.append(f"| {resolution} | {count:,} |")
    lines += ["", "| Frame rate | Videos |", "|---|---:|"]
    for fps, count in data["video_fps"].items():
        lines.append(f"| {fps} FPS | {count:,} |")

    lines += [
        "",
        "## Runs and cleanup estimate",
        "",
        f"- Run directories: **{data['run_count']:,}**",
        f"- Attempt report files: **{total.get('attempt_report_files', 0):,}**",
        f"- Attempt-report-only runs: **{outcomes.get('attempt_report_only', 0):,}**",
        f"- Runs with failure reports: **{outcomes.get('failed', 0):,}**",
        f"- Failure report files: **{total.get('failure_report_files', 0):,}**",
        f"- Runs containing both frames and encoded video: "
        f"**{data['runs_with_frames_and_video']:,}**",
        f"- Estimated frame storage in those runs: "
        f"**{human_bytes(data['estimated_reclaimable_frame_bytes'])}**",
        "",
        "> The cleanup figure is an estimate, not an automatic deletion recommendation. "
        "Frames may still be useful for re-encoding, annotation, or debugging. An attempt "
        "report without a failure report is not proof of successful remediation.",
        "",
        "### Largest runs",
        "",
        "| Run | Size | Files | Images | Videos | Runtime |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in data["largest_runs"][:10]:
        name = str(run["name"]).replace("|", "\\|")
        lines.append(
            f"| `{name}` | {human_bytes(run.get('bytes', 0))} | "
            f"{run.get('files', 0):,} | {run.get('images', 0):,} | "
            f"{run.get('videos', 0):,} | {human_duration(run.get('video_seconds', 0))} |"
        )

    lines += [
        "",
        "## Largest individual files",
        "",
        "| File | Size |",
        "|---|---:|",
    ]
    for item in data["largest_files"][:10]:
        name = str(item["path"]).replace("|", "\\|")
        lines.append(f"| `{name}` | {human_bytes(item['bytes'])} |")

    lines += [
        "",
        "## Largest file types",
        "",
        "| Extension | Files | Storage |",
        "|---|---:|---:|",
    ]
    for item in data["top_extensions_by_bytes"][:10]:
        lines.append(
            f"| `{item['extension']}` | {item.get('files', 0):,} | "
            f"{human_bytes(item.get('bytes', 0))} |"
        )

    git = data.get("git", {})
    if git:
        lines += [
            "",
            "## Git history",
            "",
            f"- Commits: **{git.get('commits', 0):,}**",
            f"- Contributors: **{git.get('contributors', 0):,}**",
            f"- First commit: **{git.get('first_commit') or 'unknown'}**",
            f"- Latest commit: **{git.get('latest_commit') or 'unknown'}**",
        ]

    lines += [
        "",
        "## Methodology and limitations",
        "",
        "- Dependency environments, Git internals, configured caches, and vendor installations are excluded.",
        "- Source totals cover maintained project code rather than copied SDK code.",
        "- Image resolutions use reservoir sampling instead of decoding every image.",
        "- Video metadata is read from every video.",
        "- Run outcomes are inferred from report-file presence and are not a measured success rate.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", nargs="?", type=Path, default=Path(__file__).resolve().parents[2],
        help="repository root (default: inferred from this script)",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--json", action="store_true", help="print machine-readable JSON")
    output_group.add_argument(
        "--markdown", nargs="?", const="__DEFAULT__", metavar="PATH",
        help="write a Markdown report (default path: PROJECT_STATS.md)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    started = time.monotonic()
    totals = Counter()
    areas: dict[str, Counter] = {
        "repository": Counter(), "shared": Counter(), "shared_runs": Counter()
    }
    code_by_language: dict[str, Counter] = {}
    extension_stats: dict[str, Counter] = {}
    run_stats: dict[str, Counter] = {}
    run_reports: dict[str, set[str]] = {}
    failure_stages = Counter()
    yolo_classes = Counter()
    largest_files: list[tuple[int, str]] = []
    image_sample: list[Path] = []
    image_seen = 0
    rng = random.Random(2026)
    video_resolutions = Counter()
    video_fps = Counter()
    video_seconds = 0.0
    unreadable_videos: list[str] = []
    ffprobe = shutil.which("ffprobe")

    for index, path in enumerate(iter_files(root), start=1):
        try:
            file_stat = path.stat()
            size = file_stat.st_size
            relative = path.relative_to(root)
        except (OSError, ValueError):
            continue
        suffix = path.suffix.lower()
        extension = suffix or "[no extension]"
        parts = relative.parts
        area = "repository"
        if parts and parts[0] == "shared":
            area = "shared_runs" if len(parts) > 1 and parts[1] == "runs" else "shared"

        totals["files"] += 1
        totals["bytes"] += size
        areas[area]["files"] += 1
        areas[area]["bytes"] += size
        extension_stats.setdefault(extension, Counter())["files"] += 1
        extension_stats[extension]["bytes"] += size
        totals["oldest_mtime"] = min(totals.get("oldest_mtime", int(file_stat.st_mtime)), int(file_stat.st_mtime))
        totals["newest_mtime"] = max(totals.get("newest_mtime", 0), int(file_stat.st_mtime))
        item = (size, str(relative))
        if len(largest_files) < TOP_COUNT:
            heapq.heappush(largest_files, item)
        elif size > largest_files[0][0]:
            heapq.heapreplace(largest_files, item)

        run_name = None
        if len(parts) >= 3 and parts[:2] == ("shared", "runs"):
            run_name = parts[2]
        elif len(parts) >= 2 and parts[0] == "runs":
            run_name = f"local/{parts[1]}"
        if run_name:
            run = run_stats.setdefault(run_name, Counter())
            run["files"] += 1
            run["bytes"] += size

        if suffix in IMAGE_EXTENSIONS:
            totals["images"] += 1
            totals["image_bytes"] += size
            areas[area]["images"] += 1
            if run_name:
                run_stats[run_name]["images"] += 1
                if any(part.endswith("frames") or part == "frames" for part in parts):
                    run_stats[run_name]["frame_bytes"] += size
                    totals["frame_images"] += 1
                    totals["frame_bytes"] += size
            image_seen += 1
            if len(image_sample) < IMAGE_SAMPLE_SIZE:
                image_sample.append(path)
            else:
                replacement = rng.randrange(image_seen)
                if replacement < IMAGE_SAMPLE_SIZE:
                    image_sample[replacement] = path
        elif suffix in VIDEO_EXTENSIONS:
            totals["videos"] += 1
            totals["video_bytes"] += size
            areas[area]["videos"] += 1
            if run_name:
                run_stats[run_name]["videos"] += 1
            duration = (
                video_duration_ffprobe(path, ffprobe) if ffprobe else None
            )
            metadata = video_metadata_opencv(path)
            if duration is None and metadata is not None:
                duration = metadata["seconds"]
            if duration is None:
                unreadable_videos.append(str(relative))
            else:
                video_seconds += duration
                if run_name:
                    run_stats[run_name]["video_seconds"] += duration
            if metadata is not None:
                resolution = f"{int(metadata['width'])}x{int(metadata['height'])}"
                video_resolutions[resolution] += 1
                video_fps[f"{metadata['fps']:.1f}"] += 1

        if suffix in MODEL_EXTENSIONS:
            totals["model_files"] += 1
            totals["model_bytes"] += size
        if suffix in {".usd", ".usda", ".usdc"}:
            totals["scene_files"] += 1
            totals["scene_bytes"] += size

        if run_name and path.name in {"attempt_report.json", "failure_report.json"}:
            report_type = "failure" if path.name == "failure_report.json" else "attempt"
            totals[f"{report_type}_report_files"] += 1
            run_reports.setdefault(run_name, set()).add(report_type)
            try:
                report = json.loads(path.read_text(errors="replace"))
                if report_type == "failure":
                    stage = str(report.get("stage") or report.get("error_type") or "unknown")
                    failure_stages[stage] += 1
            except (OSError, json.JSONDecodeError):
                totals["unreadable_reports"] += 1

        if suffix == ".txt" and "labels" in parts and size < 5_000_000:
            try:
                object_count = 0
                with path.open("r", errors="replace") as label_file:
                    for row in label_file:
                        fields = row.split()
                        if fields:
                            yolo_classes[fields[0]] += 1
                            object_count += 1
                totals["yolo_label_files"] += 1
                totals["yolo_objects"] += object_count
            except OSError:
                pass

        # Count maintained project code, not copied SDKs, generated runs, or data.
        is_project_code = len(parts) == 1 or (parts and parts[0] in CODE_ROOT_NAMES)
        if suffix in CODE_EXTENSIONS and is_project_code:
            lines = count_lines(path)
            if lines is not None:
                total_lines, nonblank_lines, todo_count = lines
                language = suffix.removeprefix(".")
                bucket = code_by_language.setdefault(language, Counter())
                bucket["files"] += 1
                bucket["lines"] += total_lines
                bucket["nonblank"] += nonblank_lines
                totals["code_files"] += 1
                totals["code_lines"] += total_lines
                totals["nonblank_code_lines"] += nonblank_lines
                totals["todo_fixme"] += todo_count
                if path.name.startswith("test_") or path.name.endswith("_test.py"):
                    try:
                        source = path.read_text(errors="replace")
                        totals["test_files"] += 1
                        totals["test_functions"] += sum(
                            1 for row in source.splitlines()
                            if row.lstrip().startswith(("def test_", "async def test_"))
                        )
                    except OSError:
                        pass

        if suffix in DOCUMENT_EXTENSIONS and (
            len(parts) == 1 or (parts and parts[0] in {"docs", "results"})
        ):
            try:
                text = path.read_text(errors="replace")
                totals["document_files"] += 1
                totals["document_words"] += len(text.split())
            except OSError:
                pass

        if not args.json and index % 100_000 == 0:
            print(f"Scanned {index:,} files...", file=sys.stderr)

    image_resolutions = Counter()
    unreadable_sample_images = 0
    try:
        import cv2
        for image_path in image_sample:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                unreadable_sample_images += 1
            else:
                height, width = image.shape[:2]
                image_resolutions[f"{width}x{height}"] += 1
    except Exception:
        unreadable_sample_images = len(image_sample)

    report_outcomes = Counter()
    for run_name, reports in run_reports.items():
        if "failure" in reports:
            report_outcomes["failed"] += 1
        elif "attempt" in reports:
            report_outcomes["attempt_report_only"] += 1
    reclaimable_frame_bytes = sum(
        values["frame_bytes"] for values in run_stats.values()
        if values["videos"] and values["frame_bytes"]
    )
    reclaimable_runs = sum(
        1 for values in run_stats.values() if values["videos"] and values["frame_bytes"]
    )

    git_stats = {}
    if (root / ".git").exists():
        def git(*arguments: str) -> str:
            try:
                return subprocess.run(
                    ["git", *arguments], cwd=root, capture_output=True, check=True,
                    text=True, timeout=30,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                return ""
        commit_dates = git("log", "--format=%aI").splitlines()
        git_stats = {
            "commits": int(git("rev-list", "--count", "HEAD") or 0),
            "contributors": len(set(git("log", "--format=%aN").splitlines())),
            "first_commit": commit_dates[-1] if commit_dates else "",
            "latest_commit": commit_dates[0] if commit_dates else "",
        }

    elapsed = time.monotonic() - started
    output = {
        "root": str(root),
        "scan_seconds": round(elapsed, 3),
        "total": dict(totals),
        "areas": {name: dict(values) for name, values in areas.items()},
        "code_by_extension": {
            name: dict(values) for name, values in sorted(code_by_language.items())
        },
        "video_seconds": round(video_seconds, 3),
        "video_minutes": round(video_seconds / 60.0, 3),
        "unreadable_videos": unreadable_videos,
        "image_resolution_sample": dict(image_resolutions.most_common()),
        "sampled_images": len(image_sample),
        "unreadable_sample_images": unreadable_sample_images,
        "video_resolutions": dict(video_resolutions.most_common()),
        "video_fps": dict(video_fps.most_common()),
        "run_count": len(run_stats),
        "report_outcomes": dict(report_outcomes),
        "failure_stages": dict(failure_stages.most_common()),
        "yolo_objects_by_class_id": dict(yolo_classes.most_common()),
        "estimated_reclaimable_frame_bytes": reclaimable_frame_bytes,
        "runs_with_frames_and_video": reclaimable_runs,
        "largest_files": [
            {"path": path, "bytes": size}
            for size, path in sorted(largest_files, reverse=True)
        ],
        "largest_runs": [
            {"name": name, **dict(values)}
            for name, values in sorted(run_stats.items(), key=lambda item: item[1]["bytes"], reverse=True)[:TOP_COUNT]
        ],
        "top_extensions_by_bytes": [
            {"extension": name, **dict(values)}
            for name, values in sorted(extension_stats.items(), key=lambda item: item[1]["bytes"], reverse=True)[:TOP_COUNT]
        ],
        "git": git_stats,
    }
    if args.json:
        print(json.dumps(output, indent=2))
        return 0
    if args.markdown:
        destination = (
            root / "PROJECT_STATS.md"
            if args.markdown == "__DEFAULT__"
            else Path(args.markdown).expanduser().resolve()
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_markdown(output), encoding="utf-8")
        print(f"Markdown report written to {destination}")
        return 0

    print(f"\nProject statistics: {root.name}")
    print(f"  Total data:       {human_bytes(totals['bytes'])} in {totals['files']:,} files")
    print(f"  Images:           {totals['images']:,} ({human_bytes(totals['image_bytes'])})")
    print(f"  Videos:           {totals['videos']:,} ({human_bytes(totals['video_bytes'])})")
    print(f"  Video runtime:    {human_duration(video_seconds)} ({video_seconds / 60:,.1f} min)")
    print(f"  Source code:      {totals['code_lines']:,} lines in {totals['code_files']:,} files")
    print(f"  Nonblank code:    {totals['nonblank_code_lines']:,} lines")
    print(f"  TODO/FIXME:       {totals['todo_fixme']:,}")
    print(f"  Model files:      {totals['model_files']:,} ({human_bytes(totals['model_bytes'])})")
    print(f"  USD scenes:       {totals['scene_files']:,} ({human_bytes(totals['scene_bytes'])})")
    print(f"  Documentation:    {totals['document_words']:,} words in {totals['document_files']:,} files")
    print(f"  Tests:            {totals['test_functions']:,} functions in {totals['test_files']:,} files")
    print(f"  YOLO labels:      {totals['yolo_objects']:,} objects in {totals['yolo_label_files']:,} files")
    print(f"  Generated frames: {totals['frame_images']:,} ({human_bytes(totals['frame_bytes'])})")
    print("\nStorage by area:")
    for name in ("repository", "shared", "shared_runs"):
        values = areas[name]
        label = {"repository": "Repo (not shared)", "shared": "Shared assets", "shared_runs": "Shared runs"}[name]
        print(f"  {label:18} {human_bytes(values['bytes']):>12}  {values['files']:>12,} files  {values['images']:>12,} images")
    print("\nCode by extension:")
    for extension, values in sorted(
        code_by_language.items(), key=lambda item: item[1]["lines"], reverse=True
    ):
        print(f"  .{extension:<8} {values['lines']:>12,} lines  {values['files']:>7,} files")
    print("\nRuns and cleanup estimates:")
    print(f"  Run directories:          {len(run_stats):,}")
    print(f"  Runs with failures:       {report_outcomes['failed']:,}")
    print(f"  Failure report files:     {totals['failure_report_files']:,}")
    print(f"  Attempt-report-only runs: {report_outcomes['attempt_report_only']:,}")
    print(f"  Frames with video copy:   {human_bytes(reclaimable_frame_bytes)} across {reclaimable_runs:,} runs")
    if failure_stages:
        print("  Common failure stages:    " + ", ".join(
            f"{name} ({count:,})" for name, count in failure_stages.most_common(5)
        ))
    if yolo_classes:
        print("  YOLO objects by class:    " + ", ".join(
            f"{name} ({count:,})" for name, count in yolo_classes.most_common(8)
        ))
    print("\nLargest runs:")
    for name, values in sorted(run_stats.items(), key=lambda item: item[1]["bytes"], reverse=True)[:10]:
        print(f"  {human_bytes(values['bytes']):>10}  {values['files']:>9,} files  {name}")
    print("\nMost common sampled image resolutions:")
    for resolution, count in image_resolutions.most_common(8):
        print(f"  {resolution:<14} {count:>5,} / {len(image_sample):,} sampled")
    print("\nVideo formats:")
    print("  Resolutions: " + ", ".join(f"{key} ({value})" for key, value in video_resolutions.most_common(6)))
    print("  Frame rates: " + ", ".join(f"{key} fps ({value})" for key, value in video_fps.most_common(6)))
    print("\nLargest files:")
    for size, name in sorted(largest_files, reverse=True)[:10]:
        print(f"  {human_bytes(size):>10}  {name}")
    print("\nLargest file types:")
    for extension, values in sorted(extension_stats.items(), key=lambda item: item[1]["bytes"], reverse=True)[:10]:
        print(f"  {human_bytes(values['bytes']):>10}  {extension:<14} {values['files']:>10,} files")
    if git_stats:
        print("\nGit history:")
        print(f"  {git_stats['commits']:,} commits by {git_stats['contributors']:,} contributors")
        print(f"  First: {git_stats['first_commit'] or 'unknown'}")
        print(f"  Latest: {git_stats['latest_commit'] or 'unknown'}")
    if unreadable_videos:
        print(f"\n  Warning: duration unavailable for {len(unreadable_videos):,} video(s).")
    print(f"\nScan completed in {elapsed:,.1f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
