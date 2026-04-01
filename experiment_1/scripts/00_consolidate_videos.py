"""
00_consolidate_videos.py

One-time setup script. Moves all .mp4 files from the nested split_* directories
into a single flat folder so inference scripts can find videos by filename alone.

Usage:
    python scripts/00_consolidate_videos.py \
        --src /scratch/jjtribb/EgoBlind_Videos \
        --dst /scratch/jjtribb/EgoBlind_Videos/flat \
        --csv data/test_half_release.csv

    Add --dry_run to preview without moving anything.
"""

import argparse
import os
import shutil
import pandas as pd
from pathlib import Path


def main(args):
    src = Path(args.src)
    dst = Path(args.dst)

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    # Find all .mp4 files under src (excluding dst itself to avoid recursion)
    all_mp4s = []
    for path in src.rglob("*.mp4"):
        if dst in path.parents or path.parent == dst:
            continue
        all_mp4s.append(path)

    print(f"Found {len(all_mp4s)} .mp4 files under {src}")

    # Check for duplicate filenames across splits
    seen = {}
    duplicates = []
    for path in all_mp4s:
        name = path.name
        if name in seen:
            duplicates.append((name, seen[name], path))
        else:
            seen[name] = path

    if duplicates:
        print(f"WARNING: {len(duplicates)} duplicate filenames found:")
        for name, first, second in duplicates:
            print(f"  {name}: {first} vs {second}")

    # Move files into flat directory
    moved = 0
    skipped = 0
    for path in all_mp4s:
        target = dst / path.name
        if args.dry_run:
            print(f"[DRY RUN] Would move: {path} -> {target}")
            moved += 1
        else:
            if target.exists():
                skipped += 1
                continue
            shutil.move(str(path), str(target))
            moved += 1

    print(f"\nMoved: {moved}, Skipped (already exist): {skipped}")

    # Cross-check against CSV to find any missing videos
    if args.csv:
        df = pd.read_csv(args.csv)
        # video_name column is an integer like 923; format as 5-digit zero-padded
        csv_names = set(f"{int(v):05d}.mp4" for v in df["video_name"].unique())
        flat_names = set(p.name for p in dst.glob("*.mp4")) if not args.dry_run else set(seen.keys())
        missing = csv_names - flat_names
        if missing:
            print(f"\nWARNING: {len(missing)} videos in CSV not found in flat dir:")
            for name in sorted(missing)[:20]:
                print(f"  {name}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
        else:
            print(f"\nAll {len(csv_names)} unique videos from CSV are present in flat dir.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/scratch/jjtribb/EgoBlind_Videos",
                        help="Root directory containing split_* subdirectories")
    parser.add_argument("--dst", default="/scratch/jjtribb/EgoBlind_Videos/flat",
                        help="Destination flat directory")
    parser.add_argument("--csv", default="data/test_half_release.csv",
                        help="Path to test_half_release.csv for cross-checking")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview actions without moving files")
    args = parser.parse_args()
    main(args)
