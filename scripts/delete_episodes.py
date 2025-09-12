#!/usr/bin/env python3

import argparse
import contextlib
import json
import os
from pathlib import Path

import jsonlines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete specific episodes from a LeRobot dataset")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=os.environ.get("DATASET_REPO_ID"),
        help="Hugging Face dataset repo id (e.g. user/dataset). Defaults to $DATASET_REPO_ID",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        required=True,
        help="Comma-separated episode indices to delete, e.g. '12,15,23'",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Local dataset root. Defaults to ~/.cache/huggingface/lerobot/{repo-id}",
    )
    parser.add_argument(
        "--update-hub",
        action="store_true",
        help="Also delete files from the Hugging Face Hub and upload updated meta",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without performing deletions",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [row for row in jsonlines.open(path)]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as w:
        w.write_all(rows)


def main() -> None:
    args = parse_args()
    if not args.repo_id:
        raise SystemExit("--repo-id is required (or set $DATASET_REPO_ID)")

    repo_id = args.repo_id
    root = Path(args.root_dir) if args.root_dir else Path.home() / ".cache/huggingface/lerobot" / repo_id
    info_path = root / "meta/info.json"
    episodes_path = root / "meta/episodes.jsonl"
    episodes_stats_path = root / "meta/episodes_stats.jsonl"

    if not info_path.exists():
        raise SystemExit(f"info.json not found: {info_path}")

    try:
        to_remove = sorted({int(x) for x in args.episodes.split(",") if x.strip()})
    except ValueError as e:
        raise SystemExit(f"Invalid --episodes value: {args.episodes}") from e

    info = load_json(info_path)
    chunks_size = int(info["chunks_size"])  # Max episodes per chunk
    video_keys = [k for k, ft in info["features"].items() if ft["dtype"] == "video"]

    def ep_chunk(ep: int) -> int:
        return ep // chunks_size

    def parquet_rel(ep: int) -> str:
        return f"data/chunk-{ep_chunk(ep):03d}/episode_{ep:06d}.parquet"

    def mp4_rel(ep: int, key: str) -> str:
        return f"videos/chunk-{ep_chunk(ep):03d}/{key}/episode_{ep:06d}.mp4"

    # Collect existing episodes to keep/remove
    eps_rows = read_jsonl(episodes_path)
    existing_indices = {row["episode_index"] for row in eps_rows}
    actually_remove = [ep for ep in to_remove if ep in existing_indices]
    skipped_missing = [ep for ep in to_remove if ep not in existing_indices]

    if args.dry_run:
        print("[DRY-RUN] Would remove:")
        for ep in actually_remove:
            print(" -", parquet_rel(ep))
            for k in video_keys:
                print(" -", mp4_rel(ep, k))
        if skipped_missing:
            print("[DRY-RUN] Episodes not found and skipped:", skipped_missing)
        return

    # Delete local files
    for ep in actually_remove:
        (root / parquet_rel(ep)).unlink(missing_ok=True)
        for k in video_keys:
            (root / mp4_rel(ep, k)).unlink(missing_ok=True)

    # Update meta/episodes.jsonl
    kept_rows = [row for row in eps_rows if row["episode_index"] not in actually_remove]
    write_jsonl(kept_rows, episodes_path)

    # Update meta/episodes_stats.jsonl
    stats_rows = read_jsonl(episodes_stats_path)
    if stats_rows:
        stats_rows = [row for row in stats_rows if row["episode_index"] not in actually_remove]
        write_jsonl(stats_rows, episodes_stats_path)

    # Update meta/info.json
    total_episodes = len(kept_rows)
    total_frames = int(sum(row.get("length", 0) for row in kept_rows)) if kept_rows else 0
    remaining_indices = [row["episode_index"] for row in kept_rows]
    total_chunks = (max(remaining_indices) // chunks_size + 1) if remaining_indices else 0
    total_videos = (len(video_keys) * total_episodes) if video_keys else 0

    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["total_chunks"] = total_chunks
    info["total_videos"] = total_videos
    write_json(info, info_path)

    # Optionally update Hub
    if args.update_hub and actually_remove:
        from huggingface_hub import HfApi

        api = HfApi()
        for ep in actually_remove:
            with contextlib.suppress(Exception):
                api.delete_file(repo_id=repo_id, repo_type="dataset", path_in_repo=parquet_rel(ep))
            for k in video_keys:
                with contextlib.suppress(Exception):
                    api.delete_file(repo_id=repo_id, repo_type="dataset", path_in_repo=mp4_rel(ep, k))

        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=episodes_path,
            path_in_repo="meta/episodes.jsonl",
        )
        if episodes_stats_path.exists():
            api.upload_file(
                repo_id=repo_id,
                repo_type="dataset",
                path_or_fileobj=episodes_stats_path,
                path_in_repo="meta/episodes_stats.jsonl",
            )
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=info_path,
            path_in_repo="meta/info.json",
        )

    print("Removed episodes:", actually_remove)
    if skipped_missing:
        print("Skipped (not found):", skipped_missing)
    print("Now total_episodes:", total_episodes, "total_frames:", total_frames)


if __name__ == "__main__":
    main()

