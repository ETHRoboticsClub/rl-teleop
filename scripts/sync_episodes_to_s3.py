#!/usr/bin/env python3
"""Sync completed local episode recordings to S3.

The script loads AWS credentials and sync settings from .env, then shells out to
the AWS CLI so large files use the CLI's mature transfer behavior.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REQUIRED_AWS_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


def load_dotenv(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from a dotenv file."""
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_no}: expected KEY=VALUE")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"{path}:{line_no}: empty key")

        if value and value[0] in ("'", '"'):
            try:
                parsed = shlex.split(value, posix=True)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid quoted value: {exc}") from exc
            value = parsed[0] if parsed else ""
        else:
            value = value.split(" #", 1)[0].strip()

        values[key] = value
    return values


def merged_env(dotenv: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in dotenv.items():
        if value != "":
            env.setdefault(key, value)

    if "AWS_REGION" in env and "AWS_DEFAULT_REGION" not in env:
        env["AWS_DEFAULT_REGION"] = env["AWS_REGION"]
    elif "AWS_DEFAULT_REGION" in env and "AWS_REGION" not in env:
        env["AWS_REGION"] = env["AWS_DEFAULT_REGION"]

    return env


def normalize_s3_uri(uri: str) -> str:
    uri = uri.rstrip("/")
    if not uri.startswith("s3://") or uri == "s3://":
        raise ValueError("S3_EPISODE_URI must be an s3://bucket/prefix URI")
    return uri


def is_episode_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("episode_") and (path / "session_meta.json").is_file()


def newest_mtime(path: Path) -> float:
    newest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime)
        except FileNotFoundError:
            continue
    return newest


def discover_episodes(recordings_root: Path, min_age_seconds: float) -> list[Path]:
    if not recordings_root.exists():
        raise FileNotFoundError(f"Recordings root does not exist: {recordings_root}")
    if not recordings_root.is_dir():
        raise NotADirectoryError(f"Recordings root is not a directory: {recordings_root}")

    cutoff = time.time() - min_age_seconds
    episodes: list[Path] = []
    for path in sorted(recordings_root.glob("**/episode_*")):
        if not is_episode_dir(path):
            continue
        if newest_mtime(path) > cutoff:
            continue
        episodes.append(path)
    return episodes


def s3_target_for_episode(s3_uri: str, recordings_root: Path, episode_dir: Path) -> str:
    rel = episode_dir.relative_to(recordings_root).as_posix()
    return f"{s3_uri}/{rel}/"


def require_env(env: dict[str, str], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if not env.get(key)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required AWS environment variable(s): {joined}")


def run_sync(episode_dir: Path, target: str, env: dict[str, str], dry_run: bool) -> None:
    cmd = [
        "aws",
        "s3",
        "sync",
        str(episode_dir),
        target,
        "--only-show-errors",
        "--no-progress",
    ]
    if dry_run:
        cmd.append("--dryrun")
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync completed episode recordings to S3.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="dotenv file containing AWS credentials and S3_EPISODE_URI",
    )
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=None,
        help="local root containing day folders and episode_* directories",
    )
    parser.add_argument(
        "--s3-uri",
        default=None,
        help="destination S3 URI, overriding S3_EPISODE_URI from .env",
    )
    parser.add_argument(
        "--min-age-seconds",
        type=float,
        default=None,
        help="skip episodes with files modified more recently than this",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print AWS CLI sync actions without uploading",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        dotenv = load_dotenv(args.env_file)
        env = merged_env(dotenv)

        recordings_root = args.recordings_root or Path(env.get("RECORDINGS_ROOT", "recordings"))
        recordings_root = recordings_root.expanduser().resolve()
        s3_uri = normalize_s3_uri(args.s3_uri or env.get("S3_EPISODE_URI", ""))
        min_age_seconds = args.min_age_seconds
        if min_age_seconds is None:
            min_age_seconds = float(env.get("EPISODE_MIN_AGE_SECONDS", "60"))

        require_env(env, REQUIRED_AWS_ENV)
        episodes = discover_episodes(recordings_root, min_age_seconds)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not episodes:
        print(f"No complete episodes older than {min_age_seconds:g}s found under {recordings_root}.")
        return 0

    print(f"Syncing {len(episodes)} episode(s) from {recordings_root} to {s3_uri}")
    if args.dry_run:
        print("Dry run only; no files will be uploaded.")

    for episode_dir in episodes:
        target = s3_target_for_episode(s3_uri, recordings_root, episode_dir)
        print(f"{episode_dir.relative_to(recordings_root)} -> {target}")
        try:
            run_sync(episode_dir, target, env, args.dry_run)
        except FileNotFoundError:
            print("error: aws CLI not found. Install/configure AWS CLI v2 and try again.", file=sys.stderr)
            return 127
        except subprocess.CalledProcessError as exc:
            print(f"error: aws s3 sync failed for {episode_dir} with exit code {exc.returncode}", file=sys.stderr)
            return exc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
