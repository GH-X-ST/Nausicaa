from __future__ import annotations

import subprocess


def get_git_version() -> str:
    """Return a short git description string (tag/commit), or 'unknown' if not in a repo."""
    try:
        desc = subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--tags"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return desc
    except Exception:
        return "unknown"