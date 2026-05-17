from __future__ import annotations

import re
from pathlib import Path


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Path validation constants
# 2) Result tree creation
# =============================================================================


# =============================================================================
# 1) Path Validation Constants
# =============================================================================
# Campaigns are path segments, not free-form labels, to keep generated evidence
# deterministic and shell-safe.
SAFE_CAMPAIGN_PATTERN = re.compile(r"^[a-z0-9_]+$")
RESULT_SUBDIRECTORIES = ("metrics", "logs", "figures", "manifests", "reports")


# =============================================================================
# 2) Result Tree Creation
# =============================================================================
def make_result_tree(
    root: Path,
    campaign: str,
    run_id: int,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Create and return metrics, logs, figures, manifests, and reports paths."""

    if not SAFE_CAMPAIGN_PATTERN.fullmatch(campaign):
        raise ValueError("campaign must contain only lowercase letters, digits, and underscores.")
    if not isinstance(run_id, int) or run_id < 0:
        raise ValueError("run_id must be a nonnegative integer.")

    run_root = Path(root) / campaign / f"{run_id:03d}"
    if run_root.exists() and not overwrite:
        raise ValueError(f"result tree already exists: {run_root}")

    paths = {"root": run_root}
    for name in RESULT_SUBDIRECTORIES:
        paths[name] = run_root / name
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
