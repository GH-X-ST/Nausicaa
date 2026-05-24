from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_post_w3_cluster_merge(*, input_root: Path) -> dict[str, object]:
    """Block compact-library merging until W3-surviving variants exist."""

    return {
        "status": "blocked",
        "input_root": Path(input_root).as_posix(),
        "blocked_reason": "post_W3_clustering_requires_W3_surviving_variants",
        "allowed_before_w3_survivors": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blocked post-W3 cluster/merge scaffold.")
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_post_w3_cluster_merge(input_root=args.input_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
