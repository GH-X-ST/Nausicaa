from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggregate_w0_dense_archive import export_diagnostic_slice  # noqa: E402


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) CLI Wrapper
# =============================================================================


# =============================================================================
# 1) CLI Wrapper
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=13)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--layout-branch-id", default=None)
    parser.add_argument("--failure-label", default=None)
    parser.add_argument("--cluster-key", default=None)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    path = export_diagnostic_slice(
        run_id=args.run_id,
        result_root=args.result_root,
        layout_branch_id=args.layout_branch_id,
        failure_label=args.failure_label,
        cluster_key_filter=args.cluster_key,
        max_rows=args.max_rows,
        output_path=args.output_path,
    )
    print(f"w0_diagnostic_slice={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
