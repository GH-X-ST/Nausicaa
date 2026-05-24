from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_primitive_selector_report(*, input_root: Path) -> dict[str, object]:
    """Block compact primitive-library reporting until W3-surviving variants exist."""

    return {
        "status": "blocked",
        "input_root": Path(input_root).as_posix(),
        "blocked_reason": "compact_primitive_library_requires_W3_surviving_variants",
        "claim_status": "simulation_only_blocked",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blocked compact primitive-library report scaffold.")
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_primitive_selector_report(input_root=args.input_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
