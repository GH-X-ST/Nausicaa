"""retired_not_active: old contextual dense runner and result root owner.

The archived dense results now live under
`03_Control/99_Archive/retired_pd_contextual_v1_4/results/` and are not active
evidence. Use the active LQR runners for any new thesis evidence.
"""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "retired_not_active: use run_lqr_tuning_sweep.py and related LQR runners."
    )


if __name__ == "__main__":
    main()

