from __future__ import annotations

from run_lqr_w01_dense_chunked import (
    W01DenseRunConfig,
    main,
    run_lqr_w01_dense_chunked,
)


LQRTuningSweepConfig = W01DenseRunConfig


def run_lqr_tuning_sweep(config: W01DenseRunConfig) -> dict[str, object]:
    """Compatibility entrypoint for the active W01 dense runner."""

    return run_lqr_w01_dense_chunked(config)


if __name__ == "__main__":
    raise SystemExit(main())
