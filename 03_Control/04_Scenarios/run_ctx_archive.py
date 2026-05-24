from __future__ import annotations

from run_lqr_w01_dense_chunked import (
    W01DenseRunConfig,
    main,
    run_lqr_w01_dense_chunked,
)


ContextArchiveConfig = W01DenseRunConfig


def run_context_archive(config: W01DenseRunConfig) -> dict[str, object]:
    """Compatibility entrypoint for the active W01 primitive variant workflow."""

    return run_lqr_w01_dense_chunked(config)


if __name__ == "__main__":
    raise SystemExit(main())
