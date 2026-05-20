from __future__ import annotations

from pathlib import Path

import aggregate_w0_dense_archive as aggregate
from test_w0_dense_archive_aggregation import _write_fake_chunks


def test_audit_package_is_compact_and_excludes_full_partitions(tmp_path: Path) -> None:
    result_root = tmp_path / "11_w0_dense_archive"
    _write_fake_chunks(result_root, run_id=13)

    paths = aggregate.aggregate_w0_dense_archive(
        run_id=13,
        planning_run_id=12,
        result_root=result_root,
        expected_trials_total=8,
        expected_trials_per_branch=4,
        storage_format="csv_gz",
        archive_scale_mode="strict",
        build_upload_package=True,
    )

    package = paths["upload_package_dir"]
    assert package.exists()
    assert not any("tables" in path.parts for path in package.rglob("*") if path.is_file())
    assert all(
        path.stat().st_size < aggregate.UPLOAD_PACKAGE_MAX_BYTES
        for path in package.rglob("*")
        if path.is_file()
    )
