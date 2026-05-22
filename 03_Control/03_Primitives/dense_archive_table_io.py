from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Data Containers and Storage Detection
# 2) Partition Read/Write Helpers
# 3) Manifest Helpers
# 4) Conversion Helpers
# =============================================================================


# =============================================================================
# 1) Data Containers and Storage Detection
# =============================================================================
StorageFormat = Literal["auto", "parquet", "csv_gz", "csv"]
_STORAGE_FORMATS = frozenset({"auto", "parquet", "csv_gz", "csv"})


@dataclass(frozen=True)
class TablePartition:
    table_name: str
    relative_path: str
    storage_format: str
    row_count: int
    byte_count: int
    columns: tuple[str, ...]
    checksum_sha256: str


@dataclass(frozen=True)
class TableManifest:
    run_id: int
    root: str
    storage_format: str
    tables: tuple[TablePartition, ...]


def parquet_supported() -> bool:
    """Return whether an optional parquet engine is importable."""

    return (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )


def resolve_storage_format(requested: str = "auto") -> str:
    """Resolve the requested contextual dense-run storage format."""

    value = str(requested).strip().lower()
    if value not in _STORAGE_FORMATS:
        raise ValueError("storage_format must be one of: auto, parquet, csv_gz, csv.")
    if value == "auto":
        return "parquet" if parquet_supported() else "csv_gz"
    if value == "parquet" and not parquet_supported():
        raise RuntimeError(
            "storage_format='parquet' requires pyarrow or fastparquet; "
            "use storage_format='auto' or 'csv_gz' on this environment."
        )
    return value


def table_extension(storage_format: str) -> str:
    """Return the filename extension for a resolved storage format."""

    value = resolve_storage_format(storage_format)
    if value == "parquet":
        return "parquet"
    if value == "csv_gz":
        return "csv.gz"
    if value == "csv":
        return "csv"
    raise ValueError(f"unknown resolved storage format: {value!r}")


def filesystem_path(path: Path) -> Path:
    """Return a path suitable for direct filesystem calls."""

    return _filesystem_path(Path(path))


def ensure_directory(path: Path) -> None:
    """Create a directory, including long Windows paths when needed."""

    _mkdir(Path(path))


# =============================================================================
# 2) Partition Read/Write Helpers
# =============================================================================
def write_table_partition(
    frame: pd.DataFrame,
    path: Path,
    *,
    storage_format: str,
    compression_level: int = 1,
) -> TablePartition:
    """Write one deterministic table partition and return its manifest row."""

    resolved_format = resolve_storage_format(storage_format)
    _validate_compression_level(compression_level)
    output_path = Path(path)
    _mkdir(output_path.parent)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    tmp_fs_path = _filesystem_path(tmp_path)
    output_fs_path = _filesystem_path(output_path)
    if tmp_fs_path.exists():
        tmp_fs_path.unlink()

    try:
        if resolved_format == "parquet":
            frame.to_parquet(tmp_fs_path, index=False)
        elif resolved_format == "csv_gz":
            _write_csv_gz(frame, tmp_path, compression_level=int(compression_level))
        elif resolved_format == "csv":
            frame.to_csv(tmp_fs_path, index=False)
        else:
            raise ValueError(f"unsupported storage format: {resolved_format!r}")
    except Exception:
        if tmp_fs_path.exists():
            tmp_fs_path.unlink()
        raise

    tmp_fs_path.replace(output_fs_path)
    stat = output_fs_path.stat()
    return TablePartition(
        table_name=_table_name_from_path(output_path),
        relative_path=_relative_path_from_tables(output_path),
        storage_format=resolved_format,
        row_count=int(len(frame)),
        byte_count=int(stat.st_size),
        columns=tuple(str(column) for column in frame.columns),
        checksum_sha256=file_sha256(output_path),
    )


def read_table_partition(
    path: Path,
    *,
    storage_format: str | None = None,
) -> pd.DataFrame:
    """Read one table partition, inferring the format from the suffix when needed."""

    input_path = Path(path)
    resolved_format = (
        _format_from_path(input_path)
        if storage_format is None
        else resolve_storage_format(storage_format)
    )
    if resolved_format == "parquet":
        return pd.read_parquet(_filesystem_path(input_path))
    if resolved_format == "csv_gz":
        return pd.read_csv(_filesystem_path(input_path), compression="gzip")
    if resolved_format == "csv":
        return pd.read_csv(_filesystem_path(input_path))
    raise ValueError(f"unsupported storage format: {resolved_format!r}")


def list_table_partitions(root: Path, table_name: str) -> list[Path]:
    """List partition files for one contextual table under a run or tables root."""

    root_path = Path(root)
    candidates = (
        root_path / str(table_name),
        root_path / "tables" / str(table_name),
    )
    table_root = next(
        (path for path in candidates if filesystem_path(path).exists()),
        candidates[0],
    )
    if not filesystem_path(table_root).exists():
        return []
    paths: list[Path] = []
    for dirpath, _, filenames in os.walk(filesystem_path(table_root)):
        for filename in filenames:
            path = _normal_path_from_filesystem(Path(dirpath) / filename)
            if path.name.endswith(".tmp"):
                continue
            if _is_table_partition_path(path):
                paths.append(path)
    return sorted(paths, key=lambda item: item.as_posix())


def file_sha256(path: Path) -> str:
    """Return a SHA256 checksum for a written partition."""

    digest = hashlib.sha256()
    with _filesystem_path(Path(path)).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partition_row_count(path: Path, *, storage_format: str | None = None) -> int:
    """Return a partition row count using the normal table reader."""

    return int(len(read_table_partition(path, storage_format=storage_format)))


def _validate_compression_level(compression_level: int) -> None:
    level = int(compression_level)
    if level < 0 or level > 9:
        raise ValueError("compression_level must be in [0, 9].")


def _write_csv_gz(
    frame: pd.DataFrame,
    tmp_path: Path,
    *,
    compression_level: int,
) -> None:
    # Writing the pandas gzip stream to an in-memory handle avoids embedding
    # the filesystem path in the gzip header while preserving mtime=0.
    deterministic = {
        "method": "gzip",
        "compresslevel": int(compression_level),
        "mtime": 0,
    }
    try:
        buffer = io.BytesIO()
        frame.to_csv(buffer, index=False, compression=deterministic)
        _filesystem_path(Path(tmp_path)).write_bytes(buffer.getvalue())
    except (TypeError, ValueError):
        frame.to_csv(
            _filesystem_path(Path(tmp_path)),
            index=False,
            compression="gzip",
        )


# =============================================================================
# 3) Manifest Helpers
# =============================================================================
def write_table_manifest(path: Path, manifest: TableManifest) -> None:
    """Write a table manifest as deterministic JSON."""

    payload = asdict(manifest)
    payload["tables"] = [asdict(partition) for partition in manifest.tables]
    output_path = Path(path)
    _mkdir(output_path.parent)
    _filesystem_path(output_path).write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="ascii",
    )


def load_table_manifest(path: Path) -> TableManifest:
    """Load a table manifest written by `write_table_manifest`."""

    payload = json.loads(_filesystem_path(Path(path)).read_text(encoding="ascii"))
    return TableManifest(
        run_id=int(payload["run_id"]),
        root=str(payload["root"]),
        storage_format=str(payload["storage_format"]),
        tables=tuple(
            TablePartition(
                table_name=str(row["table_name"]),
                relative_path=str(row["relative_path"]),
                storage_format=str(row["storage_format"]),
                row_count=int(row["row_count"]),
                byte_count=int(row["byte_count"]),
                columns=tuple(str(column) for column in row["columns"]),
                checksum_sha256=str(row["checksum_sha256"]),
            )
            for row in payload.get("tables", [])
        ),
    )


# =============================================================================
# 4) Conversion Helpers
# =============================================================================
def _format_from_path(path: Path) -> str:
    text = Path(path).name.lower()
    if text.endswith(".parquet"):
        return resolve_storage_format("parquet")
    if text.endswith(".csv.gz"):
        return "csv_gz"
    if text.endswith(".csv"):
        return "csv"
    raise ValueError(f"cannot infer table storage format from path: {path}")


def _is_table_partition_path(path: Path) -> bool:
    text = Path(path).name.lower()
    return (
        text.endswith(".parquet")
        or text.endswith(".csv.gz")
        or text.endswith(".csv")
    )


def _table_name_from_path(path: Path) -> str:
    parts = Path(path).parts
    if "tables" in parts:
        index = parts.index("tables")
        if index + 1 < len(parts):
            return str(parts[index + 1])
    return str(Path(path).parent.name)


def _relative_path_from_tables(path: Path) -> str:
    parts = Path(path).parts
    if "tables" in parts:
        index = parts.index("tables")
        return Path(*parts[index + 1 :]).as_posix()
    return Path(path).name


def csv_gz_text(path: Path) -> str:
    """Return gzipped CSV text for small test/debug partitions."""

    with gzip.open(_filesystem_path(Path(path)), "rt", encoding="utf-8") as handle:
        return handle.read()


def _mkdir(path: Path) -> None:
    _filesystem_path(Path(path)).mkdir(parents=True, exist_ok=True)


def _filesystem_path(path: Path) -> Path:
    """Return a path suitable for filesystem calls on long Windows paths."""

    raw = Path(path)
    if os.name != "nt":
        return raw
    text = str(raw)
    if text.startswith("\\\\?\\"):
        return raw
    resolved = raw.resolve()
    resolved_text = str(resolved)
    if resolved_text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + resolved_text.lstrip("\\"))
    return Path("\\\\?\\" + resolved_text)


def _normal_path_from_filesystem(path: Path) -> Path:
    text = str(path)
    if os.name != "nt":
        return Path(text)
    if text.startswith("\\\\?\\UNC\\"):
        return Path("\\\\" + text[len("\\\\?\\UNC\\") :])
    if text.startswith("\\\\?\\"):
        return Path(text[len("\\\\?\\") :])
    return Path(text)
