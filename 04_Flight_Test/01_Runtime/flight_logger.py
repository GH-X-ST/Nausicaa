from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class FlightLogger:
    def __init__(self, run_root: Path) -> None:
        self.run_root = Path(run_root)
        self.metrics_root = self.run_root / "metrics"
        self.manifest_root = self.run_root / "manifests"
        self.report_root = self.run_root / "reports"
        for path in (self.metrics_root, self.manifest_root, self.report_root):
            path.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, tuple[Any, csv.DictWriter]] = {}

    def write_manifest(self, name: str, payload: dict[str, Any]) -> None:
        path = self.manifest_root / name
        path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="ascii")

    def append_metric_row(self, name: str, row: dict[str, Any]) -> None:
        row = {key: _scalar(value) for key, value in row.items()}
        if name not in self._writers:
            path = self.metrics_root / name
            handle = path.open("w", newline="", encoding="ascii")
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            self._writers[name] = (handle, writer)
        handle, writer = self._writers[name]
        writer.writerow(row)
        handle.flush()

    def write_report(self, name: str, lines: list[str]) -> None:
        (self.report_root / name).write_text("\n".join(lines) + "\n", encoding="ascii")

    def close(self) -> None:
        for handle, _ in self._writers.values():
            handle.close()
        self._writers.clear()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return value.hex()
    return json.dumps(_json_ready(value), separators=(",", ":"))
