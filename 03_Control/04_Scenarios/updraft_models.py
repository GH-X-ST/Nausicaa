from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


SINGLE_FAN_CENTER_XY = (4.2, 2.4)
FOUR_FAN_CENTERS_XY = (
    (3.0, 3.6),
    (5.4, 3.6),
    (3.0, 1.2),
    (5.4, 1.2),
)


class WindField(Protocol):
    name: str
    source: str

    def __call__(self, points_w_up_m: np.ndarray) -> np.ndarray:
        """Return wind in public world z-up frame, shape (N, 3), m/s."""


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_paths(repo_root: Path, relative_paths: tuple[str, ...]) -> list[Path]:
    return [repo_root / rel for rel in relative_paths]


def _first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    text = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing fitted updraft workbook. Tried:\n{text}")


def _source_text(path: Path, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")


@dataclass(frozen=True)
class GaussianVarWindField:
    name: str
    source: str
    z_axis_m: np.ndarray
    interpolators: dict[str, PchipInterpolator]
    fan_centers_xy: tuple[tuple[float, float], ...]
    fan_ids: tuple[str, ...]

    def __call__(self, points_w_up_m: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_w_up_m, dtype=float).reshape(-1, 3)
        z = np.clip(pts[:, 2], self.z_axis_m[0], self.z_axis_m[-1])
        w_up = np.zeros(pts.shape[0], dtype=float)
        for fan_id, (cx, cy) in zip(self.fan_ids, self.fan_centers_xy):
            suffix = "" if fan_id == "" else f"_{fan_id}"
            a = np.asarray(self.interpolators[f"A{suffix}"](z), dtype=float)
            delta = np.maximum(
                np.asarray(self.interpolators[f"delta{suffix}"](z), dtype=float),
                1e-12,
            )
            w0 = np.asarray(self.interpolators[f"w0{suffix}"](z), dtype=float)
            r = np.hypot(pts[:, 0] - float(cx), pts[:, 1] - float(cy))
            w_up += w0 + a * np.exp(-((r / delta) ** 2))
        return np.column_stack([np.zeros_like(w_up), np.zeros_like(w_up), w_up])


@dataclass(frozen=True)
class AnnularGPGridWindField:
    name: str
    source: str
    z_axis_m: np.ndarray
    x_axis_m: np.ndarray
    y_axis_m: np.ndarray
    mean_grids: tuple[np.ndarray, ...]

    def __call__(self, points_w_up_m: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_w_up_m, dtype=float).reshape(-1, 3)
        zq = np.clip(pts[:, 2], self.z_axis_m[0], self.z_axis_m[-1])
        plane_values = np.vstack(
            [
                _bilinear_grid(
                    grid=grid,
                    x_axis=self.x_axis_m,
                    y_axis=self.y_axis_m,
                    xq=pts[:, 0],
                    yq=pts[:, 1],
                )
                for grid in self.mean_grids
            ]
        )
        w_up = np.empty(pts.shape[0], dtype=float)
        for idx, z_val in enumerate(zq):
            w_up[idx] = np.interp(z_val, self.z_axis_m, plane_values[:, idx])
        return np.column_stack([np.zeros_like(w_up), np.zeros_like(w_up), w_up])


@dataclass(frozen=True)
class AnalyticDebugProxy:
    name: str = "analytic_debug_proxy"
    source: str = "deterministic analytic debug proxy; not measured"

    def __call__(self, points_w_up_m: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_w_up_m, dtype=float).reshape(-1, 3)
        dx = pts[:, 0] - SINGLE_FAN_CENTER_XY[0]
        dy = pts[:, 1] - SINGLE_FAN_CENTER_XY[1]
        radial = np.exp(-(dx * dx + dy * dy) / (2.0 * 1.0**2))
        vertical = np.exp(-((pts[:, 2] - 1.6) ** 2) / (2.0 * 1.2**2))
        w_up = 0.25 * radial * vertical
        return np.column_stack([np.zeros_like(w_up), np.zeros_like(w_up), w_up])


def load_updraft_model(name: str, repo_root: Path | None = None) -> WindField:
    root = repo_root or _repo_root_from_here()
    if name == "single_gaussian_var":
        path = _first_existing(
            _candidate_paths(
                root,
                (
                    "01_Thermal/B_results/single_var_params.xlsx",
                    "B_results/single_var_params.xlsx",
                ),
            )
        )
        return _load_gaussian(path, root, name, "single_var", (SINGLE_FAN_CENTER_XY,), ("",))
    if name == "four_gaussian_var":
        path = _first_existing(
            _candidate_paths(
                root,
                (
                    "01_Thermal/B_results/four_var_params.xlsx",
                    "B_results/four_var_params.xlsx",
                ),
            )
        )
        return _load_four_gaussian(path, root)
    if name == "single_annular_gp_grid":
        path = _first_existing(
            _candidate_paths(
                root,
                (
                    "01_Thermal/B_results/Single_Fan_Annular_GP/single_annular_gp_grid_predictions.xlsx",
                    "B_results/Single_Fan_Annular_GP/single_annular_gp_grid_predictions.xlsx",
                ),
            )
        )
        return _load_annular_grid(path, root, name)
    if name == "four_annular_gp_grid":
        path = _first_existing(
            _candidate_paths(
                root,
                (
                    "01_Thermal/B_results/Four_Fan_Annular_GP/four_annular_gp_grid_predictions.xlsx",
                    "B_results/Four_Fan_Annular_GP/four_annular_gp_grid_predictions.xlsx",
                ),
            )
        )
        return _load_annular_grid(path, root, name)
    if name == "analytic_debug_proxy":
        return AnalyticDebugProxy()
    raise ValueError(f"Unknown updraft model '{name}'.")


def _load_gaussian(
    path: Path,
    repo_root: Path,
    name: str,
    sheet_name: str,
    fan_centers_xy: tuple[tuple[float, float], ...],
    fan_ids: tuple[str, ...],
) -> GaussianVarWindField:
    df = pd.read_excel(path, sheet_name=sheet_name)
    z_axis = df["z_m"].to_numpy(dtype=float)
    interpolators = {}
    for col in df.columns:
        if col == "z_m":
            continue
        interpolators[str(col)] = PchipInterpolator(z_axis, df[col].to_numpy(dtype=float))
    return GaussianVarWindField(
        name=name,
        source=_source_text(path, repo_root),
        z_axis_m=z_axis,
        interpolators=interpolators,
        fan_centers_xy=fan_centers_xy,
        fan_ids=fan_ids,
    )


def _load_four_gaussian(path: Path, repo_root: Path) -> GaussianVarWindField:
    df = pd.read_excel(path, sheet_name="four_var")
    z_axis = df["z_m"].to_numpy(dtype=float)
    fan_ids = ("F01", "F02", "F03", "F04")
    interpolators = {}
    for fan_id in fan_ids:
        for base in ("A", "delta", "w0"):
            col = f"{base}_{fan_id}"
            if col not in df.columns:
                raise ValueError(f"Missing four-fan Gaussian column: {col}")
            interpolators[col] = PchipInterpolator(z_axis, df[col].to_numpy(dtype=float))
    return GaussianVarWindField(
        name="four_gaussian_var",
        source=_source_text(path, repo_root),
        z_axis_m=z_axis,
        interpolators=interpolators,
        fan_centers_xy=FOUR_FAN_CENTERS_XY,
        fan_ids=fan_ids,
    )


def _load_annular_grid(path: Path, repo_root: Path, name: str) -> AnnularGPGridWindField:
    xls = pd.ExcelFile(path)
    mean_sheets = sorted(
        sheet for sheet in xls.sheet_names if sheet.endswith("_annular_gp_mean")
    )
    if not mean_sheets:
        raise ValueError(f"No annular-GP mean sheets found in {path}")
    z_values = []
    grids = []
    x_axis = None
    y_axis = None
    for sheet in mean_sheets:
        z_values.append(_sheet_height_m(sheet))
        df = pd.read_excel(path, sheet_name=sheet)
        xs = np.asarray(df.columns[1:], dtype=float)
        ys = df.iloc[:, 0].to_numpy(dtype=float)
        grid = df.iloc[:, 1:].to_numpy(dtype=float)
        if x_axis is None:
            x_axis = xs
            y_axis = ys
        grids.append(grid)
    order = np.argsort(z_values)
    return AnnularGPGridWindField(
        name=name,
        source=_source_text(path, repo_root),
        z_axis_m=np.asarray(z_values, dtype=float)[order],
        x_axis_m=np.asarray(x_axis, dtype=float),
        y_axis_m=np.asarray(y_axis, dtype=float),
        mean_grids=tuple(grids[idx] for idx in order),
    )


def _sheet_height_m(sheet_name: str) -> float:
    code = sheet_name.split("_", 1)[0]
    if not (code.startswith("z") and code[1:].isdigit()):
        raise ValueError(f"Invalid height sheet name: {sheet_name}")
    return int(code[1:]) / 100.0


def _bilinear_grid(
    grid: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    x = np.clip(np.asarray(xq, dtype=float), x_axis[0], x_axis[-1])
    y = np.clip(np.asarray(yq, dtype=float), y_axis[0], y_axis[-1])
    ix = np.clip(np.searchsorted(x_axis, x, side="right") - 1, 0, x_axis.size - 2)
    iy = np.clip(np.searchsorted(y_axis, y, side="right") - 1, 0, y_axis.size - 2)
    x0 = x_axis[ix]
    x1 = x_axis[ix + 1]
    y0 = y_axis[iy]
    y1 = y_axis[iy + 1]
    tx = (x - x0) / np.maximum(x1 - x0, 1e-12)
    ty = (y - y0) / np.maximum(y1 - y0, 1e-12)
    f00 = grid[iy, ix]
    f10 = grid[iy, ix + 1]
    f01 = grid[iy + 1, ix]
    f11 = grid[iy + 1, ix + 1]
    return (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )
