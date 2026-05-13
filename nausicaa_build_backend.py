from __future__ import annotations

import base64
import hashlib
import zipfile
from pathlib import Path


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Build Metadata
# 2) PEP 517 Hooks
# 3) Wheel Assembly Helpers
# =============================================================================

# =============================================================================
# 1) Build Metadata
# =============================================================================
NAME = "nausicaa"
VERSION = "0.1.1"
DIST_INFO = f"{NAME}-{VERSION}.dist-info"
WHEEL_NAME = f"{NAME}-{VERSION}-py3-none-any.whl"
REQUIRES_DIST = (
    "aerosandbox>=4.2,<5",
    "numpy>=1.24",
    "scipy>=1.10",
    "pandas>=2.0",
    "matplotlib>=3.7",
    "casadi>=3.6",
    "openpyxl>=3.1",
)


# =============================================================================
# 2) PEP 517 Hooks
# =============================================================================
def get_requires_for_build_wheel(config_settings: object | None = None) -> list[str]:
    del config_settings
    return []


def get_requires_for_build_editable(config_settings: object | None = None) -> list[str]:
    del config_settings
    return []


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: object | None = None,
) -> str:
    del config_settings
    return _write_metadata_dir(Path(metadata_directory))


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: object | None = None,
) -> str:
    del config_settings
    return _write_metadata_dir(Path(metadata_directory))


def build_wheel(
    wheel_directory: str,
    config_settings: object | None = None,
    metadata_directory: str | None = None,
) -> str:
    del config_settings, metadata_directory
    return _write_wheel(Path(wheel_directory), editable=False)


def build_editable(
    wheel_directory: str,
    config_settings: object | None = None,
    metadata_directory: str | None = None,
) -> str:
    del config_settings, metadata_directory
    return _write_wheel(Path(wheel_directory), editable=True)


# =============================================================================
# 3) Wheel Assembly Helpers
# =============================================================================
def _metadata_text() -> str:
    requires = "".join(f"Requires-Dist: {item}\n" for item in REQUIRES_DIST)
    return (
        "Metadata-Version: 2.1\n"
        f"Name: {NAME}\n"
        f"Version: {VERSION}\n"
        "Summary: Primitive-level control and simulation evidence for the Nausicaa indoor glider.\n"
        "Requires-Python: >=3.10\n"
        f"{requires}"
    )


def _wheel_text() -> str:
    return (
        "Wheel-Version: 1.0\n"
        "Generator: nausicaa-build-backend\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    )


def _write_metadata_dir(metadata_root: Path) -> str:
    dist_info = metadata_root / DIST_INFO
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_metadata_text(), encoding="utf-8")
    (dist_info / "WHEEL").write_text(_wheel_text(), encoding="utf-8")
    return DIST_INFO


def _editable_pth_text() -> str:
    repo_root = Path(__file__).resolve().parent
    paths = (
        repo_root / "03_Control" / "02_Inner_Loop",
        repo_root / "03_Control" / "03_Primitives",
        repo_root / "03_Control" / "04_Scenarios",
        repo_root / "03_Control" / "05_Results",
    )
    # Editable installs expose script-style research folders without renaming them.
    return "".join(f"{path}\n" for path in paths)


def _write_wheel(wheel_root: Path, editable: bool) -> str:
    wheel_root.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_root / WHEEL_NAME
    entries = {
        f"{DIST_INFO}/METADATA": _metadata_text().encode("utf-8"),
        f"{DIST_INFO}/WHEEL": _wheel_text().encode("utf-8"),
    }
    if editable:
        entries[f"{NAME}_editable.pth"] = _editable_pth_text().encode("utf-8")

    record_rows: list[str] = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for archive_name, payload in entries.items():
            wheel.writestr(archive_name, payload)
            record_rows.append(_record_row(archive_name, payload))
        record_name = f"{DIST_INFO}/RECORD"
        record_rows.append(f"{record_name},,\n")
        wheel.writestr(record_name, "".join(record_rows).encode("utf-8"))
    return wheel_path.name


def _record_row(archive_name: str, payload: bytes) -> str:
    digest = hashlib.sha256(payload).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"{archive_name},sha256={encoded},{len(payload)}\n"
