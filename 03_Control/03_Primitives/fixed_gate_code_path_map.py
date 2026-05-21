from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class CodePathRecord:
    code_path: str
    implementation_status: str
    fixed_gate_role: str
    action: str
    default_claim_use: str
    note: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


ACTIVE_CODE_PATH = (
    "fixed launch gate",
    "primitive rollout archive over launch-gate and reachable downstream states",
    "W0/W1 fixed-gate archive",
    "W2/W3 focused replay",
    "primitive-envelope clustering",
    "governor candidate package",
    "repeated-launch episode simulation",
    "real-flight ingest and matched replay",
)

CODE_PATH_RECORDS = (
    CodePathRecord(
        "state_contract.py, command_contract.py",
        "existing",
        "canonical contracts",
        "reuse_directly",
        "all fixed-gate evidence",
        "Preserve state order and command order.",
    ),
    CodePathRecord(
        "dense_archive_table_io.py and chunked archive runners",
        "existing",
        "storage/runtime backend",
        "reuse_directly",
        "simulation-only until matched with real data",
        "Keep compressed partitions, checksums, and branch-local manifests.",
    ),
    CodePathRecord(
        "wing_wind_descriptors.py",
        "existing",
        "wing-scale wind evidence",
        "reuse_directly",
        "W1/W2/W3 simulation evidence",
        "Keep centre, wing mean, left/right, and spanwise descriptors.",
    ),
    CodePathRecord(
        "latency.py",
        "existing",
        "timing semantics",
        "reuse_directly",
        "latency-labelled simulation evidence",
        "State delay, command delay, and actuator lag remain distinct.",
    ),
    CodePathRecord(
        "primitive_interface.py and rollout.py",
        "existing",
        "primitive execution backend",
        "wrap_for_fixed_gate_rows",
        "simulation-only primitive rollout evidence",
        "Use fixed-gate/reachable entry states instead of broad all-arena starts.",
    ),
    CodePathRecord(
        "paired W0/W1 planning and archive runners",
        "existing",
        "W0/W1 fixed-gate archive backend",
        "adapt_with_fixed_gate_inputs",
        "simulation-only W0/W1 evidence",
        "W1 remains independent of W0 success and branches stay separate.",
    ),
    CodePathRecord(
        "primitive_library_governor.py",
        "existing",
        "online accept/reject authority",
        "reuse_and_extend_rejection_labels",
        "candidate safety evidence only",
        "Clustering proposes packages; the governor still filters execution.",
    ),
    CodePathRecord(
        "dense_start_state_sampling.py broad start classes",
        "existing",
        "old dense-sweep source",
        "demote_to_diagnostic",
        "diagnostic only",
        "Do not feed final fixed-gate transfer claims by default.",
    ),
    CodePathRecord(
        "all-yaw and all-arena sweeps",
        "existing/implicit",
        "old exploration source",
        "demote_to_diagnostic",
        "diagnostic only",
        "Allowed only as labelled appendix/boundary evidence.",
    ),
    CodePathRecord(
        "same-flight recapture logic",
        "old objective",
        "not active mission path",
        "demote_to_deprecated",
        "not claim-supporting",
        "Repeated launch replaces same-flight recapture as the active mission.",
    ),
    CodePathRecord(
        "agile high-angle expansion paths",
        "existing",
        "boundary evidence",
        "demote_to_boundary_only",
        "diagnostic boundary evidence",
        "Not required for final mission unless explicitly requested later.",
    ),
)


def active_code_path_text() -> str:
    """Return the active mission path as a compact arrow-separated string."""

    return " -> ".join(ACTIVE_CODE_PATH)


def code_path_map_frame() -> pd.DataFrame:
    """Return the active/deprecated code-path map for reports and manifests."""

    return pd.DataFrame([record.as_dict() for record in CODE_PATH_RECORDS])


def deprecated_default_paths() -> tuple[str, ...]:
    """Return paths that must not feed final claims without explicit promotion."""

    frame = code_path_map_frame()
    demoted = frame[frame["action"].astype(str).str.startswith("demote")]
    return tuple(str(item) for item in demoted["code_path"])
