from __future__ import annotations

from pathlib import Path

from run_active_contract_audit import run_active_contract_audit


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_active_contract_audit_has_no_findings() -> None:
    assert run_active_contract_audit(REPO_ROOT) == []
