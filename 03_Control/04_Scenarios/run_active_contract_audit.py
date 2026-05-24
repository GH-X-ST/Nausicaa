from __future__ import annotations

from run_w01_w2_w3_contract_audit import (
    AuditFinding,
    main,
    run_w01_w2_w3_contract_audit,
)


def run_active_contract_audit(repo_root=".") -> list[AuditFinding]:
    return run_w01_w2_w3_contract_audit(repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
