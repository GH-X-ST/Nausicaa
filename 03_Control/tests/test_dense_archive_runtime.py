from __future__ import annotations

import dense_archive_runtime as runtime


def test_runtime_manifest_fields_preserve_shared_contract_versions() -> None:
    payload = runtime.runtime_manifest_fields(
        simulation_stage="paired_w0_w1_proof",
        environment_mode="multiple",
        branch_decision_scope="branch_local_only_no_cross_layout_decision_transfer",
    )

    assert payload["runtime_core_version"] == runtime.RUNTIME_CORE_VERSION
    assert payload["storage_contract_version"] == runtime.STORAGE_CONTRACT_VERSION
    assert payload["governor_package_schema_version"] == runtime.GOVERNOR_PACKAGE_SCHEMA_VERSION
    assert "GPU acceleration is deferred" in runtime.GPU_ACCELERATION_ASSESSMENT
