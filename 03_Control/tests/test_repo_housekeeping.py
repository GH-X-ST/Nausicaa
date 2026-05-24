from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cleanup_manifest_exists_and_records_claim_boundary() -> None:
    manifest = REPO_ROOT / "docs" / "reset" / "w01_w2_w3_alignment_cleanup_manifest.md"
    text = manifest.read_text(encoding="ascii")

    assert manifest.is_file()
    assert "Claim Boundary" in text
    assert "corrected W0/W1 rich primitive-controller dense" in text
    assert "generation readiness" in text


def test_active_results_placeholder_exists() -> None:
    placeholder = REPO_ROOT / "03_Control" / "05_Results" / ".gitkeep"
    assert placeholder.is_file()
