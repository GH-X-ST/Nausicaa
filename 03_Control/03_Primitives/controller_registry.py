from __future__ import annotations

from lqr_controller import LQRController, controller_is_executable_lqr


def load_controller_registry(path=None) -> dict[str, LQRController]:
    """Compatibility stub: active W01 code uses primitive_variant_registry."""

    del path
    return {}
