from __future__ import annotations

import os
from pathlib import Path


def _detect_providers() -> tuple[bool, bool, Path | None]:
    """Detect available providers and return (cuda, cpu, lib_path)."""
    cuda_available = False
    cpu_available = False
    lib_path = None

    try:
        from polars_fastembed_cuda import get_ort_lib_path

        path = get_ort_lib_path()
        if path and path.exists():
            cuda_available = True
            lib_path = path
    except ImportError:
        pass

    try:
        from polars_fastembed_cpu import get_ort_lib_path

        path = get_ort_lib_path()
        if path and path.exists():
            cpu_available = True
            if lib_path is None:
                lib_path = path
    except ImportError:
        pass

    return cuda_available, cpu_available, lib_path


# Run detection at import time
_CUDA_AVAILABLE, _CPU_AVAILABLE, _ORT_LIB_PATH = _detect_providers()


def configure_ort() -> None:
    if "ORT_DYLIB_PATH" in os.environ:
        return

    if _ORT_LIB_PATH is None:
        raise ImportError(
            "No ONNX Runtime provider found.\n\n"
            "Install one of:\n"
            "  pip install polars-fastembed[cpu]\n"
            "  pip install polars-fastembed[cuda]\n",
        )

    os.environ["ORT_DYLIB_PATH"] = str(_ORT_LIB_PATH)
    libs_dir = str(_ORT_LIB_PATH.parent)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{libs_dir}:{existing}" if existing else libs_dir
