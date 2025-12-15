from __future__ import annotations

import os
from pathlib import Path


def _find_ort_library() -> Path | None:
    try:
        from polars_fastembed_cuda import get_ort_lib_path

        path = get_ort_lib_path()
        if path and path.exists():
            return path
    except ImportError:
        pass

    try:
        from polars_fastembed_cpu import get_ort_lib_path

        path = get_ort_lib_path()
        if path and path.exists():
            return path
    except ImportError:
        pass

    return None


def configure_ort() -> None:
    if "ORT_DYLIB_PATH" in os.environ:
        return

    ort_lib = _find_ort_library()

    if ort_lib is None:
        raise ImportError(
            "No ONNX Runtime provider found.\n\n"
            "Install one of:\n"
            "  pip install polars-fastembed[cpu]\n"
            "  pip install polars-fastembed[cuda]\n",
        )

    os.environ["ORT_DYLIB_PATH"] = str(ort_lib)
    libs_dir = str(ort_lib.parent)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{libs_dir}:{existing}" if existing else libs_dir
