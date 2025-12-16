from pathlib import Path


def get_ort_lib_path() -> Path | None:
    """Return the path to the bundled ORT library, or None if not found."""
    libs_dir = Path(__file__).parent / "libs"

    # Platform-specific library names
    for name in ["libonnxruntime.so", "libonnxruntime.dylib", "onnxruntime.dll"]:
        lib_path = libs_dir / name
        if lib_path.exists():
            return lib_path

    return None
