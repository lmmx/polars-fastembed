def test_cpu_provider_lib_exists():
    from polars_fastembed_cpu import get_ort_lib_path

    path = get_ort_lib_path()
    assert path is not None, "get_ort_lib_path() returned None"
    assert path.exists(), f"ORT lib does not exist at {path}"
    assert path.stat().st_size > 1_000_000, (
        f"ORT lib seems too small: {path.stat().st_size} bytes"
    )


def test_cpu_provider_lib_is_valid():
    from polars_fastembed_cpu import get_ort_lib_path

    path = get_ort_lib_path()
    assert path is not None

    # Check it's a valid shared library by reading magic bytes
    with open(path, "rb") as f:
        magic = f.read(4)

    # ELF magic (Linux), Mach-O magic (macOS), or MZ (Windows DLL)
    valid_magics = [
        b"\x7fELF",  # ELF (Linux)
        b"\xcf\xfa\xed\xfe",  # Mach-O 64-bit (macOS)
        b"\xca\xfe\xba\xbe",  # Mach-O universal
        b"MZ",  # Windows PE/DLL
    ]

    assert any(magic.startswith(m) for m in valid_magics), (
        f"Invalid magic bytes: {magic.hex()}"
    )


def test_cpu_provider_lib_loads():
    import ctypes

    from polars_fastembed_cpu import get_ort_lib_path

    path = get_ort_lib_path()
    ctypes.CDLL(str(path))  # This would fail on corrupted .so
