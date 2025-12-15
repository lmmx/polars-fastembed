def test_cuda_provider_lib_exists():
    from polars_fastembed_cuda import get_ort_lib_path

    path = get_ort_lib_path()
    assert path is not None, "get_ort_lib_path() returned None"
    assert path.exists(), f"ORT lib does not exist at {path}"
    assert path.stat().st_size > 1_000_000, (
        f"ORT lib seems too small: {path.stat().st_size} bytes"
    )


def test_cuda_provider_lib_is_valid():
    from polars_fastembed_cuda import get_ort_lib_path

    path = get_ort_lib_path()
    assert path is not None

    with open(path, "rb") as f:
        magic = f.read(4)

    valid_magics = [
        b"\x7fELF",
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
        b"MZ",
    ]

    assert any(magic.startswith(m) for m in valid_magics), (
        f"Invalid magic bytes: {magic.hex()}"
    )
