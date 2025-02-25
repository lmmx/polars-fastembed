#!/usr/bin/env python3
import os
from pathlib import Path


def main():
    repo_root = Path("rewrite")
    dirs = [
        "onnx_placeholder/armv7",
        "onnx_placeholder/x86",
        "onnx_placeholder/s390x",
        "onnx_placeholder/ppc64le",
        "onnx_placeholder/x86_win",
    ]

    for d in dirs:
        os.makedirs(repo_root / d, exist_ok=True)

    placeholders = [
        ("onnx_placeholder/armv7/libonnxruntime.so", 1024),
        ("onnx_placeholder/x86/libonnxruntime.so", 1024),
        ("onnx_placeholder/s390x/libonnxruntime.so", 1024),
        ("onnx_placeholder/ppc64le/libonnxruntime.so", 1024),
        ("onnx_placeholder/x86_win/onnxruntime.dll", 1024),
    ]

    for file_path, size in placeholders:
        with open(repo_root / file_path, "wb") as f:
            f.write(b"\0" * size)
        print(f"Created placeholder: {file_path}")


if __name__ == "__main__":
    main()
