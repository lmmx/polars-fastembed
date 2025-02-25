#!/usr/bin/env python3
"""
CI script to create placeholder ONNX Runtime libraries for build time.

This script creates empty placeholder files for ONNX Runtime libraries
to allow the build to proceed in CI. At runtime, users will need to
actually have ONNX Runtime installed or set ORT_DYLIB_PATH.
"""

import os
from pathlib import Path


def main():
    # Create directories for placeholder libraries
    script_dir = Path(__file__).parent.parent  # Assuming this script is in scripts/
    repo_root = script_dir / "rewrite"  # Adjust if your path structure is different

    # Architecture-specific directories
    arch_dirs = [
        "onnx_placeholder/armv7",
        "onnx_placeholder/x86",
        "onnx_placeholder/s390x",
        "onnx_placeholder/ppc64le",
        "onnx_placeholder/x86_win",
    ]

    # Create directory structure
    for arch_dir in arch_dirs:
        os.makedirs(repo_root / arch_dir, exist_ok=True)

    # Create placeholder files
    placeholders = [
        ("onnx_placeholder/armv7/libonnxruntime.so", 1024),
        ("onnx_placeholder/x86/libonnxruntime.so", 1024),
        ("onnx_placeholder/s390x/libonnxruntime.so", 1024),
        ("onnx_placeholder/ppc64le/libonnxruntime.so", 1024),
        ("onnx_placeholder/x86_win/onnxruntime.dll", 1024),
    ]

    for file_path, size in placeholders:
        with open(repo_root / file_path, "wb") as f:
            f.write(b"\0" * size)  # Just create a file with zeros
            print(f"Created placeholder: {file_path}")

    # Create a README to explain these are just placeholders
    readme_path = repo_root / "onnx_placeholder/README.md"
    readme_content = """# ONNX Runtime Placeholders

These files are empty placeholders used during the build process.
They are not functional ONNX Runtime libraries.

At runtime, you need to:
1. Install ONNX Runtime appropriate for your platform
2. Set the ORT_DYLIB_PATH environment variable to point to the real library
"""

    with open(readme_path, "w") as f:
        f.write(readme_content)
        print(f"Created README: {readme_path}")


if __name__ == "__main__":
    main()
