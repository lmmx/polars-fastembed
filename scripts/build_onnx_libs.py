#!/usr/bin/env python3
"""
Build ONNX Runtime for specialized architectures (s390x, ppc64le).

This script builds ONNX Runtime from source for architectures that don't 
have pre-built binaries available.
"""

import os
import subprocess
import shutil
import sys
from pathlib import Path

# ONNX Runtime version to build
ONNX_VERSION = "1.17.0"

def clone_repo(version):
    """Clone the ONNX Runtime repository at the specified version."""
    print(f"Cloning ONNX Runtime repository at version {version}...")
    if os.path.exists("onnxruntime"):
        shutil.rmtree("onnxruntime")
    
    subprocess.run(["git", "clone", "--recursive", "--branch", f"v{version}", 
                    "https://github.com/microsoft/onnxruntime.git"], check=True)

def build_for_s390x():
    """Build ONNX Runtime for s390x architecture."""
    print("Building ONNX Runtime for s390x...")
    cwd = os.path.join(os.getcwd(), "onnxruntime")
    
    # Configure and build
    build_cmd = [
        "./build.sh",
        "--config", "Release",
        "--build_shared_lib",
        "--parallel",
        "--skip_tests",
        "--skip_onnx_tests",
        "--cmake_extra_defines", "CMAKE_SYSTEM_NAME=Linux CMAKE_SYSTEM_PROCESSOR=s390x"
    ]
    
    subprocess.run(build_cmd, cwd=cwd, check=True)
    
    # Copy the built library
    src_lib = os.path.join(cwd, "build", "Linux", "Release", "libonnxruntime.so")
    target_dir = os.path.join(os.getcwd(), "onnx_libs", "s390x")
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy2(src_lib, os.path.join(target_dir, "libonnxruntime.so"))

def build_for_ppc64le():
    """Build ONNX Runtime for ppc64le architecture."""
    print("Building ONNX Runtime for ppc64le...")
    cwd = os.path.join(os.getcwd(), "onnxruntime")
    
    # Configure and build
    build_cmd = [
        "./build.sh",
        "--config", "Release",
        "--build_shared_lib",
        "--parallel",
        "--skip_tests",
        "--skip_onnx_tests",
        "--cmake_extra_defines", "CMAKE_SYSTEM_NAME=Linux CMAKE_SYSTEM_PROCESSOR=ppc64le"
    ]
    
    subprocess.run(build_cmd, cwd=cwd, check=True)
    
    # Copy the built library
    src_lib = os.path.join(cwd, "build", "Linux", "Release", "libonnxruntime.so")
    target_dir = os.path.join(os.getcwd(), "onnx_libs", "ppc64le")
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy2(src_lib, os.path.join(target_dir, "libonnxruntime.so"))

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: build_onnx_libs.py <architecture>")
        print("Available architectures: s390x, ppc64le")
        return
    
    arch = sys.argv[1].lower()
    
    # Clone the repository
    clone_repo(ONNX_VERSION)
    
    # Build for the specified architecture
    if arch == "s390x":
        build_for_s390x()
    elif arch == "ppc64le":
        build_for_ppc64le()
    else:
        print(f"Unsupported architecture: {arch}")
        print("Available architectures: s390x, ppc64le")
        return

if __name__ == "__main__":
    main()