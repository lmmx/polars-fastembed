ORT_VERSION := "1.23.0"

# Download and extract ORT for CPU provider
[unix]
_download-ort-cpu platform archive lib_pattern:
    curl -L -o ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v{{ORT_VERSION}}/{{archive}}"
    tar -xzf ort.tgz
    mkdir -p polars-fastembed-cpu/python/polars_fastembed_cpu/libs
    cp onnxruntime-*/lib/{{lib_pattern}} polars-fastembed-cpu/python/polars_fastembed_cpu/libs/

[windows]
_download-ort-cpu platform archive lib_pattern:
    curl.exe -L -o ort.zip "https://github.com/microsoft/onnxruntime/releases/download/v{{ORT_VERSION}}/{{archive}}"
    powershell -Command "Expand-Archive -Path ort.zip -DestinationPath ."
    powershell -Command "New-Item -ItemType Directory -Force -Path polars-fastembed-cpu/python/polars_fastembed_cpu/libs"
    powershell -Command "Copy-Item 'onnxruntime-win-x64-{{ORT_VERSION}}/lib/{{lib_pattern}}' polars-fastembed-cpu/python/polars_fastembed_cpu/libs/"

# Download and extract ORT for CUDA provider
[unix]
_download-ort-cuda platform archive lib_pattern:
    curl -L -o ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v{{ORT_VERSION}}/{{archive}}"
    tar -xzf ort.tgz
    mkdir -p polars-fastembed-cuda/python/polars_fastembed_cuda/libs
    cp onnxruntime-*/lib/{{lib_pattern}} polars-fastembed-cuda/python/polars_fastembed_cuda/libs/

[windows]
_download-ort-cuda platform archive lib_pattern:
    curl.exe -L -o ort.zip "https://github.com/microsoft/onnxruntime/releases/download/v{{ORT_VERSION}}/{{archive}}"
    powershell -Command "Expand-Archive -Path ort.zip -DestinationPath ."
    powershell -Command "New-Item -ItemType Directory -Force -Path polars-fastembed-cuda/python/polars_fastembed_cuda/libs"
    powershell -Command "Copy-Item 'onnxruntime-win-x64-gpu-{{ORT_VERSION}}/lib/{{lib_pattern}}' polars-fastembed-cuda/python/polars_fastembed_cuda/libs/"

# Build CPU provider wheel
build-ort-cpu:
    uv build --wheel polars-fastembed-cpu

# Build CUDA provider wheel
build-ort-cuda:
    uv build --wheel polars-fastembed-cuda

# Rename wheel with platform tag (replaces 'any' with platform)
rename-wheel dir plat:
    cd {{dir}}/dist && for f in *.whl; do mv "$f" "$(echo $f | sed 's/-any\.whl/-{{plat}}.whl/')"; done

# Test CPU provider wheel
test-ort-cpu wheel_dir:
    #!/usr/bin/env bash
    set -euo pipefail
    min_py=$(just _min_version)
    uv venv test-env --python "$min_py"
    uv pip install --python test-env pytest
    uv pip install --python test-env {{wheel_dir}}/*.whl
    uv run --python test-env pytest polars-fastembed-cpu/tests -v

# Test CUDA provider wheel
test-ort-cuda wheel_dir:
    #!/usr/bin/env bash
    set -euo pipefail
    min_py=$(just _min_version)
    uv venv test-env --python "$min_py"
    uv pip install --python test-env pytest
    uv pip install --python test-env {{wheel_dir}}/*.whl
    uv run --python test-env pytest polars-fastembed-cuda/tests -v

# Build CUDA provider locally (requires CUDA toolkit)
build-local-cuda-provider sm_arch="86":
    #!/usr/bin/env bash
    set -e

    REPO_ROOT="$(pwd)"
    ORT_BUILD_DIR="$HOME/lab/ort/build"
    ORT_VERSION="1.23.0"

    cd "$ORT_BUILD_DIR"
    rm -rf onnxruntime
    git clone --recursive --depth 1 --branch v${ORT_VERSION} https://github.com/microsoft/onnxruntime

    # Apply patch for gather_block_quantized.cu
    cp "$REPO_ROOT/scripts/cuda-patches/gather_block_quantized.cu" \
       onnxruntime/onnxruntime/contrib_ops/cuda/quantization/gather_block_quantized.cu

    cd onnxruntime

    ./build.sh \
      --config Release \
      --build_shared_lib \
      --parallel \
      --use_cuda \
      --cuda_home /usr/lib/nvidia-cuda-toolkit \
      --cudnn_home /usr \
      --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="{{sm_arch}}" \
      --cmake_extra_defines onnxruntime_USE_FLASH_ATTENTION=OFF \
      --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
      --cmake_extra_defines onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF \
      --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-error=attributes" \
      --cmake_extra_defines CMAKE_CUDA_FLAGS="-Xcompiler=-Wno-error=attributes" \
      --skip_tests

    # Copy to provider package back in repo
    LIBS_DIR="$REPO_ROOT/polars-fastembed-cuda/python/polars_fastembed_cuda/libs"
    mkdir -p "$LIBS_DIR"
    cp build/Linux/Release/libonnxruntime.so.${ORT_VERSION} "$LIBS_DIR/"
    cp build/Linux/Release/libonnxruntime_providers_cuda.so "$LIBS_DIR/"
    cp build/Linux/Release/libonnxruntime_providers_shared.so "$LIBS_DIR/"

    upx --best "$LIBS_DIR/libonnxruntime.so.${ORT_VERSION}"
    upx --best "$LIBS_DIR/libonnxruntime_providers_cuda.so"

    cd "$LIBS_DIR"
    ln -sf libonnxruntime.so.${ORT_VERSION} libonnxruntime.so.1
    ln -sf libonnxruntime.so.${ORT_VERSION} libonnxruntime.so

# Compress libs with UPX (skips symlinks)
[unix]
compress-libs-upx dir:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{dir}}
    shopt -s nullglob
    for f in *.so *.so.* *.dylib *.dll; do
        if [ -f "$f" ] && [ ! -L "$f" ]; then
            upx --best "$f" || true
        fi
    done

[windows]
compress-libs-upx dir:
    powershell -Command "cd '{{dir}}'; Get-ChildItem -Filter '*.dll' | % { mise exec -- upx --best $$_.FullName }"

# Check no file in wheel exceeds size limit (default 100MB)
check-wheel-size dir max_mb="100":
    #!/usr/bin/env bash
    set -euo pipefail
    max_bytes=$(( {{max_mb}} * 1048576 ))
    for whl in {{dir}}/*.whl; do
        size=$(stat -c%s "$whl" 2>/dev/null || stat -f%z "$whl")
        size_mb=$(( size / 1048576 ))
        if [ "$size" -gt "$max_bytes" ]; then
            echo "ERROR: $whl is ${size_mb}MB, exceeds {{max_mb}}MB limit"
            exit 1
        fi
        echo "OK: $whl is ${size_mb}MB"
    done
