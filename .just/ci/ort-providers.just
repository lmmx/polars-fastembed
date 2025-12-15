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
    uv venv test-env
    uv pip install --python test-env pytest
    uv pip install --python test-env {{wheel_dir}}/*.whl
    uv run --python test-env pytest polars-fastembed-cpu/tests -v

# Test CUDA provider wheel
test-ort-cuda wheel_dir:
    uv venv test-env
    uv pip install --python test-env pytest
    uv pip install --python test-env {{wheel_dir}}/*.whl
    uv run --python test-env pytest polars-fastembed-cuda/tests -v
