# Architecture-specific build configurations for CI

# Linux x86_64 setup (runs inside manylinux container)
[linux]
_setup-linux-x86_64:
    yum install -y openssl-devel

# Linux x86 (32-bit) setup - requires building ONNX Runtime from source
[linux]
_setup-linux-x86:
    #!/usr/bin/env -S bash -euxo pipefail
    yum install \
        openssl-devel.i686 \
        pkg-config \
        glibc-devel.i686 \
        libstdc++-devel.i686 \
        gcc-toolset-14-libstdc++-devel.i686 \
        -y
    which gcc
    ln -s $(which gcc) /usr/bin/i686-linux-gnu-gcc
    ls -l /usr/bin/i686-linux-gnu-gcc
    /usr/bin/i686-linux-gnu-gcc --version
    i686-linux-gnu-gcc --version

    # Build ONNX Runtime from source for 32-bit
    git clone https://github.com/microsoft/onnxruntime.git
    pushd onnxruntime
    git submodule update --init --recursive
    export PATH=/opt/python/cp38-cp38/bin:$PATH
    python -m pip install flatbuffers pysnooper
    sed -i '/def main():/i import pysnooper\n\n@pysnooper.snoop(depth=3)' tools/ci_build/build.py
    ./build.sh \
        --config Release \
        --build_shared_lib \
        --parallel \
        --skip_submodule_sync \
        --allow_running_as_root \
        --skip_tests \
        --compile_no_warning_as_error \
        --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
        --cmake_extra_defines CMAKE_TOOLCHAIN_FILE=$PWD/../scripts/i686-toolchain.cmake \
        --cmake_extra_defines CMAKE_VERBOSE_MAKEFILE=ON \
        --cmake_extra_defines onnxruntime_USE_AVX2=OFF
    file $(readlink -f build/Linux/Release/libonnxruntime.so)
    popd

    # Configure Cargo for cross-compilation
    mkdir -p rewrite/.cargo
    cat > rewrite/.cargo/config.toml << 'EOF'
    [target.i686-unknown-linux-gnu]
    rustflags = [
      "-C", "link-search=/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release",
      "-C", "link-arg=-v",
    ]
    EOF
    cat rewrite/.cargo/config.toml

# Linux aarch64 setup (runs on host before container)
_setup-linux-aarch64:
    #!/usr/bin/env -S bash -euxo pipefail
    sudo apt-get update -y
    sudo apt-get install pkg-config libssl-dev -y

# Linux armv7 setup - requires building ONNX Runtime from source
_setup-linux-armv7:
    #!/usr/bin/env -S bash -euxo pipefail
    sudo apt-get update -y
    sudo apt-get install -y gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf libssl-dev

    git clone https://github.com/microsoft/onnxruntime.git
    pushd onnxruntime
    git submodule update --init --recursive
    python3 -m pip install --upgrade pip flatbuffers pysnooper cmake
    sed -i '/def main():/i import pysnooper\n\n@pysnooper.snoop(depth=3)' tools/ci_build/build.py
    ./build.sh \
        --config Release \
        --build_shared_lib \
        --parallel \
        --skip_submodule_sync \
        --allow_running_as_root \
        --skip_tests \
        --compile_no_warning_as_error \
        --arm \
        --cmake_extra_defines CMAKE_VERBOSE_MAKEFILE=ON \
        --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
        --cmake_extra_defines CMAKE_TOOLCHAIN_FILE=$PWD/../scripts/armv7-linux-gnueabihf-toolchain.cmake \
        --cmake_extra_defines CMAKE_C_COMPILER=arm-linux-gnueabihf-gcc \
        --cmake_extra_defines CMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++
    file $(readlink -f build/Linux/Release/libonnxruntime.so)
    echo "Checking ONNX Runtime library in build/Linux/Release:"
    ls -la build/Linux/Release/
    echo 'int main() { return 0; }' > /tmp/test.c
    arm-linux-gnueabihf-gcc /tmp/test.c -lonnxruntime "-L${PWD}/build/Linux/Release" -o /tmp/linktest
    echo "Linkage successful!"
    arm-linux-gnueabihf-readelf -d build/Linux/Release/libonnxruntime.so
    popd

    mkdir -p rewrite/.cargo
    cat > rewrite/.cargo/config.toml << 'EOF'
    [target.armv7-unknown-linux-gnueabihf]
    rustflags = [
      "-C", "link-arg=-Wl,--verbose",
      "-C", "link-search=/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release",
      "-C", "link-arg=-Wl,-rpath,/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release",
      "-C", "link-search=native=/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release",
    ]
    EOF
    cat rewrite/.cargo/config.toml

    cat << 'EOF' > rewrite/build.rs
    fn main() {
        println!("cargo:rustc-link-search=native=/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release");
    }
    EOF

# Linux s390x setup - requires building ONNX Runtime from source
_setup-linux-s390x:
    #!/usr/bin/env -S bash -euxo pipefail
    sudo apt-get update -y
    sudo apt-get install -y gcc-s390x-linux-gnu g++-s390x-linux-gnu libssl-dev
    sudo ln -s "$(command -v s390x-linux-gnu-gcc)" /usr/bin/s390x-unknown-linux-gnu-gcc

    git clone https://github.com/microsoft/onnxruntime.git
    pushd onnxruntime
    git submodule update --init --recursive
    python3 -m pip install flatbuffers pysnooper
    python3 -m pip install --upgrade cmake
    sed -i '/def main():/i import pysnooper\n\n@pysnooper.snoop(depth=3)' tools/ci_build/build.py
    ./build.sh \
        --config Release \
        --build_shared_lib \
        --parallel \
        --skip_submodule_sync \
        --allow_running_as_root \
        --skip_tests \
        --compile_no_warning_as_error \
        --cmake_extra_defines CMAKE_VERBOSE_MAKEFILE=ON \
        --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
        --cmake_extra_defines CMAKE_TOOLCHAIN_FILE=$PWD/../scripts/s390x-linux-gnu-toolchain.cmake \
        --cmake_extra_defines CMAKE_C_COMPILER=s390x-linux-gnu-gcc \
        --cmake_extra_defines CMAKE_CXX_COMPILER=s390x-linux-gnu-g++
    file $(readlink -f build/Linux/Release/libonnxruntime.so)
    echo "Checking ONNX Runtime library:"
    ls -la build/Linux/Release/
    echo "Testing linker:"
    echo 'int main() { return 0; }' > /tmp/test.c
    s390x-linux-gnu-gcc /tmp/test.c -lonnxruntime "-L${PWD}/build/Linux/Release" -o /tmp/linktest
    echo "Linkage successful!"
    s390x-linux-gnu-readelf -d build/Linux/Release/libonnxruntime.so
    popd

    mkdir -p rewrite/.cargo
    cat > rewrite/.cargo/config.toml << 'EOF'
    [target.s390x-unknown-linux-gnu]
    rustflags = [
      "-C", "link-arg=-Wl,--verbose",
      "-C", "link-search=/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release",
      "-C", "link-arg=-Wl,-rpath,/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release",
      "-C", "link-search=native=/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release",
    ]
    EOF
    cat rewrite/.cargo/config.toml

    cat << 'EOF' > rewrite/build.rs
    fn main() {
        println!("cargo:rustc-link-search=native=/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release");
    }
    EOF

# Linux ppc64le setup - requires building ONNX Runtime from source
_setup-linux-ppc64le:
    #!/usr/bin/env -S bash -euxo pipefail
    sudo apt-get update -y
    sudo apt-get install -y gcc-powerpc64le-linux-gnu g++-powerpc64le-linux-gnu libssl-dev
    sudo ln -s "$(command -v powerpc64le-linux-gnu-gcc)" /usr/bin/powerpc64le-unknown-linux-gnu-gcc

    git clone https://github.com/microsoft/onnxruntime.git
    pushd onnxruntime
    git submodule update --init --recursive
    python3 -m pip install flatbuffers pysnooper
    python3 -m pip install --upgrade cmake
    sed -i '/def main():/i import pysnooper\n\n@pysnooper.snoop(depth=3)' tools/ci_build/build.py
    ./build.sh \
        --config Release \
        --build_shared_lib \
        --parallel \
        --skip_submodule_sync \
        --allow_running_as_root \
        --skip_tests \
        --compile_no_warning_as_error \
        --cmake_extra_defines CMAKE_VERBOSE_MAKEFILE=ON \
        --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
        --cmake_extra_defines CMAKE_TOOLCHAIN_FILE=$PWD/../scripts/ppc64le-linux-gnu-toolchain.cmake \
        --cmake_extra_defines CMAKE_C_COMPILER=powerpc64le-linux-gnu-gcc \
        --cmake_extra_defines CMAKE_CXX_COMPILER=powerpc64le-linux-gnu-g++
    file $(readlink -f build/Linux/Release/libonnxruntime.so)
    echo "Checking ONNX Runtime library:"
    ls -la build/Linux/Release/
    echo "Testing linker:"
    echo 'int main() { return 0; }' > /tmp/test.c
    powerpc64le-linux-gnu-gcc /tmp/test.c -lonnxruntime "-L${PWD}/build/Linux/Release" -o /tmp/linktest
    echo "Linkage successful!"
    powerpc64le-linux-gnu-readelf -d build/Linux/Release/libonnxruntime.so
    popd

    mkdir -p rewrite/.cargo
    cat > rewrite/.cargo/config.toml << 'EOF'
    [target.powerpc64le-unknown-linux-gnu]
    rustflags = [
      "-C", "link-arg=-Wl,--verbose",
      "-C", "link-search=/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release",
      "-C", "link-arg=-Wl,-rpath,/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release",
      "-C", "link-search=native=/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release",
    ]
    EOF
    cat rewrite/.cargo/config.toml

    cat << 'EOF' > rewrite/build.rs
    fn main() {
        println!("cargo:rustc-link-search=native=/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release");
    }
    EOF

# Dispatcher for platform setup - call with os and target
# Usage: just ci-setup-platform linux x86_64
ci-setup-platform os target:
    just _setup-{{os}}-{{target}}

# Output JSON matrix for CI - matches the CI.yml structure
ci-platform-matrix:
    #!/usr/bin/env -S bash -euo pipefail
    cat << 'EOF'
    {
      "include": [
        {
          "os": "linux",
          "runner": "ubuntu-22.04",
          "target": "x86_64",
          "rust_target": "x86_64-unknown-linux-gnu",
          "manylinux": "auto",
          "features": "",
          "skip": false,
          "dockeropts": ""
        },
        {
          "os": "linux",
          "runner": "ubuntu-22.04",
          "target": "x86",
          "rust_target": "i686-unknown-linux-gnu",
          "manylinux": "2_28",
          "features": "",
          "skip": false,
          "dockeropts": "--env CC_i686_unknown_linux_gnu=/usr/bin/i686-linux-gnu-gcc --env CFLAGS_i686_unknown_linux_gnu=-m32 --env LIBRARY_PATH=/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release:/usr/lib:$LIBRARY_PATH --env LD_LIBRARY_PATH=/home/runner/work/polars-fastembed/polars-fastembed/onnxruntime/build/Linux/Release:/usr/lib:$LD_LIBRARY_PATH --env PKG_CONFIG_ALLOW_CROSS=1 --env PKG_CONFIG_PATH=/usr/lib/pkgconfig:/usr/lib64/pkgconfig --env OPENSSL_DIR=/usr --env ORT_LIB_LOCATION=$PWD/onnxruntime/build/Linux/Release --env ORT_LIB_PROFILE=Release --env ORT_STRATEGY=system --env ORT_PREFER_DYNAMIC_LINK=1"
        },
        {
          "os": "linux",
          "runner": "ubuntu-22.04",
          "target": "aarch64",
          "rust_target": "aarch64-unknown-linux-gnu",
          "manylinux": "2_28",
          "features": "--features openssl-vendored",
          "skip": false,
          "dockeropts": "--env OPENSSL_DIR=/usr"
        },
        {
          "os": "linux",
          "runner": "ubuntu-22.04",
          "target": "armv7",
          "rust_target": "armv7-unknown-linux-gnueabihf",
          "manylinux": "auto",
          "features": "--features openssl-vendored",
          "skip": false,
          "dockeropts": "--env CC_armv7_unknown_linux_gnueabihf=arm-linux-gnueabihf-gcc --env CXX_armv7_unknown_linux_gnueabihf=arm-linux-gnueabihf-g++ --env AR_armv7_unknown_linux_gnueabihf=arm-linux-gnueabihf-ar --env LIBRARY_PATH=$PWD/onnxruntime/build/Linux/Release:/usr/lib:$LIBRARY_PATH --env LD_LIBRARY_PATH=$PWD/onnxruntime/build/Linux/Release:/usr/lib:$LD_LIBRARY_PATH --env PKG_CONFIG_ALLOW_CROSS=1 --env OPENSSL_DIR=/usr --env ORT_LIB_LOCATION=$PWD/onnxruntime/build/Linux/Release --env ORT_LIB_PROFILE=Release --env ORT_STRATEGY=system --env ORT_PREFER_DYNAMIC_LINK=1"
        },
        {
          "os": "linux",
          "runner": "ubuntu-22.04",
          "target": "s390x",
          "rust_target": "s390x-unknown-linux-gnu",
          "manylinux": "auto",
          "features": "--features openssl-vendored",
          "skip": false,
          "dockeropts": "--env CC_s390x_unknown_linux_gnu=s390x-linux-gnu-gcc --env CXX_s390x_unknown_linux_gnu=s390x-linux-gnu-g++ --env AR_s390x_unknown_linux_gnu=s390x-linux-gnu-ar --env CFLAGS_s390x_unknown_linux_gnu=-march=z10 --env LIBRARY_PATH=$PWD/onnxruntime/build/Linux/Release:/usr/lib:$LIBRARY_PATH --env LD_LIBRARY_PATH=$PWD/onnxruntime/build/Linux/Release:/usr/lib:$LD_LIBRARY_PATH --env PKG_CONFIG_ALLOW_CROSS=1 --env OPENSSL_DIR=/usr --env ORT_LIB_LOCATION=$PWD/onnxruntime/build/Linux/Release --env ORT_LIB_PROFILE=Release --env ORT_STRATEGY=system --env ORT_PREFER_DYNAMIC_LINK=1"
        },
        {
          "os": "linux",
          "runner": "ubuntu-22.04",
          "target": "ppc64le",
          "rust_target": "powerpc64le-unknown-linux-gnu",
          "manylinux": "auto",
          "features": "--features openssl-vendored",
          "skip": false,
          "dockeropts": "--env CC_powerpc64le_unknown_linux_gnu=powerpc64le-linux-gnu-gcc --env CXX_powerpc64le_unknown_linux_gnu=powerpc64le-linux-gnu-g++ --env AR_powerpc64le_unknown_linux_gnu=powerpc64le-linux-gnu-ar --env LIBRARY_PATH=$PWD/onnxruntime/build/Linux/Release:/usr/lib:$LIBRARY_PATH --env LD_LIBRARY_PATH=$PWD/onnxruntime/build/Linux/Release:/usr/lib:$LD_LIBRARY_PATH --env PKG_CONFIG_ALLOW_CROSS=1 --env OPENSSL_DIR=/usr --env ORT_LIB_LOCATION=$PWD/onnxruntime/build/Linux/Release --env ORT_LIB_PROFILE=Release --env ORT_STRATEGY=system --env ORT_PREFER_DYNAMIC_LINK=1"
        },
        {
          "os": "windows",
          "runner": "windows-latest",
          "target": "x64",
          "rust_target": "x86_64-pc-windows-msvc",
          "features": "",
          "skip": false
        },
        {
          "os": "macos",
          "runner": "macos-13",
          "target": "x86_64",
          "rust_target": "x86_64-apple-darwin",
          "features": "",
          "skip": false
        },
        {
          "os": "macos",
          "runner": "macos-14",
          "target": "aarch64",
          "rust_target": "aarch64-apple-darwin",
          "features": "",
          "skip": false
        }
      ]
    }
    EOF

# Output simplified test matrix (native platforms only - no cross-compilation)
# For testing, we only need platforms where we can actually run the tests
ci-test-matrix:
    #!/usr/bin/env -S bash -euo pipefail
    cat << 'EOF'
    {
      "include": [
        {
          "os": "linux",
          "runner": "ubuntu-22.04",
          "target": "x86_64",
          "features": ""
        },
        {
          "os": "windows",
          "runner": "windows-latest",
          "target": "x64",
          "features": ""
        },
        {
          "os": "macos",
          "runner": "macos-13",
          "target": "x86_64",
          "features": ""
        },
        {
          "os": "macos",
          "runner": "macos-14",
          "target": "aarch64",
          "features": ""
        }
      ]
    }
    EOF
