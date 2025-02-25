# Special Architecture Support

This document explains how to use `polars-fastembed` on architectures that require
dynamic loading of ONNX Runtime.

## Supported Architectures

`polars-fastembed` provides pre-built wheels for the following architectures that require
special handling:

- Linux armv7
- Linux x86 (32-bit)
- Linux s390x
- Linux ppc64le
- Windows x86 (32-bit)

## Using on Special Architectures

For these architectures, ONNX Runtime must be loaded dynamically at runtime rather than
being statically linked. This means you'll need to:

1. Install ONNX Runtime separately.
2. Make the library available through the `ORT_DYLIB_PATH` environment variable.

### Installing ONNX Runtime

You can install ONNX Runtime using pip:

```bash
pip install onnxruntime
```

Or download the appropriate binary for your platform from the
[ONNX Runtime GitHub releases](https://github.com/microsoft/onnxruntime/releases).

### Setting Up Environment Variables

Once ONNX Runtime is installed, you need to set the `ORT_DYLIB_PATH` environment variable
to point to the location of the ONNX Runtime library.

#### Linux

```bash
# Find the libonnxruntime.so path
find / -name "libonnxruntime.so" 2>/dev/null
# Set the environment variable
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
```

#### Windows

```powershell
# Find the onnxruntime.dll path
Get-ChildItem -Path C:\ -Filter onnxruntime.dll -Recurse -ErrorAction SilentlyContinue
# Set the environment variable
$env:ORT_DYLIB_PATH = "C:\path\to\onnxruntime.dll"
```

### Verifying Installation

You can check if the library is correctly found by running a simple test:

```python
import polars as pl
from polars_fastembed import register_model

model_id = "BAAI/bge-small-en-v1.5"
register_model(model_id)

df = pl.DataFrame({"text": ["Hello, world!"]})
df_emb = df.fastembed.embed(columns="text", model_name=model_id)
print(df_emb)
```

If this works without errors, your setup is correct.

## Troubleshooting

If you encounter errors like:

```
ORT Error: ./ort_helper.rs:55: Failed to load ONNX Runtime library
```

It means the library couldn't be found. Make sure:

1. ONNX Runtime is correctly installed.
2. The `ORT_DYLIB_PATH` environment variable is correctly set.
3. The path points to a valid ONNX Runtime library file.
