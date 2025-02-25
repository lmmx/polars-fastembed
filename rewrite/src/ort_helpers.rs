// src/ort_helpers.rs

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Platform-specific ONNX Runtime lib filename
#[cfg(target_os = "linux")]
pub const ONNX_LIB_NAME: &str = "libonnxruntime.so";

#[cfg(target_os = "macos")]
pub const ONNX_LIB_NAME: &str = "libonnxruntime.dylib";

#[cfg(target_os = "windows")]
pub const ONNX_LIB_NAME: &str = "onnxruntime.dll";

/// Either download or extract the ONNX Runtime shared library appropriate for the current platform
/// and architecture, and set the ORT_DYLIB_PATH environment variable to point to it.
pub fn setup_ort_dylib() -> Result<(), String> {
    // First, check if ORT_DYLIB_PATH is already set and points to a valid file
    if let Ok(existing_path) = env::var("ORT_DYLIB_PATH") {
        if Path::new(&existing_path).exists() {
            return Ok(());
        }
    }

    let dylib_path = get_ort_dylib_path()?;

    // Set the environment variable for ORT to find the dynamic library
    env::set_var("ORT_DYLIB_PATH", &dylib_path);
    Ok(())
}

// We don't embed binaries - instead, in CI environment we'll use placeholders
// Then at runtime in the real environment, a proper ONNX Runtime library
// needs to be accessible via ORT_DYLIB_PATH
static ONNX_LIB_BYTES: &[u8] = &[];

fn get_ort_dylib_path() -> Result<String, String> {
    // If ORT_DYLIB_PATH environment variable is set, use that path directly
    if let Ok(env_path) = env::var("ORT_DYLIB_PATH") {
        return Ok(env_path);
    }

    // Look for onnxruntime in standard locations
    let possible_lib_paths = get_standard_lib_paths();
    for path in possible_lib_paths {
        if path.exists() {
            return path.to_str()
                .map(|s| s.to_string())
                .ok_or_else(|| "Path contains invalid unicode".to_string());
        }
    }

    // If not found, return instructions for user
    Err(format!(
        "ONNX Runtime library not found for this platform.\n\
         Please install ONNX Runtime and set ORT_DYLIB_PATH environment variable \
         to point to the {} file.",
        ONNX_LIB_NAME
    ))
}

fn get_standard_lib_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // Add some standard library locations for different platforms
    #[cfg(target_os = "linux")]
    {
        // Common locations on Linux
        paths.push(PathBuf::from("/usr/lib").join(ONNX_LIB_NAME));
        paths.push(PathBuf::from("/usr/local/lib").join(ONNX_LIB_NAME));

        // Python environment locations
        if let Ok(python_path) = env::var("PYTHONPATH") {
            for path in python_path.split(':') {
                paths.push(PathBuf::from(path).join("lib").join(ONNX_LIB_NAME));
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Common locations on Windows
        if let Ok(program_files) = env::var("ProgramFiles") {
            paths.push(PathBuf::from(program_files).join("ONNX Runtime").join(ONNX_LIB_NAME));
        }

        // Python environment
        if let Ok(python_path) = env::var("PYTHONPATH") {
            for path in python_path.split(';') {
                paths.push(PathBuf::from(path).join("Lib").join("site-packages").join("onnxruntime").join(ONNX_LIB_NAME));
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Common locations on macOS
        paths.push(PathBuf::from("/usr/local/lib").join(ONNX_LIB_NAME));
        paths.push(PathBuf::from("/opt/homebrew/lib").join(ONNX_LIB_NAME));
    }

    paths
}

// This method is no longer needed since we're not extracting any embedded libraries
// We'll keep the dependency on dirs though in case we need it for other purposes later
