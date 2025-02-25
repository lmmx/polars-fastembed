use std::env;
use std::path::Path;

fn main() {
    // Set up build configuration for dynamic ONNX Runtime loading
    #[cfg(feature = "ort-dynamic")]
    {
        println!("cargo:rustc-cfg=ort_dynamic");
        
        // This tells cargo to re-run this script if any of these files change
        for arch in &["armv7", "x86", "s390x", "ppc64le", "x86_win"] {
            let placeholder_path = format!("onnx_placeholder/{}", arch);
            println!("cargo:rerun-if-changed={}", placeholder_path);
        }
    }
    
    // Set up build configuration for OpenSSL
    #[cfg(feature = "openssl-vendored")]
    {
        println!("cargo:rustc-cfg=openssl_vendored");
    }
}