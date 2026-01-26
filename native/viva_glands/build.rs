fn main() {
    // Link cuFFT library
    println!("cargo:rustc-link-lib=cufft");

    // CUDA library paths (already set by candle, but ensuring cufft is found)
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    }
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
