pub mod trait_def;
pub mod linux;
pub mod fallback;

// Conditional compilation for Windows module
#[cfg(target_os = "windows")]
pub mod windows;
// #[cfg(target_os = "macos")]
// pub mod macos;
