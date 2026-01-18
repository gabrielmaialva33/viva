pub mod fallback;
pub mod linux;
pub mod trait_def;
pub mod wsl;


// Conditional compilation for Windows module
#[cfg(target_os = "windows")]
pub mod windows;
// #[cfg(target_os = "macos")]
// pub mod macos;
