//! Build script for VIVA Body
//!
//! Cross-platform build for Linux, macOS, and Windows.
//! Generates compile-time constants:
//! - VIVA_GIT_HASH: Short commit hash
//! - VIVA_BUILD_TIME: ISO 8601 timestamp

use std::process::Command;

fn main() {
    // Link C++ standard library (required for Jolt Physics)
    // Windows MSVC links automatically, GNU/Linux needs explicit link
    #[cfg(all(target_os = "linux", not(target_env = "msvc")))]
    println!("cargo:rustc-link-lib=stdc++");

    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");

    // Windows MSVC - no explicit link needed, but ensure we have correct runtime
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        // MSVC links C++ runtime automatically
        // Ensure we use the same CRT as Jolt
        println!("cargo:rustc-link-arg=/MD"); // Dynamic CRT
    }

    // Rerun if git HEAD changes
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");

    // Git commit hash (cross-platform)
    let git_hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    // Build timestamp (cross-platform)
    let build_time = get_build_time();

    println!("cargo:rustc-env=VIVA_GIT_HASH={}", git_hash);
    println!("cargo:rustc-env=VIVA_BUILD_TIME={}", build_time);
}

fn get_build_time() -> String {
    // Try Unix date first
    if let Some(time) = try_unix_date() {
        return time;
    }

    // Try PowerShell on Windows
    #[cfg(target_os = "windows")]
    if let Some(time) = try_powershell_date() {
        return time;
    }

    "unknown".to_string()
}

fn try_unix_date() -> Option<String> {
    Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
}

#[cfg(target_os = "windows")]
fn try_powershell_date() -> Option<String> {
    Command::new("powershell")
        .args(["-Command", "Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ' -AsUTC"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
}
