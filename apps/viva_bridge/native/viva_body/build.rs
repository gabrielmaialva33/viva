//! Build script for VIVA Body
//!
//! Generates compile-time constants:
//! - VIVA_GIT_HASH: Short commit hash
//! - VIVA_BUILD_TIME: ISO 8601 timestamp

use std::process::Command;

fn main() {
    // Rerun if git HEAD changes
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");

    // Git commit hash
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

    // Build timestamp (naive, no chrono dep)
    let build_time = Command::new("date")
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
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=VIVA_GIT_HASH={}", git_hash);
    println!("cargo:rustc-env=VIVA_BUILD_TIME={}", build_time);
}
