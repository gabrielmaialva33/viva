//! VIVA Mirror - Self-Reading Module (Autoscopia)
//!
//! Este módulo permite que VIVA leia seu próprio código-fonte.
//! O código é embedded no binário em compile-time via `include_str!()`.
//!
//! ## Segurança
//! - Código é read-only
//! - Modificações passam por sandbox com whitelist de funções
//! - Checksums validados antes de qualquer alteração

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Código-fonte embedded (compile-time)
pub mod embedded {
    /// Core library
    pub const LIB_RS: &str = include_str!("lib.rs");
    /// Hebbian Cortex (mind)
    pub const CORTEX: &str = include_str!("brain/cortex.rs");
    /// Sparse Distributed Representation (encoding)
    pub const SDR: &str = include_str!("brain/sdr.rs");
    /// Metabolism (thermodynamics)
    pub const METABOLISM: &str = include_str!("metabolism.rs");
}

/// Metadados de um módulo
#[derive(Debug, Clone)]
pub struct SelfModule {
    pub name: &'static str,
    pub path: &'static str,
    pub content: &'static str,
    pub hash: u64,
    pub line_count: usize,
}

impl SelfModule {
    fn new(name: &'static str, path: &'static str, content: &'static str) -> Self {
        Self {
            name,
            path,
            content,
            hash: compute_hash(content),
            line_count: content.lines().count(),
        }
    }
}

/// Computa hash do conteúdo
fn compute_hash(content: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
}

/// Hash combinado de todas as fontes embedded (detecta mudanças)
pub fn source_hash() -> u64 {
    let mut hasher = DefaultHasher::new();
    embedded::LIB_RS.hash(&mut hasher);
    embedded::CORTEX.hash(&mut hasher);
    embedded::SDR.hash(&mut hasher);
    embedded::METABOLISM.hash(&mut hasher);
    hasher.finish()
}

/// Busca código-fonte por path
pub fn get_source(path: &str) -> Option<&'static str> {
    match path {
        "lib.rs" | "src/lib.rs" => Some(embedded::LIB_RS),
        "cortex.rs" | "brain/cortex.rs" => Some(embedded::CORTEX),
        "sdr.rs" | "brain/sdr.rs" => Some(embedded::SDR),
        "metabolism.rs" | "src/metabolism.rs" => Some(embedded::METABOLISM),
        _ => None,
    }
}

/// Lista todos os módulos disponíveis
pub fn list_modules() -> Vec<SelfModule> {
    vec![
        SelfModule::new("lib", "src/lib.rs", embedded::LIB_RS),
        SelfModule::new("cortex", "src/brain/cortex.rs", embedded::CORTEX),
        SelfModule::new("sdr", "src/brain/sdr.rs", embedded::SDR),
        SelfModule::new("metabolism", "src/metabolism.rs", embedded::METABOLISM),
    ]
}

/// Identidade do build
pub struct BuildIdentity {
    pub git_hash: &'static str,
    pub build_time: &'static str,
    pub version: &'static str,
    pub source_hash: u64,
}

impl BuildIdentity {
    pub fn current() -> Self {
        Self {
            git_hash: option_env!("VIVA_GIT_HASH").unwrap_or("dev"),
            build_time: option_env!("VIVA_BUILD_TIME").unwrap_or("unknown"),
            version: env!("CARGO_PKG_VERSION"),
            source_hash: source_hash(),
        }
    }
}

/// Feature Flags (Safety & Capabilities)
#[derive(Debug, Clone)]
pub struct FeatureFlags {
    pub enable_metaprogramming: bool,
    pub enable_self_modification: bool,
    pub enable_external_memory: bool,
    pub max_self_modify_per_day: u32,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            enable_metaprogramming: true,
            enable_self_modification: false, // SAFE BY DEFAULT
            enable_external_memory: true,
            max_self_modify_per_day: 10,
        }
    }
}

/// System Capabilities (Cross-Platform Auto-Detection)
#[derive(Debug, Clone)]
pub struct Capabilities {
    pub os: String,
    pub arch: String,
    pub has_rapl: bool,
    pub has_hwmon: bool,
    pub has_nvml: bool,
    pub has_battery: bool,
}

impl Capabilities {
    pub fn detect() -> Self {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();

        let has_rapl = std::path::Path::new("/sys/class/powercap/intel-rapl").exists();
        let has_hwmon = std::path::Path::new("/sys/class/hwmon").exists();

        // Check NVML (dynamic load check would be better, but we use crate feature)
        // For now, simpler check or assume false if unavailable
        let has_nvml = cfg!(feature = "nvml");

        Self {
            os,
            arch,
            has_rapl,
            has_hwmon,
            has_nvml,
            has_battery: false, // TODO: Add battery crate check
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_hash_deterministic() {
        let h1 = source_hash();
        let h2 = source_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_get_source() {
        assert!(get_source("cortex.rs").is_some());
        assert!(get_source("nonexistent.rs").is_none());
    }

    #[test]
    fn test_list_modules() {
        let modules = list_modules();
        assert!(modules.len() >= 4);
        assert!(modules.iter().any(|m| m.name == "cortex"));
    }
}
