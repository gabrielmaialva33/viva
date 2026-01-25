//! CPU Cache Topology Detection
//!
//! Detects L1/L2/L3 cache sizes using CPUID.
//! Supports both Intel and AMD processors.

use crate::asm;

/// CPU Cache Information
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheInfo {
    pub l1_data_kb: u32,
    pub l1_inst_kb: u32,
    pub l2_kb: u32,
    pub l3_kb: u32,
}

impl CacheInfo {
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Cache type from CPUID Leaf 0x04
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum CacheType {
    Null = 0,
    Data = 1,
    Instruction = 2,
    Unified = 3,
}

impl From<u32> for CacheType {
    fn from(val: u32) -> Self {
        match val & 0x1F {
            1 => CacheType::Data,
            2 => CacheType::Instruction,
            3 => CacheType::Unified,
            _ => CacheType::Null,
        }
    }
}

/// Detects CPU Cache Topology using CPUID
///
/// - **Intel**: Uses Leaf 0x04 (Deterministic Cache Parameters)
/// - **AMD**: Uses Leaf 0x80000006 (Extended L2/L3 info)
#[cfg(target_arch = "x86_64")]
pub fn detect_cache_topology() -> CacheInfo {
    let vendor = get_cpu_vendor();

    if vendor.starts_with("GenuineIntel") {
        detect_cache_intel()
    } else if vendor.starts_with("AuthenticAMD") {
        detect_cache_amd()
    } else {
        // Unknown vendor - try AMD method as fallback (more universal)
        detect_cache_amd()
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn detect_cache_topology() -> CacheInfo {
    CacheInfo::empty()
}

/// Intel cache detection using Leaf 0x04
///
/// Iterates through cache levels (subleaf 0, 1, 2, ...) until type = 0
#[cfg(target_arch = "x86_64")]
fn detect_cache_intel() -> CacheInfo {
    let mut info = CacheInfo::empty();

    // Check if Leaf 0x04 is supported
    let (max_leaf, _, _, _) = unsafe { asm::cpuid(0, 0) };
    if max_leaf < 4 {
        return detect_cache_amd(); // Fallback
    }

    // Iterate through cache levels
    for subleaf in 0..16 {
        let (eax, ebx, ecx, _edx) = unsafe { asm::cpuid(0x04, subleaf) };

        let cache_type = CacheType::from(eax);
        if cache_type == CacheType::Null {
            break; // No more caches
        }

        // Extract cache parameters from EAX, EBX, ECX
        let cache_level = (eax >> 5) & 0x7;

        // EBX contains:
        // - Bits 0-11: Line size - 1
        // - Bits 12-21: Partitions - 1
        // - Bits 22-31: Ways - 1
        let line_size = (ebx & 0xFFF) + 1;
        let partitions = ((ebx >> 12) & 0x3FF) + 1;
        let ways = ((ebx >> 22) & 0x3FF) + 1;

        // ECX contains: Sets - 1
        let sets = ecx + 1;

        // Cache size = Ways × Partitions × Line_Size × Sets
        let cache_size_bytes = ways * partitions * line_size * sets;
        let cache_size_kb = cache_size_bytes / 1024;

        match (cache_level, cache_type) {
            (1, CacheType::Data) => info.l1_data_kb = cache_size_kb,
            (1, CacheType::Instruction) => info.l1_inst_kb = cache_size_kb,
            (2, CacheType::Unified) => info.l2_kb = cache_size_kb,
            (3, CacheType::Unified) => info.l3_kb = cache_size_kb,
            _ => {}
        }
    }

    info
}

/// AMD cache detection using Leaf 0x80000006
///
/// Simpler method that works for most AMD and older Intel CPUs
#[cfg(target_arch = "x86_64")]
fn detect_cache_amd() -> CacheInfo {
    let mut info = CacheInfo::empty();

    // Check max extended function
    let (max_ext, _, _, _) = unsafe { asm::cpuid(0x80000000, 0) };

    // L1 cache info (Leaf 0x80000005)
    if max_ext >= 0x80000005 {
        let (_, _, ecx, edx) = unsafe { asm::cpuid(0x80000005, 0) };

        // ECX: L1 Data cache
        // Bits 24-31: Size in KB
        info.l1_data_kb = (ecx >> 24) & 0xFF;

        // EDX: L1 Instruction cache
        // Bits 24-31: Size in KB
        info.l1_inst_kb = (edx >> 24) & 0xFF;
    }

    // L2/L3 cache info (Leaf 0x80000006)
    if max_ext >= 0x80000006 {
        let (_, _, ecx, edx) = unsafe { asm::cpuid(0x80000006, 0) };

        // ECX: L2 Cache
        // Bits 16-31: Size in KB
        info.l2_kb = (ecx >> 16) & 0xFFFF;

        // EDX: L3 Cache (AMD Zen specific)
        // Bits 18-31: Size in 512KB units
        let l3_512kb_blocks = (edx >> 18) & 0x3FFF;
        info.l3_kb = l3_512kb_blocks * 512;
    }

    info
}

/// Get CPU vendor string (e.g., "GenuineIntel", "AuthenticAMD")
#[cfg(target_arch = "x86_64")]
fn get_cpu_vendor() -> String {
    let (_, ebx, ecx, edx) = unsafe { asm::cpuid(0, 0) };

    let mut vendor = [0u8; 12];

    // Vendor string is in EBX:EDX:ECX order (not EBX:ECX:EDX!)
    vendor[0..4].copy_from_slice(&ebx.to_le_bytes());
    vendor[4..8].copy_from_slice(&edx.to_le_bytes());
    vendor[8..12].copy_from_slice(&ecx.to_le_bytes());

    String::from_utf8_lossy(&vendor).to_string()
}

#[cfg(not(target_arch = "x86_64"))]
fn get_cpu_vendor() -> String {
    "Unknown".to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_detect_cache() {
        let info = detect_cache_topology();

        // Should detect at least L2 on any modern CPU
        println!("Cache Info: {:?}", info);

        // Basic sanity checks
        assert!(info.l2_kb > 0 || info.l3_kb > 0, "Should detect some cache");

        // L3 should be larger than L2 if both exist
        if info.l2_kb > 0 && info.l3_kb > 0 {
            assert!(info.l3_kb >= info.l2_kb, "L3 should be >= L2");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_get_vendor() {
        let vendor = get_cpu_vendor();
        println!("CPU Vendor: {}", vendor);

        assert!(
            vendor.contains("Intel") || vendor.contains("AMD") || vendor.len() == 12,
            "Should be a valid vendor string"
        );
    }
}
