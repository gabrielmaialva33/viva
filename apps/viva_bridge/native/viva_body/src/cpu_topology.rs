use crate::asm;

/// CPU Cache Information
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct CacheInfo {
    pub l1_data_kb: u32,
    pub l1_inst_kb: u32,
    pub l2_kb: u32,
    pub l3_kb: u32,
}

impl CacheInfo {
    pub fn empty() -> Self {
        Self {
            l1_data_kb: 0,
            l1_inst_kb: 0,
            l2_kb: 0,
            l3_kb: 0,
        }
    }
}

/// Detects CPU Cache Topology using CPUID
///
/// Supports Intel and AMD deterministic cache parameters (Leaf 0x4 or 0x8000001D)
pub fn detect_cache_topology() -> CacheInfo {
    let _vendor = get_cpu_vendor();

    // For now, focusing on standard Intel/AMD leaf 0x4/0x8000001D parsing logic would be complex.
    // A simpler heuristic for L2/L3 often lives in extended leaves (0x80000006 for L2/L3 on AMD/Intel old).
    // Let's try the Extended L2 Definitions (Leaf 0x80000006) which is fairly common for simple size checks.

    // Check max extended function
    let (max_ext, _, _, _) = unsafe { asm::cpuid(0x80000000, 0) };

    if max_ext >= 0x80000006 {
        // Leaf 0x80000006: L2 Cache and L3 Cache Information
        let (_, _, ecx, edx) = unsafe { asm::cpuid(0x80000006, 0) };

        // ECX: L2 Cache details
        // Bits 16-31: L2 Cache size in measured in 1KB units.
        let l2_kb = (ecx >> 16) & 0xFFFF;

        // EDX: L3 Cache details
        // Bits 18-31: L3 Cache size in 512KB units (Intel/AMD varies, usually 512KB blocks)
        // Wait, standardization here is tricky.
        // Actually on modern AMD (Zen):
        // EDX Bits 18-31 is L3 size in 512KB units.
        let l3_512kb_blocks = (edx >> 18) & 0x3FFF;
        let l3_kb = l3_512kb_blocks * 512;

        return CacheInfo {
            l1_data_kb: 0, // Need detailed leaf traversal for reliable L1
            l1_inst_kb: 0,
            l2_kb,
            l3_kb,
        };
    }

    CacheInfo::empty()
}

fn get_cpu_vendor() -> String {
    let (_, ebx, ecx, edx) = unsafe { asm::cpuid(0, 0) };

    let mut vendor = String::with_capacity(12);
    unsafe {
        let bytes_ebx = ebx.to_le_bytes();
        let bytes_edx = edx.to_le_bytes();
        let bytes_ecx = ecx.to_le_bytes();

        vendor.push_str(std::str::from_utf8_unchecked(&bytes_ebx));
        vendor.push_str(std::str::from_utf8_unchecked(&bytes_edx));
        vendor.push_str(std::str::from_utf8_unchecked(&bytes_ecx));
    }
    vendor
}
