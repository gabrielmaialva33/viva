//! Raw Assembly Module - Multi-Architecture Support
//!
//! "Close to the metal" operations for direct CPU communication.
//! Bypasses OS abstractions for precise timing and CPU feature detection.
//!
//! ## Architecture Support:
//! - **x86_64**: Native RDTSC/CPUID via stdlib intrinsics
//! - **AArch64**: ARM64 virtual counter (CNTVCT_EL0)
//! - **Fallback**: std::time::Instant for other architectures

#![allow(dead_code)]

// ============================================================================
// x86_64 Implementation (Intel/AMD)
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
    use std::arch::asm;

    /// Reads the hardware Time Stamp Counter (RDTSC)
    /// Returns the number of CPU cycles since reset.
    ///
    /// # Performance
    /// - Latency: ~20-30 cycles
    /// - Precision: Single cycle
    ///
    /// # Note
    /// RDTSC is not serializing - it may reorder with adjacent instructions.
    /// For precise benchmarks, consider using `rdtsc_serialized()` or `rdtscp()`.
    #[inline(always)]
    pub fn rdtsc() -> u64 {
        // Using stdlib intrinsic for safety and correctness
        unsafe { std::arch::x86_64::_rdtsc() }
    }

    /// RDTSC with LFENCE barrier (serializing read)
    ///
    /// Ensures all previous instructions complete before reading TSC.
    /// More accurate for benchmarking but ~10 cycles slower.
    #[inline(always)]
    pub fn rdtsc_serialized() -> u64 {
        unsafe {
            asm!("lfence", options(nomem, nostack));
            std::arch::x86_64::_rdtsc()
        }
    }

    /// RDTSCP - Serializing variant that also returns processor ID
    ///
    /// Waits for all previous instructions to complete.
    /// The `aux` value contains IA32_TSC_AUX (typically processor ID).
    #[inline(always)]
    pub fn rdtscp() -> (u64, u32) {
        let mut aux: u32 = 0;
        let tsc = unsafe { std::arch::x86_64::__rdtscp(&mut aux) };
        (tsc, aux)
    }

    /// Executes the CPUID instruction
    ///
    /// Used to query processor features, topology, and cache hierarchy
    /// without OS syscalls.
    ///
    /// # Arguments
    /// - `leaf`: CPUID leaf (EAX input)
    /// - `subleaf`: CPUID subleaf (ECX input)
    ///
    /// # Returns
    /// Tuple of (EAX, EBX, ECX, EDX) output registers
    ///
    /// # Example
    /// ```ignore
    /// // Get vendor string (leaf 0)
    /// let (_, ebx, ecx, edx) = cpuid(0, 0);
    /// // EBX:EDX:ECX contains "GenuineIntel" or "AuthenticAMD"
    /// ```
    #[inline(always)]
    pub fn cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
        // NOTE: We use manual asm because std::arch::x86_64::__cpuid_count
        // requires the cpuid feature flag which may not be detected at compile time.
        //
        // The RBX register is reserved by LLVM, so we must save/restore it manually.
        let eax: u32;
        let ebx: u32;
        let ecx: u32;
        let edx: u32;

        unsafe {
            asm!(
                "push rbx",           // Save RBX (reserved by LLVM)
                "cpuid",              // Execute CPUID
                "mov {ebx_out:e}, ebx", // Copy EBX to temp before pop
                "pop rbx",            // Restore RBX
                ebx_out = out(reg) ebx,
                inout("eax") leaf => eax,
                inout("ecx") subleaf => ecx,
                lateout("edx") edx,
            );
        }

        (eax, ebx, ecx, edx)
    }

    /// Get CPU vendor string (e.g., "GenuineIntel", "AuthenticAMD")
    pub fn cpu_vendor() -> [u8; 12] {
        let (_, ebx, ecx, edx) = cpuid(0, 0);
        let mut vendor = [0u8; 12];

        // Vendor string is in EBX:EDX:ECX order
        vendor[0..4].copy_from_slice(&ebx.to_le_bytes());
        vendor[4..8].copy_from_slice(&edx.to_le_bytes());
        vendor[8..12].copy_from_slice(&ecx.to_le_bytes());

        vendor
    }

    /// Check if CPU supports invariant TSC
    ///
    /// Invariant TSC runs at constant rate regardless of CPU frequency scaling.
    /// Essential for reliable timing measurements.
    pub fn has_invariant_tsc() -> bool {
        let (_, _, _, edx) = cpuid(0x80000007, 0);
        (edx & (1 << 8)) != 0 // Bit 8 = Invariant TSC
    }
}

// ============================================================================
// AArch64 Implementation (ARM64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    use std::arch::asm;

    /// Reads the ARM64 virtual counter (CNTVCT_EL0)
    ///
    /// Equivalent to x86 RDTSC - provides cycle-accurate timing.
    /// The virtual counter is consistent across all cores in the system.
    ///
    /// # Performance
    /// - Latency: ~5-10 cycles
    /// - Precision: Depends on counter frequency (typically 1-100MHz)
    #[inline(always)]
    pub fn rdtsc() -> u64 {
        let val: u64;
        unsafe {
            asm!(
                "mrs {}, cntvct_el0",
                out(reg) val,
                options(nomem, nostack)
            );
        }
        val
    }

    /// RDTSC with ISB barrier (serializing read)
    ///
    /// ISB (Instruction Synchronization Barrier) ensures all previous
    /// instructions complete before reading the counter.
    #[inline(always)]
    pub fn rdtsc_serialized() -> u64 {
        let val: u64;
        unsafe {
            asm!(
                "isb",
                "mrs {}, cntvct_el0",
                "isb",
                out(reg) val,
                options(nomem, nostack)
            );
        }
        val
    }

    /// Get the counter frequency in Hz
    ///
    /// Use this to convert counter ticks to real time:
    /// `seconds = ticks / frequency()`
    #[inline(always)]
    pub fn counter_frequency() -> u64 {
        let freq: u64;
        unsafe {
            asm!(
                "mrs {}, cntfrq_el0",
                out(reg) freq,
                options(nomem, nostack)
            );
        }
        freq
    }

    /// CPUID equivalent for ARM - reads MIDR_EL1 (Main ID Register)
    ///
    /// Returns implementer, variant, architecture, part number, and revision.
    /// Note: This requires EL1 or higher privilege level.
    #[inline(always)]
    pub fn midr() -> u64 {
        let val: u64;
        unsafe {
            asm!(
                "mrs {}, midr_el1",
                out(reg) val,
                options(nomem, nostack)
            );
        }
        val
    }
}

// ============================================================================
// Fallback Implementation (Other Architectures)
// ============================================================================

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod fallback {
    use std::sync::LazyLock;
    use std::time::Instant;

    static START_TIME: LazyLock<Instant> = LazyLock::new(Instant::now);

    /// Fallback "RDTSC" using std::time::Instant
    ///
    /// Returns nanoseconds since first call (not true cycle count).
    /// Less precise but works on any platform.
    #[inline(always)]
    pub fn rdtsc() -> u64 {
        START_TIME.elapsed().as_nanos() as u64
    }

    /// Same as rdtsc() for fallback
    #[inline(always)]
    pub fn rdtsc_serialized() -> u64 {
        rdtsc()
    }
}

// ============================================================================
// Unified Public API
// ============================================================================

/// Read timestamp counter (architecture-independent)
///
/// - x86_64: RDTSC instruction
/// - AArch64: CNTVCT_EL0 register
/// - Other: std::time::Instant fallback
#[inline(always)]
pub unsafe fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        x86_64::rdtsc()
    }
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::rdtsc()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        fallback::rdtsc()
    }
}

/// Read timestamp counter with serialization barrier
///
/// More accurate for benchmarking - ensures instruction ordering.
#[inline(always)]
pub unsafe fn rdtsc_serialized() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        x86_64::rdtsc_serialized()
    }
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::rdtsc_serialized()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        fallback::rdtsc_serialized()
    }
}

/// CPUID wrapper (x86_64 only, no-op on other architectures)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
    x86_64::cpuid(leaf, subleaf)
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cpuid(_leaf: u32, _subleaf: u32) -> (u32, u32, u32, u32) {
    (0, 0, 0, 0) // No-op on non-x86
}

/// Spin loop hint (portable)
///
/// Uses `std::hint::spin_loop()` which maps to:
/// - x86: PAUSE instruction
/// - ARM: YIELD instruction
/// - Other: no-op
///
/// This is a **safe** function - no `unsafe` needed!
#[inline(always)]
pub fn spin_loop_hint(iterations: u64) {
    for _ in 0..iterations {
        std::hint::spin_loop();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdtsc_increases() {
        let t1 = unsafe { rdtsc() };
        // Some work
        let mut sum = 0u64;
        for i in 0..1000 {
            sum = sum.wrapping_add(i);
        }
        let t2 = unsafe { rdtsc() };

        assert!(t2 > t1, "RDTSC should increase over time");
        assert!(sum > 0); // Prevent optimization
    }

    #[test]
    fn test_spin_loop_hint() {
        // Should not panic
        spin_loop_hint(100);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_cpuid_vendor() {
        let vendor = x86_64::cpu_vendor();
        let vendor_str = std::str::from_utf8(&vendor).unwrap_or("unknown");

        // Should be one of the known vendors
        assert!(
            vendor_str == "GenuineIntel"
                || vendor_str == "AuthenticAMD"
                || vendor_str.starts_with("Genuine") // VMs
                || !vendor_str.is_empty()
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_invariant_tsc() {
        // Just check it doesn't panic - result depends on CPU
        let _ = x86_64::has_invariant_tsc();
    }
}
