//! Raw Assembly Module
//!
//! "Close to the metal" operations using inline assembly.
//! Bypassing OS abstractions for direct CPU communication.

#![allow(dead_code)]

use std::arch::asm;

/// Reads the hardware Time Stamp Counter (RDTSC)
/// Returns the number of CPU cycles since reset.
///
/// Instruction: `rdtsc`
/// Latency: ~20-30 cycles
/// Precision: Single cycle
#[inline(always)]
pub unsafe fn rdtsc() -> u64 {
    let low: u32;
    let high: u32;

    // RDTSC stores result in EDX:EAX
    // nomem: doesn't read/write memory
    // nostack: doesn't touch stack
    asm!(
        "rdtsc",
        out("eax") low,
        out("edx") high,
        options(nomem, nostack)
    );

    ((high as u64) << 32) | (low as u64)
}

/// Executes the CPUID instruction directly
///
/// Used to query processor features, topology, and cache hierarchy
/// without OS syscalls.
#[inline(always)]
pub unsafe fn cpuid(function: u32, subfunction: u32) -> (u32, u32, u32, u32) {
    let eax: u32;
    let ebx: u32;
    let ecx: u32;
    let edx: u32;

    // CPUID writes to EAX, EBX, ECX, EDX
    // NOTE: On x86_64, RBX is reserved by LLVM. We must preserve it manually.
    // However, Rust's asm! usually handles clobbers if we don't specify it as input/output?
    // Wait, we WANT the value of EBX.
    // We must use a temporary register and move it.

    let ebx_tmp: u32;

    asm!(
        "push rbx",       // Save RBX
        "cpuid",          // Execute CPUID (clobbers EAX, EBX, ECX, EDX)
        "mov {0:e}, ebx", // Save result EBX into temp reg
        "pop rbx",        // Restore RBX
        out(reg) ebx_tmp, // Output the temp value
        inout("eax") function => eax,
        inout("ecx") subfunction => ecx,
        lateout("edx") edx,
        // We handle EBX manually
    );

    ebx = ebx_tmp;

    (eax, ebx, ecx, edx)
}

/// A "busy wait" loop implemented in assembly to minimize power variance
/// compared to OS `sleep()`. Used for precise micro-timing benchmarks.
///
/// Uses `pause` instruction (monitor/mwait hint) to be HyperThreading friendly.
pub unsafe fn spin_loop_hint(approx_cycles: u64) {
    let mut i = approx_cycles;
    while i > 0 {
        asm!("pause", options(nomem, nostack));
        i -= 1;
    }
}
