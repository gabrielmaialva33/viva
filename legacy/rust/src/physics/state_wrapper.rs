//! StateWrapper - Bidirectional state serialization (DuckStation pattern)
//!
//! Single `Do()` method for both save and load eliminates sync bugs.
//! Instead of separate `save_field()` and `load_field()` that can diverge,
//! we have one unified method that handles both directions.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};

/// Mode of state operation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateMode {
    /// Writing state to buffer (snapshot)
    Save,
    /// Reading state from buffer (restore)
    Load,
}

/// Bidirectional state wrapper for deterministic snapshot/rollback
///
/// # DuckStation Pattern
/// ```cpp
/// void Do(T& data) {
///     if (m_mode == Read) DoRead(&data); else DoWrite(&data);
/// }
/// ```
///
/// # Rust Implementation
/// ```ignore
/// fn freeze_thaw(&mut self, state: &mut StateWrapper) {
///     state.do_val(&mut self.position);
///     state.do_val(&mut self.velocity);
///     state.do_val(&mut self.rotation);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct StateWrapper {
    mode: StateMode,
    buffer: Vec<u8>,
    cursor: usize,
}

impl StateWrapper {
    /// Create a new wrapper in Save mode (empty buffer)
    pub fn for_save() -> Self {
        Self {
            mode: StateMode::Save,
            buffer: Vec::with_capacity(4096), // Pre-allocate typical size
            cursor: 0,
        }
    }

    /// Create a new wrapper in Load mode (from existing buffer)
    pub fn for_load(buffer: Vec<u8>) -> Self {
        Self {
            mode: StateMode::Load,
            buffer,
            cursor: 0,
        }
    }

    /// Get current mode
    #[inline]
    pub fn mode(&self) -> StateMode {
        self.mode
    }

    /// Get buffer (for storage after save)
    pub fn into_buffer(self) -> Vec<u8> {
        self.buffer
    }

    /// Get buffer reference
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Get current cursor position
    #[inline]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Reset cursor to beginning (for re-reading)
    pub fn reset_cursor(&mut self) {
        self.cursor = 0;
    }

    // =========================================================================
    // Core Do API (bidirectional)
    // =========================================================================

    /// Bidirectional value transfer for Pod types
    ///
    /// In Save mode: copies value to buffer
    /// In Load mode: copies buffer to value
    ///
    /// Uses manual copy to avoid alignment requirements from bytemuck::from_bytes
    #[inline]
    pub fn do_val<T: Pod>(&mut self, val: &mut T) {
        match self.mode {
            StateMode::Save => {
                let bytes = bytemuck::bytes_of(val);
                self.buffer.extend_from_slice(bytes);
                self.cursor += bytes.len();
            }
            StateMode::Load => {
                let size = std::mem::size_of::<T>();
                if self.cursor + size <= self.buffer.len() {
                    // Use ptr::copy_nonoverlapping to avoid alignment issues
                    // Safety: We're copying raw bytes into a Pod type
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            self.buffer.as_ptr().add(self.cursor),
                            val as *mut T as *mut u8,
                            size,
                        );
                    }
                    self.cursor += size;
                }
            }
        }
    }

    /// Bidirectional array transfer for Pod types
    #[inline]
    pub fn do_array<T: Pod>(&mut self, arr: &mut [T]) {
        match self.mode {
            StateMode::Save => {
                let bytes = bytemuck::cast_slice(arr);
                self.buffer.extend_from_slice(bytes);
                self.cursor += bytes.len();
            }
            StateMode::Load => {
                let size = std::mem::size_of_val(arr);
                if self.cursor + size <= self.buffer.len() {
                    let bytes = &self.buffer[self.cursor..self.cursor + size];
                    let src: &[T] = bytemuck::cast_slice(bytes);
                    arr.copy_from_slice(src);
                    self.cursor += size;
                }
            }
        }
    }

    /// Bidirectional Vec transfer (with length prefix)
    pub fn do_vec<T: Pod + Clone>(&mut self, vec: &mut Vec<T>) {
        match self.mode {
            StateMode::Save => {
                let len = vec.len() as u32;
                self.do_val(&mut { len });
                if !vec.is_empty() {
                    let bytes = bytemuck::cast_slice(vec.as_slice());
                    self.buffer.extend_from_slice(bytes);
                    self.cursor += bytes.len();
                }
            }
            StateMode::Load => {
                let mut len: u32 = 0;
                self.do_val(&mut len);
                let count = len as usize;
                let elem_size = std::mem::size_of::<T>();
                let total_size = count * elem_size;

                if self.cursor + total_size <= self.buffer.len() {
                    vec.clear();
                    vec.reserve(count);
                    let bytes = &self.buffer[self.cursor..self.cursor + total_size];
                    let src: &[T] = bytemuck::cast_slice(bytes);
                    vec.extend_from_slice(src);
                    self.cursor += total_size;
                }
            }
        }
    }

    // =========================================================================
    // Convenience methods for common types
    // =========================================================================

    /// Vec3 (12 bytes)
    #[inline]
    pub fn do_vec3(&mut self, v: &mut Vec3) {
        self.do_val(&mut v.x);
        self.do_val(&mut v.y);
        self.do_val(&mut v.z);
    }

    /// Quat (16 bytes)
    #[inline]
    pub fn do_quat(&mut self, q: &mut Quat) {
        self.do_val(&mut q.x);
        self.do_val(&mut q.y);
        self.do_val(&mut q.z);
        self.do_val(&mut q.w);
    }

    /// Compressed Quat (8 bytes) - "smallest three" encoding
    ///
    /// Stores only 3 components + index of largest, reconstructs 4th.
    /// Precision: ~0.0001 per component (good enough for physics)
    pub fn do_quat_compressed(&mut self, q: &mut Quat) {
        match self.mode {
            StateMode::Save => {
                let compressed = compress_quat(*q);
                self.buffer.extend_from_slice(&compressed);
                self.cursor += 8;
            }
            StateMode::Load => {
                if self.cursor + 8 <= self.buffer.len() {
                    let bytes: [u8; 8] = self.buffer[self.cursor..self.cursor + 8]
                        .try_into()
                        .unwrap();
                    *q = decompress_quat(bytes);
                    self.cursor += 8;
                }
            }
        }
    }

    /// Bool (1 byte)
    #[inline]
    pub fn do_bool(&mut self, b: &mut bool) {
        match self.mode {
            StateMode::Save => {
                self.buffer.push(if *b { 1 } else { 0 });
                self.cursor += 1;
            }
            StateMode::Load => {
                if self.cursor < self.buffer.len() {
                    *b = self.buffer[self.cursor] != 0;
                    self.cursor += 1;
                }
            }
        }
    }

    /// Option<f32> specialized (avoids alignment issues with generic Pod)
    pub fn do_option_f32(&mut self, opt: &mut Option<f32>) {
        match self.mode {
            StateMode::Save => {
                let has_value = opt.is_some();
                self.do_bool(&mut { has_value });
                if let Some(val) = opt {
                    self.do_val(val);
                }
            }
            StateMode::Load => {
                let mut has_value = false;
                self.do_bool(&mut has_value);
                if has_value {
                    let mut val: f32 = 0.0;
                    self.do_val(&mut val);
                    *opt = Some(val);
                } else {
                    *opt = None;
                }
            }
        }
    }

    /// Option<u32> specialized
    pub fn do_option_u32(&mut self, opt: &mut Option<u32>) {
        match self.mode {
            StateMode::Save => {
                let has_value = opt.is_some();
                self.do_bool(&mut { has_value });
                if let Some(val) = opt {
                    self.do_val(val);
                }
            }
            StateMode::Load => {
                let mut has_value = false;
                self.do_bool(&mut has_value);
                if has_value {
                    let mut val: u32 = 0;
                    self.do_val(&mut val);
                    *opt = Some(val);
                } else {
                    *opt = None;
                }
            }
        }
    }
}

// ============================================================================
// Quaternion Compression (smallest three encoding)
// ============================================================================

/// Compress quaternion to 8 bytes using "smallest three" encoding
///
/// Format:
/// - 2 bits: index of largest component (dropped)
/// - 20 bits each: 3 remaining components (scaled to [-1, 1] → [0, 2^20-1])
fn compress_quat(q: Quat) -> [u8; 8] {
    let q = q.normalize();
    let arr = [q.x, q.y, q.z, q.w];

    // Find largest component
    let mut max_idx = 0;
    let mut max_val = arr[0].abs();
    for (i, &v) in arr.iter().enumerate().skip(1) {
        if v.abs() > max_val {
            max_idx = i;
            max_val = v.abs();
        }
    }

    // Ensure largest is positive (we can flip the whole quat, -q = q in rotation space)
    let sign = if arr[max_idx] < 0.0 { -1.0 } else { 1.0 };

    // Get the three smaller components, scaled to [0, 1]
    let mut small = [0.0f32; 3];
    let mut j = 0;
    for (i, &v) in arr.iter().enumerate() {
        if i != max_idx {
            // v is in [-1, 1], scale to [0, 1]
            small[j] = (v * sign + 1.0) * 0.5;
            j += 1;
        }
    }

    // Encode: 2 bits for index, 20 bits each for 3 values = 62 bits
    let max_val_20bit = (1 << 20) - 1; // 1048575
    let a = (small[0].clamp(0.0, 1.0) * max_val_20bit as f32) as u64;
    let b = (small[1].clamp(0.0, 1.0) * max_val_20bit as f32) as u64;
    let c = (small[2].clamp(0.0, 1.0) * max_val_20bit as f32) as u64;
    let packed = ((max_idx as u64) << 60) | (a << 40) | (b << 20) | c;

    packed.to_le_bytes()
}

/// Decompress quaternion from 8 bytes
fn decompress_quat(bytes: [u8; 8]) -> Quat {
    let packed = u64::from_le_bytes(bytes);

    let max_idx = ((packed >> 60) & 0b11) as usize;
    let max_val_20bit = (1 << 20) - 1;
    let a = ((packed >> 40) & 0xFFFFF) as f32 / max_val_20bit as f32;
    let b = ((packed >> 20) & 0xFFFFF) as f32 / max_val_20bit as f32;
    let c = (packed & 0xFFFFF) as f32 / max_val_20bit as f32;

    // Decode from [0, 1] back to [-1, 1]
    let small = [a * 2.0 - 1.0, b * 2.0 - 1.0, c * 2.0 - 1.0];

    // Reconstruct 4th component: w² = 1 - x² - y² - z²
    let sum_sq: f32 = small.iter().map(|x| x * x).sum();
    let largest = (1.0 - sum_sq).max(0.0).sqrt();

    // Rebuild quaternion
    let mut arr = [0.0f32; 4];
    let mut j = 0;
    for i in 0..4 {
        if i == max_idx {
            arr[i] = largest;
        } else {
            arr[i] = small[j];
            j += 1;
        }
    }

    Quat::from_xyzw(arr[0], arr[1], arr[2], arr[3]).normalize()
}

// ============================================================================
// FreezeThaw trait for types that can be serialized
// ============================================================================

/// Trait for types that can save/load their state bidirectionally
pub trait FreezeThaw {
    /// Serialize or deserialize state depending on wrapper mode
    fn freeze_thaw(&mut self, state: &mut StateWrapper);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_roundtrip() {
        let mut x: f32 = 42.5;
        let mut y: i32 = -123;
        let mut z: u64 = 999999;

        // Save
        let mut save = StateWrapper::for_save();
        save.do_val(&mut x);
        save.do_val(&mut y);
        save.do_val(&mut z);

        let buffer = save.into_buffer();
        assert_eq!(buffer.len(), 4 + 4 + 8); // f32 + i32 + u64

        // Load
        let mut load = StateWrapper::for_load(buffer);
        let mut x2: f32 = 0.0;
        let mut y2: i32 = 0;
        let mut z2: u64 = 0;
        load.do_val(&mut x2);
        load.do_val(&mut y2);
        load.do_val(&mut z2);

        assert_eq!(x, x2);
        assert_eq!(y, y2);
        assert_eq!(z, z2);
    }

    #[test]
    fn test_vec3_roundtrip() {
        let mut v = Vec3::new(1.0, 2.0, 3.0);

        // Save
        let mut save = StateWrapper::for_save();
        save.do_vec3(&mut v);

        let buffer = save.into_buffer();

        // Load
        let mut load = StateWrapper::for_load(buffer);
        let mut v2 = Vec3::ZERO;
        load.do_vec3(&mut v2);

        assert_eq!(v, v2);
    }

    #[test]
    fn test_quat_roundtrip() {
        let mut q = Quat::from_euler(glam::EulerRot::XYZ, 0.5, 1.0, 0.3);

        // Save
        let mut save = StateWrapper::for_save();
        save.do_quat(&mut q);

        let buffer = save.into_buffer();
        assert_eq!(buffer.len(), 16); // 4 * f32

        // Load
        let mut load = StateWrapper::for_load(buffer);
        let mut q2 = Quat::IDENTITY;
        load.do_quat(&mut q2);

        assert!((q - q2).length() < 0.0001);
    }

    #[test]
    fn test_quat_compressed() {
        let original = Quat::from_euler(glam::EulerRot::XYZ, 0.5, 1.0, 0.3);
        let mut q = original;

        // Save compressed
        let mut save = StateWrapper::for_save();
        save.do_quat_compressed(&mut q);

        let buffer = save.into_buffer();
        assert_eq!(buffer.len(), 8); // 50% compression!

        // Load
        let mut load = StateWrapper::for_load(buffer);
        let mut q2 = Quat::IDENTITY;
        load.do_quat_compressed(&mut q2);

        // Should be close (not exact due to compression)
        let dot = original.dot(q2).abs();
        assert!(dot > 0.999, "Compressed quat should be very close: dot={}", dot);
    }

    #[test]
    fn test_bool_roundtrip() {
        let mut a = true;
        let mut b = false;

        // Save
        let mut save = StateWrapper::for_save();
        save.do_bool(&mut a);
        save.do_bool(&mut b);

        let buffer = save.into_buffer();

        // Load
        let mut load = StateWrapper::for_load(buffer);
        let mut a2 = false;
        let mut b2 = true;
        load.do_bool(&mut a2);
        load.do_bool(&mut b2);

        assert_eq!(a, a2);
        assert_eq!(b, b2);
    }

    #[test]
    fn test_vec_roundtrip() {
        let mut v: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Save
        let mut save = StateWrapper::for_save();
        save.do_vec(&mut v);

        let buffer = save.into_buffer();

        // Load
        let mut load = StateWrapper::for_load(buffer);
        let mut v2: Vec<f32> = Vec::new();
        load.do_vec(&mut v2);

        assert_eq!(v, v2);
    }

    #[test]
    fn test_option_roundtrip() {
        let mut some: Option<f32> = Some(42.5);
        let mut none: Option<f32> = None;

        // Save
        let mut save = StateWrapper::for_save();
        save.do_option_f32(&mut some);
        save.do_option_f32(&mut none);

        let buffer = save.into_buffer();

        // Load
        let mut load = StateWrapper::for_load(buffer);
        let mut some2: Option<f32> = None;
        let mut none2: Option<f32> = Some(0.0);
        load.do_option_f32(&mut some2);
        load.do_option_f32(&mut none2);

        assert_eq!(some, some2);
        assert_eq!(none, none2);
    }

    #[test]
    fn test_freeze_thaw_trait() {
        // Example struct implementing FreezeThaw
        struct RigidBody {
            position: Vec3,
            velocity: Vec3,
            mass: f32,
            awake: bool,
        }

        impl FreezeThaw for RigidBody {
            fn freeze_thaw(&mut self, state: &mut StateWrapper) {
                state.do_vec3(&mut self.position);
                state.do_vec3(&mut self.velocity);
                state.do_val(&mut self.mass);
                state.do_bool(&mut self.awake);
            }
        }

        let mut body = RigidBody {
            position: Vec3::new(1.0, 2.0, 3.0),
            velocity: Vec3::new(0.1, 0.2, 0.3),
            mass: 10.0,
            awake: true,
        };

        // Save
        let mut save = StateWrapper::for_save();
        body.freeze_thaw(&mut save);
        let buffer = save.into_buffer();

        // Load into new body
        let mut load = StateWrapper::for_load(buffer);
        let mut body2 = RigidBody {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            mass: 0.0,
            awake: false,
        };
        body2.freeze_thaw(&mut load);

        assert_eq!(body.position, body2.position);
        assert_eq!(body.velocity, body2.velocity);
        assert_eq!(body.mass, body2.mass);
        assert_eq!(body.awake, body2.awake);
    }
}
