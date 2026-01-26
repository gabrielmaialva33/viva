//// Integrations - Unified VIVA Subsystems Hub
////
//// "The whole is greater than the sum of its parts."
////
//// This module re-exports all VIVA subsystems for unified access.
//// Use the individual modules for full API - this provides shortcuts.

// =============================================================================
// GLANDS - Neural Endocrine System (LLM → HRR)
// =============================================================================

import viva/glands

/// Default glands config
pub fn glands_default() {
  glands.default_config()
}

/// Small glands config for testing
pub fn glands_small() {
  glands.small_config()
}

/// Qwen-specific glands config
pub fn glands_qwen() {
  glands.qwen_config()
}

/// Initialize glands system
pub fn init_glands(config) {
  glands.init(config)
}

/// Bind two HRR vectors
pub fn hrr_bind(a, b) {
  glands.bind(a, b)
}

/// Unbind HRR vectors
pub fn hrr_unbind(bound, key) {
  glands.unbind(bound, key)
}

/// Similarity between HRR vectors
pub fn hrr_similarity(a, b) {
  glands.similarity(a, b)
}

/// Superpose HRR vectors
pub fn hrr_superpose(vectors) {
  glands.superpose(vectors)
}

/// Check glands status
pub fn glands_check() {
  glands.check()
}

// =============================================================================
// SENSES - Multimodal Perception
// =============================================================================

import viva/embodied/senses
import viva/senses/perceiver
import viva/senses/vision

/// Perceive an image
pub fn see(path) {
  senses.perceive_image(path)
}

/// Perceive audio
pub fn hear(path) {
  senses.perceive_audio(path)
}

/// Internal contemplation
pub fn think(content) {
  senses.contemplate(content)
}

/// Quick vision
pub fn quick_see(path) {
  senses.quick_see(path)
}

/// Quick OCR
pub fn quick_read(path) {
  senses.quick_read(path)
}

/// Quick audio
pub fn quick_hear(path) {
  senses.quick_hear(path)
}

/// Quick thought
pub fn quick_think(prompt) {
  senses.quick_think(prompt)
}

/// Perceive multiple images
pub fn see_many(paths) {
  senses.perceive_images(paths)
}

/// Most salient perception
pub fn see_salient(paths) {
  senses.perceive_most_salient(paths)
}

/// Start perceiver actor
pub fn start_perceiver() {
  perceiver.start()
}

/// Poke curiosity
pub fn poke_curiosity(p, amount) {
  perceiver.poke_curiosity(p, amount)
}

/// Force look
pub fn look_now(p) {
  perceiver.look_now(p)
}

/// Ask perceiver
pub fn ask_perceiver(p, question) {
  perceiver.ask(p, question)
}

/// Last percept
pub fn last_percept(p) {
  perceiver.last_percept(p)
}

/// Understand image with vision
pub fn understand(request, prompt) {
  vision.understand(request, prompt)
}

// =============================================================================
// HARDWARE - Body Communication
// =============================================================================

import viva/codegen/arduino_gen
import viva/hardware/port_manager

/// Start body port manager
pub fn start_body(device, baud) {
  port_manager.start(device, baud)
}

/// Send to body
pub fn send_body(manager, pkt) {
  port_manager.send(manager, pkt)
}

/// Subscribe to body
pub fn subscribe_body(manager, subscriber) {
  port_manager.subscribe(manager, subscriber)
}

/// Unsubscribe from body
pub fn unsubscribe_body(manager, subscriber) {
  port_manager.unsubscribe(manager, subscriber)
}

/// Body stats
pub fn body_stats(manager) {
  port_manager.get_stats(manager)
}

/// Shutdown body
pub fn shutdown_body(manager) {
  port_manager.shutdown(manager)
}

/// Generate Arduino protocol
pub fn codegen_arduino() {
  arduino_gen.main()
}

// =============================================================================
// NEURAL - Advanced Operations
// =============================================================================

import viva/neural/named_tensor
import viva/neural/network_accelerated
import viva/neural/serialize

/// Create batch axis
pub fn batch(size) {
  named_tensor.batch(size)
}

/// Create sequence axis
pub fn seq(size) {
  named_tensor.seq(size)
}

/// Create feature axis
pub fn feature(size) {
  named_tensor.feature(size)
}

/// Create height axis
pub fn height(size) {
  named_tensor.height(size)
}

/// Create width axis
pub fn width(size) {
  named_tensor.width(size)
}

/// Create input axis
pub fn input(size) {
  named_tensor.input(size)
}

/// Create output axis
pub fn output(size) {
  named_tensor.output(size)
}

/// Create named tensor zeros
pub fn named_zeros(axes) {
  named_tensor.zeros(axes)
}

/// Create named tensor ones
pub fn named_ones(axes) {
  named_tensor.ones(axes)
}

/// Create named tensor random
pub fn named_random(axes) {
  named_tensor.random(axes)
}

/// Auto-detect neural backend
pub fn auto_backend() {
  network_accelerated.auto_backend()
}

/// Forward pass
pub fn forward(network, input) {
  network_accelerated.forward(network, input)
}

/// Serialize network to JSON
pub fn network_to_json(network) {
  serialize.network_to_json(network)
}

/// Load network from JSON
pub fn network_from_json(json) {
  serialize.network_from_string(json)
}

// =============================================================================
// STATUS
// =============================================================================

/// Check all subsystems
pub fn status() {
  let glands_status = glands.check()
  "VIVA Integrations:\n"
  <> "- Glands: "
  <> glands_status
  <> "\n"
  <> "- Senses: Ready\n"
  <> "- Hardware: Ready\n"
  <> "- Neural: Ready"
}
