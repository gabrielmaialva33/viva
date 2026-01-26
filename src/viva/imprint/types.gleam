//// Imprint Types - Shared types for imprinting system
////
//// Separated to avoid import cycles between imprint and its submodules.

// =============================================================================
// TYPES
// =============================================================================

/// Events emitted during imprinting
pub type ImprintEvent {
  /// New sensory association learned
  SensoryLearned(stimulus_type: String, valence: Float)
  /// New motor pattern discovered
  MotorLearned(action: String, effect: String)
  /// Social attachment changed
  AttachmentChanged(entity: String, strength: Float)
  /// Danger signal learned
  DangerLearned(trigger: String, intensity: Float)
  /// Safety signal learned
  SafetyLearned(trigger: String, comfort: Float)
  /// Critical period ended
  CriticalPeriodEnded(total_learned: Int)
}
