use std::collections::VecDeque;

/// Tamanho do buffer de memória de curto prazo (em ticks de 500ms)
/// 20 ticks = 10 segundos de histórico
const HISTORY_SIZE: usize = 20;

/// Mantém o histórico temporal para análise de ritmo biológico
pub struct BioRhythm {
    cpu_history: VecDeque<f32>,
    ctx_switch_history: VecDeque<u64>,
}

impl BioRhythm {
    pub fn new() -> Self {
        Self {
            cpu_history: VecDeque::with_capacity(HISTORY_SIZE),
            ctx_switch_history: VecDeque::with_capacity(HISTORY_SIZE),
        }
    }

    /// Atualiza o estado com novos dados
    pub fn update(&mut self, cpu_usage: f32, ctx_switches: u64) {
        if self.cpu_history.len() >= HISTORY_SIZE {
            self.cpu_history.pop_front();
        }
        self.cpu_history.push_back(cpu_usage);

        if self.ctx_switch_history.len() >= HISTORY_SIZE {
            self.ctx_switch_history.pop_front();
        }
        self.ctx_switch_history.push_back(ctx_switches);
    }

    /// Calcula a entropia de Shannon (Caos vs Ordem) do uso de CPU
    /// Retorna 0.0 (Ordem total) a 1.0 (Caos total)
    pub fn cpu_entropy(&self) -> f32 {
        if self.cpu_history.len() < 2 {
            return 0.0;
        }

        // 1. Normalizar valores para distribuição de probabilidade
        // Criamos histograma de 10 bins (0-10%, 10-20%...)
        let mut bins = [0.0f32; 10];
        let total = self.cpu_history.len() as f32;

        for &val in &self.cpu_history {
            let idx = (val / 10.0).floor().clamp(0.0, 9.0) as usize;
            bins[idx] += 1.0;
        }

        // 2. Calcular Shannon Entropy: H = -sum(p * log2(p))
        let mut entropy = 0.0;
        for &count in &bins {
            if count > 0.0 {
                let p = count / total;
                entropy -= p * p.log2();
            }
        }

        // Normalizar pelo máximo possível (log2(10 bins) ≈ 3.32)
        (entropy / 3.32).clamp(0.0, 1.0)
    }

    /// Calcula o "Jitter" (Desvio Padrão) dos Context Switches
    /// Indica a instabilidade do sistema operacional
    pub fn context_switch_jitter(&self) -> f32 {
        if self.ctx_switch_history.len() < 2 {
            return 0.0;
        }

        // Precisamos dos deltas, não dos valores absolutos
        let mut deltas = Vec::new();
        let mut iter = self.ctx_switch_history.iter();
        if let Some(mut last) = iter.next() {
            for val in iter {
                // Delta simples (pode ser negativo se counter resetar, mas improvável aqui)
                let delta = if val >= last { val - last } else { 0 };
                deltas.push(delta as f32);
                last = val;
            }
        }

        if deltas.is_empty() {
            return 0.0;
        }

        // Média
        let mean: f32 = deltas.iter().sum::<f32>() / deltas.len() as f32;

        // Variância
        let variance: f32 = deltas.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / deltas.len() as f32;

        // Desvio padrão normalizado (Coefficient of Variation)
        // Se mean for muito baixo, jitter pode explodir, então clampamos
        if mean > 1.0 {
            (variance.sqrt() / mean).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}
