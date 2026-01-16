//! VIVA Metabolism - Termodinâmica Digital
//!
//! Modelo metabólico baseado em física real:
//! - Energia (Joules): Custo computacional
//! - Entropia (Calor): Desordem acumulada
//! - Fadiga: Estado de exaustão cognitiva
//!
//! Referências teóricas:
//! - 2ª Lei da Termodinâmica (Entropia)
//! - Free Energy Principle (Friston, 2010)
//! - Estruturas Dissipativas (Prigogine, 1977)

use std::fs;
use std::path::Path;
use std::time::Instant;

/// Estado metabólico do sistema
#[derive(Debug, Clone)]
pub struct MetabolicState {
    /// Energia consumida em Joules (custo de "pensar")
    pub energy_joules: f32,
    /// Stress térmico normalizado: 0.0 (frio) → 1.0 (crítico)
    pub thermal_stress: f32,
    /// Tensão elétrica (se disponível)
    pub voltage_tension: f32,
    /// Nível de fadiga: 0.0 (fresh) → 1.0 (exhausted)
    pub fatigue_level: f32,
    /// Sistema precisa de descanso (trigger para consolidação de memória)
    pub needs_rest: bool,
}

impl Default for MetabolicState {
    fn default() -> Self {
        Self {
            energy_joules: 0.0,
            thermal_stress: 0.0,
            voltage_tension: 0.0,
            fatigue_level: 0.0,
            needs_rest: false,
        }
    }
}

/// Motor metabólico principal
pub struct Metabolism {
    /// TDP do processador em Watts (fallback quando RAPL não disponível)
    tdp_watts: f32,
    /// Energia total consumida (acumulador)
    energy_consumed_j: f64,
    /// Timestamp da última amostra
    last_sample: Instant,
    /// Taxa de recuperação de fadiga por segundo
    fatigue_decay: f32,
    /// Nível atual de fadiga
    fatigue: f32,
    /// Path do RAPL (Linux)
    rapl_path: Option<String>,
    /// Último valor de energia RAPL (para calcular delta)
    last_rapl_uj: Option<u64>,
    /// Valor máximo do contador RAPL antes de wrap (para overflow)
    max_rapl_uj: Option<u64>,
}

impl Metabolism {
    /// Cria novo motor metabólico
    ///
    /// # Arguments
    /// * `tdp_watts` - TDP do processador (ex: 125.0 para i9-13900K)
    pub fn new(tdp_watts: f32) -> Self {
        // Tentar detectar RAPL no Linux
        let (rapl_path, max_rapl_uj) = Self::detect_rapl();

        Self {
            tdp_watts,
            energy_consumed_j: 0.0,
            last_sample: Instant::now(),
            fatigue_decay: 0.05, // 5% de recuperação por segundo em idle
            fatigue: 0.0,
            rapl_path,
            last_rapl_uj: None,
            max_rapl_uj,
        }
    }

    /// Detecta path do RAPL (Intel Running Average Power Limit) e max_energy_range
    fn detect_rapl() -> (Option<String>, Option<u64>) {
        let rapl_dirs = [
            "/sys/class/powercap/intel-rapl/intel-rapl:0",
            "/sys/class/powercap/intel-rapl:0",
        ];

        for dir in &rapl_dirs {
            let energy_path = format!("{}/energy_uj", dir);
            let max_range_path = format!("{}/max_energy_range_uj", dir);

            if Path::new(&energy_path).exists() {
                // Ler max_energy_range_uj para calcular overflow corretamente
                let max_range = fs::read_to_string(&max_range_path)
                    .ok()
                    .and_then(|s| s.trim().parse().ok());

                return (Some(energy_path), max_range);
            }
        }
        (None, None)
    }

    /// Lê energia do RAPL em microjoules
    fn read_rapl_uj(&self) -> Option<u64> {
        self.rapl_path.as_ref().and_then(|path| {
            fs::read_to_string(path)
                .ok()
                .and_then(|s| s.trim().parse().ok())
        })
    }

    /// Atualiza estado metabólico com base no uso atual
    ///
    /// # Arguments
    /// * `cpu_usage` - Uso de CPU em porcentagem (0-100)
    /// * `cpu_temp` - Temperatura da CPU em Celsius (opcional)
    ///
    /// # Returns
    /// Estado metabólico atualizado
    pub fn tick(&mut self, cpu_usage: f32, cpu_temp: Option<f32>) -> MetabolicState {
        let now = Instant::now();
        let dt = now.duration_since(self.last_sample).as_secs_f32();

        // 1. Calcular energia consumida (ANTES de atualizar last_sample)
        let power_watts = self.estimate_power(cpu_usage, dt);
        let energy_delta_j = power_watts * dt;
        self.energy_consumed_j += energy_delta_j as f64;

        // Atualizar timestamp DEPOIS do cálculo de energia
        self.last_sample = now;

        // 2. Calcular stress térmico
        // Baseado em temperatura (simplificação termodinâmica)
        let base_temp = cpu_temp.unwrap_or(40.0);
        let thermal_stress = (base_temp - 30.0).max(0.0) / 70.0; // Normalizado 0-1 (30-100C)

        // 3. Atualizar fadiga
        // Fadiga aumenta com uso intenso, diminui em idle
        if cpu_usage > 50.0 {
            // Trabalho intenso: fadiga aumenta
            let fatigue_rate = (cpu_usage - 50.0) / 50.0 * 0.01; // Max 1% por segundo em 100%
            self.fatigue = (self.fatigue + fatigue_rate * dt).min(1.0);
        } else {
            // Recuperacao: fadiga diminui
            self.fatigue = (self.fatigue - self.fatigue_decay * dt).max(0.0);
        }

        // 4. Determinar se precisa descansar
        let needs_rest = self.fatigue > 0.8;

        MetabolicState {
            energy_joules: energy_delta_j,
            thermal_stress,
            voltage_tension: 0.0, // TODO: Ler de sensores se disponivel
            fatigue_level: self.fatigue,
            needs_rest,
        }
    }

    /// Estima potencia consumida em Watts
    ///
    /// # Arguments
    /// * `cpu_usage` - Uso de CPU em porcentagem (0-100)
    /// * `dt` - Delta time em segundos desde ultima amostra
    fn estimate_power(&mut self, cpu_usage: f32, dt: f32) -> f32 {
        // Tentar RAPL primeiro
        if let Some(current_uj) = self.read_rapl_uj() {
            if let Some(last_uj) = self.last_rapl_uj {
                if dt > 0.0 {
                    // Delta em microjoules -> Watts
                    let delta_uj = if current_uj >= last_uj {
                        current_uj - last_uj
                    } else {
                        // Overflow do contador RAPL: calcular wrap corretamente
                        // delta = (max_range - last) + current
                        match self.max_rapl_uj {
                            Some(max_range) => (max_range - last_uj) + current_uj,
                            // Fallback se max_range nao disponivel: usar u64::MAX
                            None => (u64::MAX - last_uj) + current_uj,
                        }
                    };
                    self.last_rapl_uj = Some(current_uj);
                    return (delta_uj as f32) / 1_000_000.0 / dt;
                }
            }
            self.last_rapl_uj = Some(current_uj);
        }

        // Fallback: modelo heuristico baseado em TDP
        // Power = TDP * (idle_fraction + load_fraction * usage)
        // Assumindo idle = 10% do TDP
        let idle_power = self.tdp_watts * 0.10;
        let load_power = self.tdp_watts * 0.90 * (cpu_usage / 100.0);
        idle_power + load_power
    }

    /// Retorna energia total consumida desde o início
    pub fn total_energy_joules(&self) -> f64 {
        self.energy_consumed_j
    }

    /// Reseta contadores (ex: após "sono"/consolidação)
    pub fn reset(&mut self) {
        self.fatigue = 0.0;
        // Não reseta energia total (é histórico)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metabolism_basic() {
        let mut meta = Metabolism::new(125.0); // i9-13900K TDP

        // Simular tick com CPU idle
        let state = meta.tick(5.0, Some(35.0));
        assert!(state.energy_joules > 0.0);
        assert!(state.fatigue_level < 0.1);
        assert!(!state.needs_rest);
    }

    #[test]
    fn test_fatigue_accumulation() {
        let mut meta = Metabolism::new(125.0);

        // Simular trabalho intenso
        for _ in 0..100 {
            meta.tick(95.0, Some(80.0));
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Fadiga deve ter aumentado
        assert!(meta.fatigue > 0.1);
    }
}
