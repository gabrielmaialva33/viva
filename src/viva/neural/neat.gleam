//// NEAT - NeuroEvolution of Augmenting Topologies
////
//// Implementação em Pure Gleam do algoritmo NEAT (Stanley & Miikkulainen, 2002).
//// Permite evolução simultânea de topologia e pesos de redes neurais.
////
//// Filosofia VIVA: Souls evoluem através de gerações, comportamentos
//// emergem via seleção natural, morte tem propósito (seleção).
////
//// Referência: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}

// =============================================================================
// TYPES - Estruturas de Dados NEAT
// =============================================================================

/// Tipo de nó na rede neural
pub type NodeType {
  Input
  Hidden
  Output
  Bias
}

/// Gene de nó - representa um neurônio
pub type NodeGene {
  NodeGene(id: Int, node_type: NodeType, activation: ActivationType)
}

/// Tipo de ativação para nós
pub type ActivationType {
  Sigmoid
  Tanh
  ReLU
  Linear
}

/// Gene de conexão - representa uma sinapse
pub type ConnectionGene {
  ConnectionGene(
    in_node: Int,
    out_node: Int,
    weight: Float,
    enabled: Bool,
    innovation: Int,
  )
}

/// Genoma completo - representa uma rede neural evoluível
pub type Genome {
  Genome(
    id: Int,
    nodes: List(NodeGene),
    connections: List(ConnectionGene),
    fitness: Float,
    adjusted_fitness: Float,
    species_id: Int,
  )
}

/// Espécie - grupo de genomas similares
pub type Species {
  Species(
    id: Int,
    members: List(Genome),
    representative: Genome,
    best_fitness: Float,
    stagnation: Int,
  )
}

/// População de genomas
pub type Population {
  Population(
    genomes: List(Genome),
    species: List(Species),
    generation: Int,
    innovation_counter: Int,
    node_counter: Int,
    innovation_history: Dict(#(Int, Int), Int),
  )
}

/// Configuração do NEAT
pub type NeatConfig {
  NeatConfig(
    population_size: Int,
    num_inputs: Int,
    num_outputs: Int,
    // Probabilidades de mutação
    weight_mutation_rate: Float,
    weight_perturb_rate: Float,
    add_node_rate: Float,
    add_connection_rate: Float,
    disable_rate: Float,
    // Parâmetros de especiação
    compatibility_threshold: Float,
    excess_coefficient: Float,
    disjoint_coefficient: Float,
    weight_coefficient: Float,
    // Seleção
    survival_threshold: Float,
    elitism: Int,
    // Stagnation
    max_stagnation: Int,
  )
}

/// Config com threshold ajustado dinamicamente
pub fn adjust_threshold(config: NeatConfig, num_species: Int) -> NeatConfig {
  // Target: 5-10 espécies
  let target_min = 5
  let target_max = 10
  let adjustment = 0.1

  let new_threshold = case num_species < target_min {
    True -> config.compatibility_threshold -. adjustment  // Diminui = mais espécies
    False -> case num_species > target_max {
      True -> config.compatibility_threshold +. adjustment  // Aumenta = menos espécies
      False -> config.compatibility_threshold
    }
  }

  // Clamp entre 0.1 e 3.0
  let clamped = case new_threshold <. 0.1 {
    True -> 0.1
    False -> case new_threshold >. 3.0 {
      True -> 3.0
      False -> new_threshold
    }
  }

  NeatConfig(..config, compatibility_threshold: clamped)
}

/// Resultado de avaliação de fitness
pub type FitnessResult {
  FitnessResult(genome_id: Int, fitness: Float)
}

// =============================================================================
// CONFIGURATION - Configuração padrão
// =============================================================================

/// Configuração padrão do NEAT (otimizada conforme literatura)
pub fn default_config() -> NeatConfig {
  NeatConfig(
    population_size: 150,
    num_inputs: 2,
    num_outputs: 1,
    weight_mutation_rate: 0.9,
    // Qwen3: 0.9 é padrão NEAT
    weight_perturb_rate: 0.95,
    // Qwen3: mais perturbações vs resets
    add_node_rate: 0.03,
    add_connection_rate: 0.05,
    disable_rate: 0.01,
    compatibility_threshold: 1.7,
    // Qwen3: 1.7 para mais espécies
    excess_coefficient: 1.0,
    disjoint_coefficient: 1.0,
    weight_coefficient: 0.4,
    survival_threshold: 0.2,
    elitism: 2,
    max_stagnation: 15,
  )
}

/// Configuração para XOR (otimizada por Qwen3-235B)
pub fn xor_config() -> NeatConfig {
  NeatConfig(
    ..default_config(),
    num_inputs: 2,
    num_outputs: 1,
    population_size: 150,
    add_node_rate: 0.08,
    // Moderado para descobrir estrutura
    add_connection_rate: 0.12,
    // Adiciona conectividade gradualmente
    weight_perturb_rate: 0.95,
    // Alta taxa de perturbação (vs reset)
    compatibility_threshold: 1.7,
    // Qwen3: promove mais espécies
  )
}

/// Configuração para VIVA Soul (PAD → comportamento)
pub fn viva_soul_config() -> NeatConfig {
  NeatConfig(
    ..default_config(),
    num_inputs: 3,
    // PAD: Pleasure, Arousal, Dominance
    num_outputs: 4,
    // Ações: approach, avoid, express, suppress
    population_size: 100,
    add_node_rate: 0.05,
    add_connection_rate: 0.08,
  )
}

// =============================================================================
// INITIALIZATION - Criação de população inicial
// =============================================================================

/// Cria população inicial com genomas mínimos (fully connected, sem hidden)
pub fn create_population(config: NeatConfig, seed: Int) -> Population {
  let input_nodes =
    list.range(0, config.num_inputs - 1)
    |> list.map(fn(i) { NodeGene(id: i, node_type: Input, activation: Linear) })

  let bias_node =
    NodeGene(id: config.num_inputs, node_type: Bias, activation: Linear)

  let output_nodes =
    list.range(0, config.num_outputs - 1)
    |> list.map(fn(i) {
      NodeGene(
        id: config.num_inputs + 1 + i,
        node_type: Output,
        activation: Sigmoid,
      )
    })

  let base_nodes = list.flatten([input_nodes, [bias_node], output_nodes])

  // Cria conexões iniciais (all inputs + bias → all outputs)
  let input_ids = list.range(0, config.num_inputs)
  // inclui bias
  let output_ids =
    list.range(config.num_inputs + 1, config.num_inputs + config.num_outputs)

  let initial_connections =
    list.flatten(
      list.map(input_ids, fn(in_id) {
        list.index_map(output_ids, fn(out_id, idx) {
          let innovation = in_id * config.num_outputs + idx + 1
          ConnectionGene(
            in_node: in_id,
            out_node: out_id,
            weight: 0.0,
            // Será randomizado
            enabled: True,
            innovation: innovation,
          )
        })
      }),
    )

  let max_innovation = { config.num_inputs + 1 } * config.num_outputs

  // Create genomes with random weights and REAL structural diversity
  // (some connections removed entirely, not just disabled)
  let genomes =
    list.range(1, config.population_size)
    |> list.map(fn(i) {
      let connections =
        initial_connections
        |> list.index_map(fn(conn, idx) {
          let weight = random_weight(seed + i * 1000 + idx)
          ConnectionGene(..conn, weight: weight, enabled: True)
        })
        // Remove 10-30% of connections for structural diversity
        |> list.filter(fn(conn) {
          let r = pseudo_random(seed + i * 3000 + conn.innovation)
          let keep_rate = 0.7 +. pseudo_random(seed + i) *. 0.2
          r <. keep_rate
        })
      Genome(
        id: i,
        nodes: base_nodes,
        connections: connections,
        fitness: 0.0,
        adjusted_fitness: 0.0,
        species_id: 0,
      )
    })

  // Inicializa histórico de inovações
  let innovation_history =
    list.fold(initial_connections, dict.new(), fn(acc, conn) {
      dict.insert(acc, #(conn.in_node, conn.out_node), conn.innovation)
    })

  Population(
    genomes: genomes,
    species: [],
    generation: 0,
    innovation_counter: max_innovation,
    node_counter: config.num_inputs + config.num_outputs + 1,
    innovation_history: innovation_history,
  )
}

// =============================================================================
// ACTIVATION FUNCTIONS - Funções de ativação
// =============================================================================

/// Aplica função de ativação
pub fn activate(value: Float, activation: ActivationType) -> Float {
  case activation {
    Sigmoid -> sigmoid(value)
    Tanh -> tanh(value)
    ReLU -> relu(value)
    Linear -> value
  }
}

fn sigmoid(x: Float) -> Float {
  1.0 /. { 1.0 +. float_exp(0.0 -. x) }
}

fn tanh(x: Float) -> Float {
  let ex = float_exp(x)
  let emx = float_exp(0.0 -. x)
  { ex -. emx } /. { ex +. emx }
}

fn relu(x: Float) -> Float {
  float.max(0.0, x)
}

fn float_exp(x: Float) -> Float {
  // Aproximação de Taylor para e^x (clamp para evitar overflow)
  let clamped = float.clamp(x, -20.0, 20.0)
  exp_taylor(clamped, 20)
}

fn exp_taylor(x: Float, terms: Int) -> Float {
  list.range(0, terms - 1)
  |> list.fold(0.0, fn(acc, n) { acc +. power(x, n) /. factorial(n) })
}

fn power(base: Float, exp: Int) -> Float {
  case exp {
    0 -> 1.0
    1 -> base
    n if n > 0 -> base *. power(base, n - 1)
    _ -> 1.0
  }
}

fn factorial(n: Int) -> Float {
  case n {
    0 -> 1.0
    1 -> 1.0
    _ -> int.to_float(n) *. factorial(n - 1)
  }
}

// =============================================================================
// FORWARD PROPAGATION - Propagação para topologia variável
// =============================================================================

/// Executa forward pass em um genoma
pub fn forward(genome: Genome, inputs: List(Float)) -> List(Float) {
  // Mapeia nós por ID para acesso rápido
  let _node_map =
    list.fold(genome.nodes, dict.new(), fn(acc, node) {
      dict.insert(acc, node.id, node)
    })

  // Inicializa valores dos nós (apenas inputs e bias)
  let initial_values =
    list.fold(genome.nodes, dict.new(), fn(acc, node) {
      let value = case node.node_type {
        Input -> list_at(inputs, node.id) |> option.unwrap(0.0)
        Bias -> 1.0
        _ -> 0.0
      }
      dict.insert(acc, node.id, value)
    })

  // Ordena conexões topologicamente e propaga
  let sorted_connections = topological_sort_connections(genome)

  // Agrupa conexões por nó de destino
  let connections_by_target =
    list.fold(sorted_connections, dict.new(), fn(acc, conn) {
      case conn.enabled {
        False -> acc
        True -> {
          let existing = dict.get(acc, conn.out_node) |> result_unwrap([])
          dict.insert(acc, conn.out_node, list.append(existing, [conn]))
        }
      }
    })

  // Propaga em ordem topológica (por profundidade)
  let depths = calculate_node_depths(genome)
  let nodes_by_depth =
    genome.nodes
    |> list.filter(fn(n) { n.node_type == Hidden || n.node_type == Output })
    |> list.sort(fn(a, b) {
      let depth_a = dict.get(depths, a.id) |> result_unwrap(0)
      let depth_b = dict.get(depths, b.id) |> result_unwrap(0)
      int.compare(depth_a, depth_b)
    })

  let final_values =
    list.fold(nodes_by_depth, initial_values, fn(values, node) {
      let incoming =
        dict.get(connections_by_target, node.id) |> result_unwrap([])
      let weighted_sum =
        list.fold(incoming, 0.0, fn(acc, conn: ConnectionGene) {
          let in_value = dict.get(values, conn.in_node) |> result_unwrap(0.0)
          acc +. in_value *. conn.weight
        })
      // IMPORTANTE: Aplica ativação no nó (hidden e output)
      let activated = activate(weighted_sum, node.activation)
      dict.insert(values, node.id, activated)
    })

  // Coleta resultados dos outputs
  genome.nodes
  |> list.filter(fn(n) { n.node_type == Output })
  |> list.sort(fn(a, b) { int.compare(a.id, b.id) })
  |> list.map(fn(node) { dict.get(final_values, node.id) |> result_unwrap(0.0) })
}

/// Ordena conexões para propagação correta
fn topological_sort_connections(genome: Genome) -> List(ConnectionGene) {
  // Calcula profundidade de cada nó
  let depths = calculate_node_depths(genome)

  // Ordena conexões pela profundidade do nó de destino
  genome.connections
  |> list.filter(fn(c) { c.enabled })
  |> list.sort(fn(a, b) {
    let depth_a = dict.get(depths, a.out_node) |> result_unwrap(0)
    let depth_b = dict.get(depths, b.out_node) |> result_unwrap(0)
    int.compare(depth_a, depth_b)
  })
}

/// Calcula profundidade de cada nó (BFS)
fn calculate_node_depths(genome: Genome) -> Dict(Int, Int) {
  // Input e Bias têm depth 0
  let initial =
    genome.nodes
    |> list.filter(fn(n) { n.node_type == Input || n.node_type == Bias })
    |> list.fold(dict.new(), fn(acc, n) { dict.insert(acc, n.id, 0) })

  // Propaga depths
  propagate_depths(genome, initial, 100)
}

fn propagate_depths(
  genome: Genome,
  depths: Dict(Int, Int),
  max_iterations: Int,
) -> Dict(Int, Int) {
  case max_iterations {
    0 -> depths
    _ -> {
      let new_depths =
        list.fold(genome.connections, depths, fn(acc, conn) {
          case conn.enabled && dict.has_key(acc, conn.in_node) {
            False -> acc
            True -> {
              let in_depth = dict.get(acc, conn.in_node) |> result_unwrap(0)
              let current_depth =
                dict.get(acc, conn.out_node) |> result_unwrap(-1)
              let new_depth = in_depth + 1
              case new_depth > current_depth {
                True -> dict.insert(acc, conn.out_node, new_depth)
                False -> acc
              }
            }
          }
        })
      case dict.size(new_depths) == dict.size(depths) {
        True -> new_depths
        False -> propagate_depths(genome, new_depths, max_iterations - 1)
      }
    }
  }
}

// =============================================================================
// MUTATION - Operadores de mutação
// =============================================================================

/// Muta pesos de um genoma
pub fn mutate_weights(genome: Genome, config: NeatConfig, seed: Int) -> Genome {
  let new_connections =
    list.index_map(genome.connections, fn(conn, idx) {
      let rand = pseudo_random(seed + idx * 100)
      case rand <. config.weight_mutation_rate {
        False -> conn
        True -> {
          let perturb_rand = pseudo_random(seed + idx * 100 + 1)
          let new_weight = case perturb_rand <. config.weight_perturb_rate {
            True -> {
              // Perturbação pequena
              let delta = { pseudo_random(seed + idx * 100 + 2) -. 0.5 } *. 0.4
              conn.weight +. delta
            }
            False -> {
              // Novo peso aleatório
              random_weight(seed + idx * 100 + 3)
            }
          }
          ConnectionGene(..conn, weight: new_weight)
        }
      }
    })
  Genome(..genome, connections: new_connections)
}

/// Adiciona um novo nó dividindo uma conexão existente
pub fn mutate_add_node(
  genome: Genome,
  population: Population,
  seed: Int,
) -> #(Genome, Population) {
  // Seleciona conexão aleatória para dividir
  let enabled_connections = list.filter(genome.connections, fn(c) { c.enabled })
  case list.length(enabled_connections) {
    0 -> #(genome, population)
    len -> {
      let idx = float.round(pseudo_random(seed) *. int.to_float(len - 1))
      case list_at_int(enabled_connections, idx) {
        None -> #(genome, population)
        Some(conn) -> {
          // Cria novo nó hidden
          let new_node_id = population.node_counter
          let new_node =
            NodeGene(id: new_node_id, node_type: Hidden, activation: Sigmoid)

          // Desabilita conexão antiga
          let updated_connections =
            list.map(genome.connections, fn(c) {
              case c.innovation == conn.innovation {
                True -> ConnectionGene(..c, enabled: False)
                False -> c
              }
            })

          // Cria duas novas conexões
          let #(innov1, pop1) =
            get_innovation(population, conn.in_node, new_node_id)
          let #(innov2, pop2) = get_innovation(pop1, new_node_id, conn.out_node)

          let conn1 =
            ConnectionGene(
              in_node: conn.in_node,
              out_node: new_node_id,
              weight: 1.0,
              // Peso 1.0 para preservar comportamento
              enabled: True,
              innovation: innov1,
            )

          let conn2 =
            ConnectionGene(
              in_node: new_node_id,
              out_node: conn.out_node,
              weight: conn.weight,
              // Mantém peso original
              enabled: True,
              innovation: innov2,
            )

          let new_genome =
            Genome(
              ..genome,
              nodes: list.append(genome.nodes, [new_node]),
              connections: list.flatten([updated_connections, [conn1, conn2]]),
            )

          let new_population = Population(..pop2, node_counter: new_node_id + 1)

          #(new_genome, new_population)
        }
      }
    }
  }
}

/// Adiciona uma nova conexão entre dois nós
pub fn mutate_add_connection(
  genome: Genome,
  population: Population,
  seed: Int,
) -> #(Genome, Population) {
  // Encontra possíveis conexões (que não existem ainda)
  let existing_connections =
    list.fold(genome.connections, dict.new(), fn(acc, c) {
      dict.insert(acc, #(c.in_node, c.out_node), True)
    })

  let input_nodes =
    list.filter(genome.nodes, fn(n) {
      n.node_type == Input || n.node_type == Bias || n.node_type == Hidden
    })

  let output_nodes =
    list.filter(genome.nodes, fn(n) {
      n.node_type == Output || n.node_type == Hidden
    })

  // Gera todas possíveis novas conexões
  let possible =
    list.flatten(
      list.map(input_nodes, fn(in_node) {
        list.filter_map(output_nodes, fn(out_node) {
          case
            in_node.id != out_node.id
            && !dict.has_key(existing_connections, #(in_node.id, out_node.id))
          {
            True -> Ok(#(in_node.id, out_node.id))
            False -> Error(Nil)
          }
        })
      }),
    )

  case list.length(possible) {
    0 -> #(genome, population)
    len -> {
      let idx = float.round(pseudo_random(seed) *. int.to_float(len - 1))
      case list_at_int(possible, idx) {
        None -> #(genome, population)
        Some(#(in_id, out_id)) -> {
          let #(innovation, new_pop) = get_innovation(population, in_id, out_id)

          let new_conn =
            ConnectionGene(
              in_node: in_id,
              out_node: out_id,
              weight: random_weight(seed + 1),
              enabled: True,
              innovation: innovation,
            )

          let new_genome =
            Genome(
              ..genome,
              connections: list.append(genome.connections, [new_conn]),
            )

          #(new_genome, new_pop)
        }
      }
    }
  }
}

/// Obtém ou cria número de inovação para uma conexão
fn get_innovation(
  population: Population,
  in_node: Int,
  out_node: Int,
) -> #(Int, Population) {
  let key = #(in_node, out_node)
  case dict.get(population.innovation_history, key) {
    Ok(innovation) -> #(innovation, population)
    Error(_) -> {
      let new_innovation = population.innovation_counter + 1
      let new_history =
        dict.insert(population.innovation_history, key, new_innovation)
      let new_pop =
        Population(
          ..population,
          innovation_counter: new_innovation,
          innovation_history: new_history,
        )
      #(new_innovation, new_pop)
    }
  }
}

// =============================================================================
// CROSSOVER - Recombinação genética
// =============================================================================

/// Realiza crossover entre dois genomas (parent1 deve ter maior fitness)
pub fn crossover(parent1: Genome, parent2: Genome, seed: Int) -> Genome {
  // Agrupa genes por innovation number
  let genes1 =
    list.fold(parent1.connections, dict.new(), fn(acc, c) {
      dict.insert(acc, c.innovation, c)
    })

  let genes2 =
    list.fold(parent2.connections, dict.new(), fn(acc, c) {
      dict.insert(acc, c.innovation, c)
    })

  // Todos os innovation numbers
  let all_innovations =
    list.unique(list.append(
      list.map(parent1.connections, fn(c) { c.innovation }),
      list.map(parent2.connections, fn(c) { c.innovation }),
    ))
    |> list.sort(int.compare)

  // Combina genes
  let child_connections =
    list.index_map(all_innovations, fn(innov, idx) {
      let gene1 = dict.get(genes1, innov)
      let gene2 = dict.get(genes2, innov)
      case gene1, gene2 {
        // Matching gene - escolhe aleatoriamente
        Ok(g1), Ok(g2) -> {
          let rand = pseudo_random(seed + idx)
          let chosen = case rand <. 0.5 {
            True -> g1
            False -> g2
          }
          // Re-enable se um dos pais tinha enabled
          let enabled = g1.enabled || g2.enabled
          Some(ConnectionGene(..chosen, enabled: enabled))
        }
        // Disjoint/Excess - herda do parent mais fit (parent1)
        Ok(g1), Error(_) -> Some(g1)
        Error(_), Ok(_g2) -> None
        // Não deveria acontecer
        Error(_), Error(_) -> None
      }
    })
    |> list.filter_map(fn(x) { option.to_result(x, Nil) })

  // Coleta todos os nós necessários
  let required_node_ids =
    list.flatten([
      list.map(child_connections, fn(c: ConnectionGene) { c.in_node }),
      list.map(child_connections, fn(c: ConnectionGene) { c.out_node }),
    ])
    |> list.unique

  // Nós do parent1 (mais fit)
  let nodes1 =
    list.fold(parent1.nodes, dict.new(), fn(acc, n) {
      dict.insert(acc, n.id, n)
    })

  let nodes2 =
    list.fold(parent2.nodes, dict.new(), fn(acc, n) {
      dict.insert(acc, n.id, n)
    })

  let child_nodes =
    list.filter_map(required_node_ids, fn(id) {
      case dict.get(nodes1, id) {
        Ok(n) -> Ok(n)
        Error(_) ->
          case dict.get(nodes2, id) {
            Ok(n) -> Ok(n)
            Error(_) -> Error(Nil)
          }
      }
    })

  Genome(
    id: 0,
    // Será atribuído depois
    nodes: child_nodes,
    connections: child_connections,
    fitness: 0.0,
    adjusted_fitness: 0.0,
    species_id: 0,
  )
}

// =============================================================================
// SPECIATION - Agrupamento de genomas similares
// =============================================================================

/// Calcula distância de compatibilidade entre dois genomas
pub fn compatibility_distance(
  genome1: Genome,
  genome2: Genome,
  config: NeatConfig,
) -> Float {
  let genes1 =
    list.fold(genome1.connections, dict.new(), fn(acc, c) {
      dict.insert(acc, c.innovation, c)
    })

  let genes2 =
    list.fold(genome2.connections, dict.new(), fn(acc, c) {
      dict.insert(acc, c.innovation, c)
    })

  let max1 =
    list.fold(genome1.connections, 0, fn(acc, c) { int.max(acc, c.innovation) })
  let max2 =
    list.fold(genome2.connections, 0, fn(acc, c) { int.max(acc, c.innovation) })

  let #(excess, disjoint, matching, weight_diff) =
    count_gene_differences(genes1, genes2, max1, max2)

  let n =
    int.to_float(int.max(
      list.length(genome1.connections),
      list.length(genome2.connections),
    ))
  let n = float.max(n, 1.0)

  let avg_weight_diff = case matching {
    0 -> 0.0
    _ -> weight_diff /. int.to_float(matching)
  }

  { config.excess_coefficient *. int.to_float(excess) /. n }
  +. { config.disjoint_coefficient *. int.to_float(disjoint) /. n }
  +. { config.weight_coefficient *. avg_weight_diff }
}

fn count_gene_differences(
  genes1: Dict(Int, ConnectionGene),
  genes2: Dict(Int, ConnectionGene),
  max1: Int,
  max2: Int,
) -> #(Int, Int, Int, Float) {
  let all_innovations =
    list.unique(list.append(dict.keys(genes1), dict.keys(genes2)))

  list.fold(all_innovations, #(0, 0, 0, 0.0), fn(acc, innov) {
    let #(excess, disjoint, matching, weight_diff) = acc
    let g1 = dict.get(genes1, innov)
    let g2 = dict.get(genes2, innov)

    case g1, g2 {
      Ok(gene1), Ok(gene2) -> {
        // Matching
        let diff = float.absolute_value(gene1.weight -. gene2.weight)
        #(excess, disjoint, matching + 1, weight_diff +. diff)
      }
      Ok(_), Error(_) -> {
        // Gene only in genome1
        case innov > max2 {
          True -> #(excess + 1, disjoint, matching, weight_diff)
          // Excess
          False -> #(excess, disjoint + 1, matching, weight_diff)
          // Disjoint
        }
      }
      Error(_), Ok(_) -> {
        // Gene only in genome2
        case innov > max1 {
          True -> #(excess + 1, disjoint, matching, weight_diff)
          // Excess
          False -> #(excess, disjoint + 1, matching, weight_diff)
          // Disjoint
        }
      }
      Error(_), Error(_) -> acc
    }
  })
}

/// Agrupa genomas em espécies
pub fn speciate(population: Population, config: NeatConfig) -> Population {
  // Limpa membros das espécies existentes, mantém representantes
  let cleared_species =
    list.map(population.species, fn(s) { Species(..s, members: []) })

  // Atribui cada genoma a uma espécie
  let #(final_species, assigned_genomes) =
    list.fold(population.genomes, #(cleared_species, []), fn(acc, genome) {
      let #(species_list, genomes) = acc
      let #(new_species_list, species_id) =
        assign_to_species(genome, species_list, config)
      let updated_genome = Genome(..genome, species_id: species_id)
      #(new_species_list, list.append(genomes, [updated_genome]))
    })

  // Remove espécies vazias
  let non_empty_species =
    list.filter(final_species, fn(s) { !list.is_empty(s.members) })

  Population(
    ..population,
    genomes: assigned_genomes,
    species: non_empty_species,
  )
}

fn assign_to_species(
  genome: Genome,
  species_list: List(Species),
  config: NeatConfig,
) -> #(List(Species), Int) {
  // Tenta encontrar espécie compatível
  let compatible =
    list.find(species_list, fn(s) {
      compatibility_distance(genome, s.representative, config)
      <. config.compatibility_threshold
    })

  case compatible {
    Ok(species) -> {
      // Adiciona à espécie existente
      let updated_species =
        list.map(species_list, fn(s) {
          case s.id == species.id {
            True -> Species(..s, members: list.append(s.members, [genome]))
            False -> s
          }
        })
      #(updated_species, species.id)
    }
    Error(_) -> {
      // Cria nova espécie
      let new_id =
        list.fold(species_list, 0, fn(acc, s) { int.max(acc, s.id) }) + 1
      let new_species =
        Species(
          id: new_id,
          members: [genome],
          representative: genome,
          best_fitness: 0.0,
          stagnation: 0,
        )
      #(list.append(species_list, [new_species]), new_id)
    }
  }
}

// =============================================================================
// SELECTION & REPRODUCTION - Seleção e reprodução
// =============================================================================

/// Evolui população para próxima geração
pub fn evolve(
  population: Population,
  fitness_results: List(FitnessResult),
  config: NeatConfig,
  seed: Int,
) -> Population {
  // Ajusta threshold dinamicamente baseado no número de espécies
  let num_species = list.length(population.species)
  let adjusted_config = adjust_threshold(config, num_species)

  // Atualiza fitness dos genomas
  let pop_with_fitness = update_fitness(population, fitness_results)

  // Especiação com threshold ajustado
  let speciated = speciate(pop_with_fitness, adjusted_config)

  // Calcula adjusted fitness
  let with_adjusted = calculate_adjusted_fitness(speciated)

  // Seleciona e reproduz
  let next_gen = reproduce(with_adjusted, adjusted_config, seed)

  Population(..next_gen, generation: population.generation + 1)
}

fn update_fitness(
  population: Population,
  results: List(FitnessResult),
) -> Population {
  let fitness_map =
    list.fold(results, dict.new(), fn(acc, r) {
      dict.insert(acc, r.genome_id, r.fitness)
    })

  let updated_genomes =
    list.map(population.genomes, fn(g) {
      let fitness = dict.get(fitness_map, g.id) |> result_unwrap(0.0)
      Genome(..g, fitness: fitness)
    })

  Population(..population, genomes: updated_genomes)
}

fn calculate_adjusted_fitness(population: Population) -> Population {
  let updated_species =
    list.map(population.species, fn(species) {
      let species_size = int.to_float(list.length(species.members))
      let updated_members =
        list.map(species.members, fn(g) {
          Genome(..g, adjusted_fitness: g.fitness /. species_size)
        })
      let best =
        list.fold(updated_members, 0.0, fn(acc, g) { float.max(acc, g.fitness) })
      Species(..species, members: updated_members, best_fitness: best)
    })

  // Atualiza genomas na população
  let all_genomes = list.flatten(list.map(updated_species, fn(s) { s.members }))

  Population(..population, genomes: all_genomes, species: updated_species)
}

fn reproduce(
  population: Population,
  config: NeatConfig,
  seed: Int,
) -> Population {
  // Calcula offspring por espécie
  let total_adjusted =
    list.fold(population.genomes, 0.0, fn(acc, g) { acc +. g.adjusted_fitness })

  let offspring_counts =
    list.map(population.species, fn(species) {
      let species_adjusted =
        list.fold(species.members, 0.0, fn(acc, g) { acc +. g.adjusted_fitness })
      let proportion = case total_adjusted >. 0.0 {
        True -> species_adjusted /. total_adjusted
        False -> 1.0 /. int.to_float(list.length(population.species))
      }
      let count =
        float.round(proportion *. int.to_float(config.population_size))
      #(species.id, int.max(1, count))
    })

  // Reproduz cada espécie
  let #(new_genomes, updated_pop) =
    list.fold(offspring_counts, #([], population), fn(acc, species_count) {
      let #(genomes, pop) = acc
      let #(species_id, count) = species_count
      let species =
        list.find(pop.species, fn(s) { s.id == species_id })
        |> result_unwrap(Species(
          id: 0,
          members: [],
          representative: Genome(
            id: 0,
            nodes: [],
            connections: [],
            fitness: 0.0,
            adjusted_fitness: 0.0,
            species_id: 0,
          ),
          best_fitness: 0.0,
          stagnation: 0,
        ))

      let #(offspring, new_pop) =
        reproduce_species(species, count, pop, config, seed + species_id * 1000)

      #(list.append(genomes, offspring), new_pop)
    })

  // Atribui IDs aos novos genomas
  let numbered_genomes =
    list.index_map(new_genomes, fn(g, idx) { Genome(..g, id: idx + 1) })

  // Atualiza representantes das espécies
  let updated_species =
    list.map(updated_pop.species, fn(species) {
      let new_rep =
        list.find(numbered_genomes, fn(g) { g.species_id == species.id })
        |> result_unwrap(species.representative)
      Species(..species, representative: new_rep, members: [])
    })

  Population(..updated_pop, genomes: numbered_genomes, species: updated_species)
}

fn reproduce_species(
  species: Species,
  count: Int,
  population: Population,
  config: NeatConfig,
  seed: Int,
) -> #(List(Genome), Population) {
  // Ordena por fitness
  let sorted =
    species.members
    |> list.sort(fn(a, b) { float.compare(b.fitness, a.fitness) })

  // Elitismo - mantém os melhores sem mutação
  let elite = list.take(sorted, int.min(config.elitism, list.length(sorted)))

  // Seleciona pais (top survival_threshold %)
  let num_parents =
    float.round(int.to_float(list.length(sorted)) *. config.survival_threshold)
  let parents = list.take(sorted, int.max(2, num_parents))

  // Gera offspring
  let remaining = count - list.length(elite)
  let #(offspring, final_pop) =
    list.fold(
      list.range(1, int.max(0, remaining)),
      #([], population),
      fn(acc, i) {
        let #(children, pop) = acc
        let parent1 = select_parent(parents, seed + i * 10)
        let parent2 = select_parent(parents, seed + i * 10 + 1)

        // Crossover
        let child = case parent1.fitness >=. parent2.fitness {
          True -> crossover(parent1, parent2, seed + i * 100)
          False -> crossover(parent2, parent1, seed + i * 100)
        }

        // Mutações
        let child = mutate_weights(child, config, seed + i * 100 + 10)

        let #(child, pop) = case
          pseudo_random(seed + i * 100 + 20) <. config.add_node_rate
        {
          True -> mutate_add_node(child, pop, seed + i * 100 + 30)
          False -> #(child, pop)
        }

        let #(child, pop) = case
          pseudo_random(seed + i * 100 + 40) <. config.add_connection_rate
        {
          True -> mutate_add_connection(child, pop, seed + i * 100 + 50)
          False -> #(child, pop)
        }

        let child = Genome(..child, species_id: species.id)
        #(list.append(children, [child]), pop)
      },
    )

  let all_offspring =
    list.append(
      list.map(elite, fn(g) { Genome(..g, species_id: species.id) }),
      offspring,
    )

  #(all_offspring, final_pop)
}

fn select_parent(parents: List(Genome), seed: Int) -> Genome {
  let len = list.length(parents)
  let idx = float.round(pseudo_random(seed) *. int.to_float(len - 1))
  list_at_int(parents, idx)
  |> option.unwrap(Genome(
    id: 0,
    nodes: [],
    connections: [],
    fitness: 0.0,
    adjusted_fitness: 0.0,
    species_id: 0,
  ))
}

// =============================================================================
// UTILITIES - Funções auxiliares
// =============================================================================

fn random_weight(seed: Int) -> Float {
  { pseudo_random(seed) -. 0.5 } *. 4.0
}

fn pseudo_random(seed: Int) -> Float {
  // Xorshift32 - mais robusto que LCG para Gleam
  let s = int.absolute_value(seed) + 1
  let x = int.bitwise_exclusive_or(s, int.bitwise_shift_left(s, 13))
  let x = int.bitwise_exclusive_or(x, int.bitwise_shift_right(x, 17))
  let x = int.bitwise_exclusive_or(x, int.bitwise_shift_left(x, 5))
  let x = int.absolute_value(x) % 1_000_000
  int.to_float(x) /. 1_000_000.0
}

fn list_at(lst: List(a), index: Int) -> Option(a) {
  lst
  |> list.drop(index)
  |> list.first
  |> option.from_result
}

fn list_at_int(lst: List(a), index: Int) -> Option(a) {
  list_at(lst, index)
}

fn result_unwrap(result: Result(a, e), default: a) -> a {
  case result {
    Ok(value) -> value
    Error(_) -> default
  }
}

// =============================================================================
// PUBLIC API - Interface principal
// =============================================================================

/// Obtém o melhor genoma da população
pub fn get_best(population: Population) -> Option(Genome) {
  population.genomes
  |> list.sort(fn(a, b) { float.compare(b.fitness, a.fitness) })
  |> list.first
  |> option.from_result
}

/// Obtém estatísticas da população
pub type PopulationStats {
  PopulationStats(
    generation: Int,
    best_fitness: Float,
    avg_fitness: Float,
    num_species: Int,
    avg_nodes: Float,
    avg_connections: Float,
  )
}

pub fn get_stats(population: Population) -> PopulationStats {
  let genomes = population.genomes
  let len = int.to_float(list.length(genomes))

  let best_fitness =
    list.fold(genomes, 0.0, fn(acc, g) { float.max(acc, g.fitness) })

  let total_fitness = list.fold(genomes, 0.0, fn(acc, g) { acc +. g.fitness })

  let total_nodes =
    list.fold(genomes, 0, fn(acc, g) { acc + list.length(g.nodes) })

  let total_connections =
    list.fold(genomes, 0, fn(acc, g) { acc + list.length(g.connections) })

  PopulationStats(
    generation: population.generation,
    best_fitness: best_fitness,
    avg_fitness: total_fitness /. float.max(len, 1.0),
    num_species: list.length(population.species),
    avg_nodes: int.to_float(total_nodes) /. float.max(len, 1.0),
    avg_connections: int.to_float(total_connections) /. float.max(len, 1.0),
  )
}

/// Serializa genoma para string (debug)
pub fn genome_to_string(genome: Genome) -> String {
  let nodes_str =
    "Nodes: "
    <> int.to_string(list.length(genome.nodes))
    <> " ("
    <> int.to_string(
      list.length(list.filter(genome.nodes, fn(n) { n.node_type == Hidden })),
    )
    <> " hidden)"

  let connections_str =
    "Connections: "
    <> int.to_string(list.length(genome.connections))
    <> " ("
    <> int.to_string(
      list.length(list.filter(genome.connections, fn(c) { c.enabled })),
    )
    <> " enabled)"

  nodes_str <> " | " <> connections_str
}
