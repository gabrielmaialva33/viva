//// VIVA-Link Demo - Test communication with Arduino body
////
//// Run with: gleam run -m viva/hardware/demo

import gleam/erlang/process
import gleam/io
import viva/hardware/packet
import viva/hardware/port_manager

pub fn main() {
  io.println("=== VIVA-Link Demo ===")
  io.println("Conectando ao Arduino em /dev/ttyUSB0...")

  // Start port manager
  case port_manager.start("/dev/ttyUSB0", 115_200) {
    Ok(manager) -> {
      io.println("✓ Port manager iniciado!")

      // Create a subject to receive packets
      let receiver = process.new_subject()

      // Subscribe to incoming packets
      port_manager.subscribe(manager, receiver)
      io.println("✓ Inscrito para receber packets")

      // Send heartbeat
      io.println("\nEnviando heartbeat...")
      let hb = packet.heartbeat(1)
      port_manager.send(manager, hb)

      // Send binaural beat command
      io.println("Enviando comando de áudio binaural (440Hz base, 10Hz beat)...")
      let audio = packet.binaural_beat(2, 440, 10, 2000)
      port_manager.send(manager, audio)

      // Send PAD state (happy/excited)
      io.println("Enviando estado PAD (feliz/animado)...")
      let pad = packet.pad_state(3, 0.8, 0.6, 0.5)
      port_manager.send(manager, pad)

      // Wait for responses
      io.println("\nAguardando respostas por 5 segundos...")
      receive_loop(receiver, 5000)

      // Get stats
      let stats = port_manager.get_stats(manager)
      io.println("\n=== Estatísticas ===")
      io.println(
        "Packets enviados: " <> int_to_string(stats.packets_sent),
      )
      io.println(
        "Packets recebidos: " <> int_to_string(stats.packets_received),
      )
      io.println("Erros CRC: " <> int_to_string(stats.crc_errors))

      // Shutdown
      io.println("\nEncerrando...")
      port_manager.shutdown(manager)
      io.println("✓ Demo finalizada!")
    }
    Error(_) -> {
      io.println("✗ Erro ao iniciar port manager")
      io.println("  Verifique se o Arduino está conectado em /dev/ttyUSB0")
    }
  }
}

fn receive_loop(receiver: process.Subject(packet.Packet), timeout_ms: Int) {
  case timeout_ms <= 0 {
    True -> Nil
    False -> {
      // Use select with timeout
      case process.receive(receiver, 100) {
        Ok(pkt) -> {
          io.println("← Recebido: " <> packet.describe(pkt))
          receive_loop(receiver, timeout_ms - 100)
        }
        Error(_) -> {
          // Timeout, continue waiting
          receive_loop(receiver, timeout_ms - 100)
        }
      }
    }
  }
}

@external(erlang, "erlang", "integer_to_list")
fn int_to_list(n: Int) -> List(Int)

fn int_to_string(n: Int) -> String {
  let chars = int_to_list(n)
  list_to_string(chars)
}

@external(erlang, "erlang", "list_to_binary")
fn list_to_string(chars: List(Int)) -> String
