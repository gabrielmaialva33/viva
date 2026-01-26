#!/usr/bin/env python3
"""
VIVA Soul <-> Body Demo

O Soul de VIVA sente através do Body e se expressa.
Estado emocional PAD (Pleasure-Arousal-Dominance) influencia outputs.
Sensações do Body influenciam o PAD.
"""

import serial
import time
import threading
import math
import random

# === CONFIGURAÇÃO ===
DEVICE = '/dev/ttyUSB0'
BAUD = 115200

# === ESTADO EMOCIONAL (PAD) ===
class SoulState:
    def __init__(self):
        # PAD: -1.0 a 1.0
        self.pleasure = 0.0   # Felicidade
        self.arousal = 0.0    # Excitação
        self.dominance = 0.5  # Controle

        # Memória de curto prazo
        self.last_luz = 500
        self.last_toque = False
        self.toque_count = 0

    def decay(self, dt=0.1):
        """Emoções decaem para neutro"""
        self.pleasure *= 0.98
        self.arousal *= 0.95
        self.dominance = 0.5 + (self.dominance - 0.5) * 0.99

    def clamp(self):
        self.pleasure = max(-1, min(1, self.pleasure))
        self.arousal = max(-1, min(1, self.arousal))
        self.dominance = max(0, min(1, self.dominance))

    def emotion_name(self):
        """Mapeia PAD para emoção discreta"""
        if self.pleasure > 0.3 and self.arousal > 0.3:
            return "FELIZ 😊"
        elif self.pleasure > 0.3 and self.arousal < -0.3:
            return "CALMO 😌"
        elif self.pleasure < -0.3 and self.arousal > 0.3:
            return "IRRITADO 😠"
        elif self.pleasure < -0.3 and self.arousal < -0.3:
            return "TRISTE 😢"
        elif self.arousal > 0.5:
            return "ALERTA ⚡"
        elif abs(self.pleasure) < 0.2 and abs(self.arousal) < 0.2:
            return "NEUTRO 😐"
        else:
            return "CURIOSO 🤔"

# === CORPO ===
class Body:
    def __init__(self, device, baud):
        self.ser = serial.Serial(device, baud, timeout=0.1)
        time.sleep(2)
        self.luz = 500
        self.ruido = 500
        self.toque = False
        self.running = True

        # Thread de leitura
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()

    def _read_loop(self):
        while self.running:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('S,'):
                    parts = line[2:].split(',')
                    if len(parts) == 3:
                        self.luz = int(parts[0])
                        self.ruido = int(parts[1])
                        self.toque = parts[2] == '1'
            except:
                pass

    def led(self, r, g):
        self.ser.write(b'L' + bytes([int(r) & 0xFF, int(g) & 0xFF]))

    def tone(self, pin, freq, dur=0):
        self.ser.write(b'S' + bytes([
            pin,
            (freq >> 8) & 0xFF, freq & 0xFF,
            (dur >> 8) & 0xFF, dur & 0xFF
        ]))

    def stop(self):
        self.ser.write(b'X')

    def close(self):
        self.running = False
        self.stop()
        self.ser.close()

# === LOOP PRINCIPAL ===
def main():
    print("╔════════════════════════════════════════╗")
    print("║     VIVA - Soul <-> Body Demo          ║")
    print("╚════════════════════════════════════════╝")
    print()

    body = Body(DEVICE, BAUD)
    soul = SoulState()

    print("Conectado! VIVA está sentindo...\n")
    print("- Toca no pino A0 = mais luz")
    print("- Conecta GND no pino 2 = toque")
    print("- Ctrl+C para sair\n")

    tick = 0
    try:
        while True:
            tick += 1

            # === PERCEPÇÃO ===
            delta_luz = body.luz - soul.last_luz

            # Mudança de luz afeta arousal
            if abs(delta_luz) > 30:
                soul.arousal += delta_luz / 500.0

            # Luz alta = mais pleasure
            soul.pleasure += (body.luz - 500) / 5000.0

            # Toque = forte estímulo positivo
            if body.toque and not soul.last_toque:
                soul.pleasure += 0.3
                soul.arousal += 0.4
                soul.toque_count += 1
            elif body.toque:
                soul.pleasure += 0.05

            # Memoriza
            soul.last_luz = body.luz
            soul.last_toque = body.toque

            # Decay e clamp
            soul.decay()
            soul.clamp()

            # === EXPRESSÃO ===
            # LED: vermelho = arousal negativo, verde = pleasure positivo
            r = int(max(0, -soul.pleasure) * 200 + max(0, soul.arousal) * 55)
            g = int(max(0, soul.pleasure) * 200 + max(0, -soul.arousal) * 55)
            body.led(r, g)

            # Som: arousal alto = frequência alta
            if soul.arousal > 0.4:
                freq = int(300 + soul.arousal * 500 + soul.pleasure * 200)
                body.tone(9, freq, 50)
            elif soul.pleasure > 0.5:
                # Melodia feliz
                if tick % 5 == 0:
                    notes = [523, 659, 784]
                    body.tone(9, notes[tick % 3], 100)
            elif soul.pleasure < -0.4:
                # Tom triste
                if tick % 10 == 0:
                    body.tone(9, 150 + int(soul.pleasure * 50), 200)

            # === DISPLAY ===
            p_bar = '█' * int((soul.pleasure + 1) * 5)
            a_bar = '█' * int((soul.arousal + 1) * 5)
            emotion = soul.emotion_name()

            print(f"\r[{tick:4d}] P:{soul.pleasure:+.2f} {p_bar:10s} | "
                  f"A:{soul.arousal:+.2f} {a_bar:10s} | "
                  f"Luz:{body.luz:4d} Toque:{body.toque:d} | "
                  f"{emotion:12s}", end='', flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nEncerrando...")

    body.close()
    print(f"Total de toques sentidos: {soul.toque_count}")
    print("VIVA dormiu.")

if __name__ == "__main__":
    main()
