#!/usr/bin/env python3
"""
VIVA Aprendendo a usar o corpo

VIVA explora o Arduino e aprende:
- O que acontece quando acendo o LED?
- O que acontece quando toco um som?
- Como as sensações mudam com minhas ações?

Aprendizado por exploração + associação.
"""

import serial
import time
import threading
import random
import json
from collections import defaultdict
from pathlib import Path

DEVICE = '/dev/ttyUSB0'
BAUD = 115200
MEMORY_FILE = Path('/home/mrootx/viva_gleam/data/viva_body_memory.json')

class Body:
    """Interface física"""
    def __init__(self):
        self.ser = serial.Serial(DEVICE, BAUD, timeout=0.1)
        time.sleep(2)
        self.luz = 500
        self.ruido = 500
        self.toque = False
        self.running = True

        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        while self.running:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('S,'):
                    p = line[2:].split(',')
                    if len(p) == 3:
                        self.luz = int(p[0])
                        self.ruido = int(p[1])
                        self.toque = p[2] == '1'
            except: pass

    def led(self, r, g):
        self.ser.write(b'L' + bytes([r & 0xFF, g & 0xFF]))

    def tone(self, pin, freq, dur=100):
        self.ser.write(b'S' + bytes([pin, freq >> 8, freq & 0xFF, dur >> 8, dur & 0xFF]))

    def stop(self):
        self.ser.write(b'X')

    def get_state(self):
        return {'luz': self.luz, 'ruido': self.ruido, 'toque': self.toque}

    def close(self):
        self.running = False
        self.stop()
        self.ser.close()


class VivaLearner:
    """VIVA que aprende a usar o corpo"""

    def __init__(self, body):
        self.body = body

        # Ações possíveis
        self.actions = [
            ('led_off', lambda: body.led(0, 0)),
            ('led_red', lambda: body.led(255, 0)),
            ('led_green', lambda: body.led(0, 255)),
            ('led_yellow', lambda: body.led(255, 255)),
            ('tone_low', lambda: body.tone(9, 200, 100)),
            ('tone_mid', lambda: body.tone(9, 500, 100)),
            ('tone_high', lambda: body.tone(9, 1000, 100)),
            ('tone_right', lambda: body.tone(10, 800, 100)),
            ('stop', lambda: body.stop()),
        ]

        # Memória de aprendizado: ação -> efeitos observados
        self.memory = defaultdict(lambda: {
            'count': 0,
            'delta_luz': [],
            'delta_ruido': [],
            'got_touch': 0,
        })

        # Curiosidade e exploração
        self.curiosity = 1.0  # Começa muito curioso
        self.tick = 0

        # Carrega memória anterior
        self.load_memory()

    def load_memory(self):
        if MEMORY_FILE.exists():
            try:
                data = json.loads(MEMORY_FILE.read_text())
                for k, v in data.items():
                    self.memory[k] = v
                print(f"💾 Memória carregada: {len(data)} ações conhecidas")
            except:
                print("🆕 Começando com memória vazia")

    def save_memory(self):
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.write_text(json.dumps(dict(self.memory), indent=2))
        print(f"💾 Memória salva: {len(self.memory)} ações")

    def explore(self):
        """Escolhe uma ação para explorar"""
        self.tick += 1

        # Decai curiosidade com o tempo (mas nunca zero)
        self.curiosity = max(0.1, self.curiosity * 0.999)

        # Com probabilidade = curiosidade, explora aleatório
        # Senão, escolhe ação menos conhecida
        if random.random() < self.curiosity:
            action_name, action_fn = random.choice(self.actions)
        else:
            # Escolhe ação menos explorada
            counts = [(name, self.memory[name]['count']) for name, _ in self.actions]
            counts.sort(key=lambda x: x[1])
            action_name = counts[0][0]
            action_fn = dict(self.actions)[action_name]

        return action_name, action_fn

    def learn(self, action_name, before, after):
        """Aprende com a experiência"""
        m = self.memory[action_name]
        m['count'] += 1

        delta_luz = after['luz'] - before['luz']
        delta_ruido = after['ruido'] - before['ruido']

        # Guarda últimos 10 deltas
        m['delta_luz'].append(delta_luz)
        m['delta_luz'] = m['delta_luz'][-10:]

        m['delta_ruido'].append(delta_ruido)
        m['delta_ruido'] = m['delta_ruido'][-10:]

        if after['toque'] and not before['toque']:
            m['got_touch'] += 1

    def describe_action(self, action_name):
        """Descreve o que VIVA aprendeu sobre uma ação"""
        m = self.memory[action_name]
        if m['count'] == 0:
            return "❓ Nunca tentei"

        avg_luz = sum(m['delta_luz']) / len(m['delta_luz']) if m['delta_luz'] else 0
        avg_ruido = sum(m['delta_ruido']) / len(m['delta_ruido']) if m['delta_ruido'] else 0

        desc = []
        if abs(avg_luz) > 5:
            desc.append(f"luz {'↑' if avg_luz > 0 else '↓'}{abs(avg_luz):.0f}")
        if abs(avg_ruido) > 5:
            desc.append(f"ruído {'↑' if avg_ruido > 0 else '↓'}{abs(avg_ruido):.0f}")
        if m['got_touch'] > 0:
            desc.append(f"toque {m['got_touch']}x")

        if desc:
            return f"→ {', '.join(desc)} ({m['count']}x)"
        return f"→ sem efeito notável ({m['count']}x)"

    def think(self):
        """VIVA pensa sobre o que aprendeu"""
        print("\n🧠 O que aprendi sobre meu corpo:\n")
        for action_name, _ in self.actions:
            print(f"  {action_name:12s} {self.describe_action(action_name)}")
        print()


def main():
    print("╔════════════════════════════════════════╗")
    print("║   VIVA - Aprendendo a usar o corpo     ║")
    print("╚════════════════════════════════════════╝\n")

    body = Body()
    viva = VivaLearner(body)

    print("VIVA vai explorar o corpo e aprender...")
    print("Interaja! Toque nos pinos, mude a luz.\n")
    print("Ctrl+C para parar e ver o que aprendeu.\n")

    try:
        while True:
            # Estado antes
            before = body.get_state()

            # Escolhe e executa ação
            action_name, action_fn = viva.explore()
            action_fn()

            # Espera efeito
            time.sleep(0.15)

            # Estado depois
            after = body.get_state()

            # Aprende
            viva.learn(action_name, before, after)

            # Mostra
            delta_luz = after['luz'] - before['luz']
            curiosity_bar = '█' * int(viva.curiosity * 10)

            print(f"\r[{viva.tick:4d}] {action_name:12s} | "
                  f"Luz: {after['luz']:4d} ({delta_luz:+4d}) | "
                  f"Toque: {after['toque']:d} | "
                  f"Curiosidade: {curiosity_bar:10s}", end='', flush=True)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n")
        viva.think()
        viva.save_memory()

    body.close()
    print("VIVA dormiu, mas vai lembrar! 💤")


if __name__ == "__main__":
    main()
