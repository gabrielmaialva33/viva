#!/usr/bin/env python3
"""
VIVA Audio - Controle simples dos speakers via VIVA-Link

Uso:
  ./viva_audio.py tone 440 1000      # Tom 440Hz por 1s
  ./viva_audio.py binaural 440 10 3000  # Binaural 440Hz base, 10Hz beat, 3s
  ./viva_audio.py alpha              # Onda alpha (10Hz)
  ./viva_audio.py theta              # Onda theta (6Hz)
  ./viva_audio.py sweep              # Sweep stereo
"""

import serial
import struct
import time
import sys

DEVICE = '/dev/ttyUSB0'
BAUD = 115200

def crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) if crc & 0x80 else (crc << 1)
            crc &= 0xFF
    return crc

def cobs_encode(data):
    output = []
    idx = 0
    while idx < len(data):
        block_start = idx
        while idx < len(data) and data[idx] != 0 and (idx - block_start) < 254:
            idx += 1
        block_len = idx - block_start
        if idx < len(data) and data[idx] == 0:
            output.append(block_len + 1)
            output.extend(data[block_start:idx])
            idx += 1
        else:
            output.append(block_len + 1 if block_len < 254 else 0xFF)
            output.extend(data[block_start:idx])
    return bytes(output)

def send_audio(ser, freq_l, freq_r, duration_ms):
    packet = struct.pack('<BBHHHB', 0x11, 0, freq_l, freq_r, duration_ms, 0)
    encoded = cobs_encode(packet + bytes([crc8(packet)])) + b'\x00'
    ser.write(encoded)

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    ser = serial.Serial(DEVICE, BAUD)
    time.sleep(0.5)

    if cmd == 'tone':
        freq = int(sys.argv[2]) if len(sys.argv) > 2 else 440
        dur = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        print(f"Tom: {freq}Hz ({dur}ms)")
        send_audio(ser, freq, freq, dur)

    elif cmd == 'binaural':
        base = int(sys.argv[2]) if len(sys.argv) > 2 else 440
        beat = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        dur = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
        print(f"Binaural: {base}Hz + {base+beat}Hz = {beat}Hz beat ({dur}ms)")
        send_audio(ser, base, base + beat, dur)

    elif cmd == 'alpha':
        print("Alpha wave: 440Hz + 450Hz = 10Hz beat (relaxamento)")
        send_audio(ser, 440, 450, 5000)

    elif cmd == 'theta':
        print("Theta wave: 300Hz + 306Hz = 6Hz beat (meditacao)")
        send_audio(ser, 300, 306, 5000)

    elif cmd == 'sweep':
        print("Sweep stereo...")
        for freq in range(200, 1000, 50):
            send_audio(ser, freq, 1200 - freq, 100)
            time.sleep(0.12)

    elif cmd == 'stop':
        send_audio(ser, 0, 0, 0)
        print("Audio parado")

    else:
        print(f"Comando desconhecido: {cmd}")
        print(__doc__)

    time.sleep(0.1)
    ser.close()

if __name__ == "__main__":
    main()
