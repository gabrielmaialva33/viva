#!/usr/bin/env python3
"""
VIVA-Link Monitor - Visualiza dados do Arduino em tempo real
"""

import serial
import struct
import time
from collections import deque

# Packet types
TYPE_HEARTBEAT = 0x00
TYPE_SENSORDATA = 0x01
TYPE_COMMAND = 0x02
TYPE_PADSTATE = 0x10
TYPE_AUDIOCOMMAND = 0x11
TYPE_ACK = 0xFE
TYPE_ERROR = 0xFF

def decode_cobs(data: bytes) -> bytes:
    """Decode COBS-encoded data (without delimiter)"""
    if not data:
        return b''

    output = []
    idx = 0

    while idx < len(data):
        code = data[idx]
        if code == 0:
            break  # Invalid

        idx += 1
        for _ in range(code - 1):
            if idx >= len(data):
                break
            output.append(data[idx])
            idx += 1

        if code < 0xFF and idx < len(data):
            output.append(0)

    # Remove trailing zero if present
    if output and output[-1] == 0:
        output.pop()

    return bytes(output)

def parse_sensor_data(data: bytes):
    """Parse SensorData packet"""
    if len(data) < 11:
        return None

    # Skip type_id (already known)
    seq = data[1]
    temp = struct.unpack('<f', data[2:6])[0]
    light = struct.unpack('<H', data[6:8])[0]
    touch = data[8] == 1
    audio = struct.unpack('<H', data[9:11])[0]

    return {
        'seq': seq,
        'temperature': temp,
        'light': light,
        'touch': touch,
        'audio_level': audio
    }

def send_audio_command(ser, freq_left, freq_right, duration_ms, waveform=0):
    """Send an AudioCommand packet"""
    # Build packet: type, seq, freq_l, freq_r, duration, waveform
    seq = 0
    packet = struct.pack('<BBHHHB',
        TYPE_AUDIOCOMMAND,
        seq,
        freq_left,
        freq_right,
        duration_ms,
        waveform
    )

    # Simple CRC-8 (XOR all bytes)
    crc = 0
    for b in packet:
        crc ^= b
    packet += bytes([crc])

    # COBS encode
    encoded = cobs_encode(packet)

    # Send with delimiter
    ser.write(encoded + b'\x00')
    print(f"Sent: AudioCommand L={freq_left}Hz R={freq_right}Hz dur={duration_ms}ms")

def cobs_encode(data: bytes) -> bytes:
    """COBS encode data"""
    output = []
    idx = 0

    while idx < len(data):
        # Find next zero or end
        block_start = idx
        while idx < len(data) and data[idx] != 0 and (idx - block_start) < 254:
            idx += 1

        block_len = idx - block_start

        if idx < len(data) and data[idx] == 0:
            # Found zero
            output.append(block_len + 1)
            output.extend(data[block_start:idx])
            idx += 1  # Skip zero
        else:
            # End of data or max block
            if block_len == 254:
                output.append(0xFF)
                output.extend(data[block_start:idx])
            else:
                output.append(block_len + 1)
                output.extend(data[block_start:idx])

    return bytes(output)

def main():
    print("=== VIVA-Link Monitor ===")
    print("Conectando ao Arduino...")

    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
    time.sleep(2)  # Wait for Arduino reset

    print("Conectado! Pressione Ctrl+C para sair.\n")

    buffer = bytearray()
    packet_count = 0
    last_display = time.time()
    temps = deque(maxlen=10)
    lights = deque(maxlen=10)

    # Send initial audio command (binaural beat: 440Hz + 450Hz = 10Hz beat)
    print("Enviando comando de áudio binaural (440Hz base, 10Hz beat)...")
    send_audio_command(ser, 440, 450, 3000)  # 3 seconds

    try:
        while True:
            # Read available data
            if ser.in_waiting > 0:
                buffer.extend(ser.read(ser.in_waiting))

            # Look for COBS delimiter (0x00)
            while 0x00 in buffer:
                idx = buffer.index(0x00)
                frame = bytes(buffer[:idx])
                buffer = buffer[idx+1:]

                if len(frame) < 2:
                    continue

                # Decode COBS
                try:
                    decoded = decode_cobs(frame)
                    if len(decoded) < 2:
                        continue

                    type_id = decoded[0]

                    if type_id == TYPE_SENSORDATA:
                        data = parse_sensor_data(decoded)
                        if data:
                            packet_count += 1
                            temps.append(data['temperature'])
                            lights.append(data['light'])

                    elif type_id == TYPE_HEARTBEAT:
                        print(f"♥ Heartbeat seq={decoded[1]}")

                    elif type_id == TYPE_ACK:
                        print(f"✓ Ack seq={decoded[1]} acked={decoded[2]}")

                except Exception as e:
                    pass  # Skip invalid packets

            # Display stats every second
            now = time.time()
            if now - last_display >= 1.0:
                if temps:
                    avg_temp = sum(temps) / len(temps)
                    avg_light = sum(lights) / len(lights) if lights else 0
                    print(f"📊 Packets/s: {packet_count:3d} | Temp: {avg_temp:5.1f}°C | Light: {avg_light:4.0f}")
                packet_count = 0
                last_display = now

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nEncerrando...")

    ser.close()
    print("Desconectado.")

if __name__ == "__main__":
    main()
