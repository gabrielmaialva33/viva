#!/usr/bin/env python3
"""
VIVA-Link Serial Bridge
Bridges stdin/stdout to serial port for Erlang port communication.
"""

import sys
import serial
import threading
import select

def serial_to_stdout(ser):
    """Read from serial and write to stdout"""
    while True:
        try:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
        except Exception as e:
            sys.stderr.write(f"Serial read error: {e}\n")
            break

def stdin_to_serial(ser):
    """Read from stdin and write to serial"""
    while True:
        try:
            # Read from stdin
            data = sys.stdin.buffer.read(1)
            if not data:
                break
            ser.write(data)
        except Exception as e:
            sys.stderr.write(f"Serial write error: {e}\n")
            break

def main():
    if len(sys.argv) < 3:
        print("Usage: serial_bridge.py <device> <baud>", file=sys.stderr)
        sys.exit(1)

    device = sys.argv[1]
    baud = int(sys.argv[2])

    try:
        ser = serial.Serial(device, baud, timeout=0.1)
        sys.stderr.write(f"Connected to {device} at {baud} baud\n")

        # Start reader thread
        reader = threading.Thread(target=serial_to_stdout, args=(ser,), daemon=True)
        reader.start()

        # Main thread handles stdin
        stdin_to_serial(ser)

    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
