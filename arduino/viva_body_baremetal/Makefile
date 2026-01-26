# VIVA Body - Bare Metal AVR Makefile
# ATmega328P @ 16MHz (Arduino Uno)
# Direct register access, no Arduino libraries

MCU = atmega328p
F_CPU = 16000000UL
BAUD = 115200

# Tools
CC = avr-gcc
OBJCOPY = avr-objcopy
AVRDUDE = avrdude
SIZE = avr-size

# Flags
CFLAGS = -mmcu=$(MCU) -DF_CPU=$(F_CPU) -Os -Wall -std=c99
LDFLAGS = -mmcu=$(MCU)

# Files
TARGET = viva_body
SRC = main.c

# Serial port (adjust as needed)
PORT ?= /dev/ttyUSB0
PROGRAMMER = arduino

.PHONY: all clean flash monitor size help

all: $(TARGET).hex size

$(TARGET).elf: $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(TARGET).hex: $(TARGET).elf
	$(OBJCOPY) -O ihex -R .eeprom $< $@

size: $(TARGET).elf
	$(SIZE) --format=avr --mcu=$(MCU) $<

flash: $(TARGET).hex
	$(AVRDUDE) -c $(PROGRAMMER) -p $(MCU) -P $(PORT) -b $(BAUD) -U flash:w:$<:i

monitor:
	screen $(PORT) 115200

clean:
	rm -f $(TARGET).elf $(TARGET).hex

# If avr-gcc not installed:
install-avr:
	sudo apt-get update
	sudo apt-get install -y gcc-avr avr-libc avrdude

# Help
help:
	@echo "VIVA Body - Bare Metal AVR"
	@echo ""
	@echo "Targets:"
	@echo "  make all       - Compile with avr-gcc (needs: make install-avr)"
	@echo "  make flash     - Flash via avrdude"
	@echo "  make monitor   - Serial monitor"
	@echo "  make clean     - Clean files"
	@echo ""
	@echo "For Arduino IDE/CLI version, use ../viva_senses/"
