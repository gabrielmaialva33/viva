/*
 * VIVA Body - Bare Metal AVR
 *
 * No Arduino libraries - direct register access
 * ATmega328P @ 16MHz
 *
 * Pins:
 *   PD5 (OC0B) -> Red LED (PWM Timer0)
 *   PD6 (OC0A) -> Green LED (PWM Timer0)
 *   PB1 (OC1A) -> Speaker L (PWM Timer1)
 *   PB2 (OC1B) -> Speaker R (PWM Timer1)
 *   PC0 (ADC0) -> Light sensor
 *   PC1 (ADC1) -> Noise sensor
 *   PD2        -> Touch (INPUT_PULLUP)
 *
 * Protocol:
 *   IN:  'L' r g | 'S' pin fH fL dH dL | 'X'
 *   OUT: "S,light,noise,touch\n" @ 20Hz
 */

#define F_CPU 16000000UL

#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#define BAUD 115200
#define UBRR_VAL ((F_CPU / 16 / BAUD) - 1)

// === USART ===

void usart_init(void) {
    // Baud rate
    UBRR0H = (uint8_t)(UBRR_VAL >> 8);
    UBRR0L = (uint8_t)UBRR_VAL;

    // Enable TX and RX
    UCSR0B = (1 << RXEN0) | (1 << TXEN0);

    // 8 bits, 1 stop, no parity
    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);
}

void usart_tx(uint8_t data) {
    while (!(UCSR0A & (1 << UDRE0)));
    UDR0 = data;
}

uint8_t usart_rx_ready(void) {
    return UCSR0A & (1 << RXC0);
}

uint8_t usart_rx(void) {
    while (!(UCSR0A & (1 << RXC0)));
    return UDR0;
}

void usart_print(const char* s) {
    while (*s) usart_tx(*s++);
}

void usart_print_int(uint16_t n) {
    char buf[6];
    int8_t i = 0;

    if (n == 0) {
        usart_tx('0');
        return;
    }

    while (n > 0) {
        buf[i++] = '0' + (n % 10);
        n /= 10;
    }

    while (--i >= 0) usart_tx(buf[i]);
}

// === ADC ===

void adc_init(void) {
    // AVCC as reference, ADC0 initial
    ADMUX = (1 << REFS0);

    // Enable ADC, prescaler 128 (125kHz @ 16MHz)
    ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);
}

uint16_t adc_read(uint8_t channel) {
    // Select channel (0-7)
    ADMUX = (ADMUX & 0xF0) | (channel & 0x0F);

    // Start conversion
    ADCSRA |= (1 << ADSC);

    // Wait for completion
    while (ADCSRA & (1 << ADSC));

    return ADC;
}

// === PWM (Timer0 for LEDs) ===

void pwm_led_init(void) {
    // PD5 and PD6 as output
    DDRD |= (1 << PD5) | (1 << PD6);

    // Fast PWM, TOP=0xFF
    TCCR0A = (1 << WGM01) | (1 << WGM00);

    // Prescaler 64 (~1kHz PWM)
    TCCR0B = (1 << CS01) | (1 << CS00);

    // Start off
    OCR0A = 0;  // Green (PD6)
    OCR0B = 0;  // Red (PD5)
}

void led_set(uint8_t red, uint8_t green) {
    // Enable/disable output compare
    if (red > 0) {
        TCCR0A |= (1 << COM0B1);  // Non-inverting PD5
        OCR0B = red;
    } else {
        TCCR0A &= ~(1 << COM0B1);
        PORTD &= ~(1 << PD5);
    }

    if (green > 0) {
        TCCR0A |= (1 << COM0A1);  // Non-inverting PD6
        OCR0A = green;
    } else {
        TCCR0A &= ~(1 << COM0A1);
        PORTD &= ~(1 << PD6);
    }
}

// === PWM (Timer1 for Speakers) ===

volatile uint16_t tone_duration_l = 0;
volatile uint16_t tone_duration_r = 0;

void pwm_speaker_init(void) {
    // PB1 and PB2 as output
    DDRB |= (1 << PB1) | (1 << PB2);

    // CTC mode, TOP=ICR1
    TCCR1A = 0;
    TCCR1B = (1 << WGM13) | (1 << WGM12);

    // Start stopped
    ICR1 = 0xFFFF;
    OCR1A = 0;
    OCR1B = 0;
}

void tone_set(uint8_t pin, uint16_t freq, uint16_t dur_ms) {
    if (freq == 0) {
        // Stop
        if (pin == 9) {
            TCCR1A &= ~(1 << COM1A0);
            PORTB &= ~(1 << PB1);
        } else if (pin == 10) {
            TCCR1A &= ~(1 << COM1B0);
            PORTB &= ~(1 << PB2);
        }
        return;
    }

    // Calculate TOP for desired frequency
    // freq = F_CPU / (2 * prescaler * (1 + TOP))
    // Prescaler 8: TOP = F_CPU / (16 * freq) - 1
    uint16_t top = (F_CPU / (16UL * freq)) - 1;
    if (top < 1) top = 1;

    // Prescaler 8
    TCCR1B = (TCCR1B & 0xF8) | (1 << CS11);
    ICR1 = top;

    if (pin == 9) {
        OCR1A = top / 2;  // 50% duty
        TCCR1A |= (1 << COM1A0);  // Toggle on match
        tone_duration_l = dur_ms;
    } else if (pin == 10) {
        OCR1B = top / 2;
        TCCR1A |= (1 << COM1B0);
        tone_duration_r = dur_ms;
    }
}

void tone_stop_all(void) {
    TCCR1A &= ~((1 << COM1A0) | (1 << COM1B0));
    PORTB &= ~((1 << PB1) | (1 << PB2));
    tone_duration_l = 0;
    tone_duration_r = 0;
}

// === GPIO ===

void gpio_init(void) {
    // PD2 as input with pullup (touch)
    DDRD &= ~(1 << PD2);
    PORTD |= (1 << PD2);
}

uint8_t touch_read(void) {
    // Pullup: LOW = touched
    return !(PIND & (1 << PD2));
}

// === Timer2 for timing (50ms interval) ===

volatile uint16_t tick_ms = 0;

void timer2_init(void) {
    // CTC mode, TOP=OCR2A
    TCCR2A = (1 << WGM21);

    // Prescaler 1024: 16MHz/1024 = 15625Hz
    // OCR2A = 156 -> ~100Hz (10ms)
    TCCR2B = (1 << CS22) | (1 << CS21) | (1 << CS20);
    OCR2A = 156;

    // Enable interrupt
    TIMSK2 = (1 << OCIE2A);
}

ISR(TIMER2_COMPA_vect) {
    tick_ms += 10;

    // Decrement tone durations
    if (tone_duration_l > 0) {
        if (tone_duration_l <= 10) {
            tone_set(9, 0, 0);
            tone_duration_l = 0;
        } else {
            tone_duration_l -= 10;
        }
    }
    if (tone_duration_r > 0) {
        if (tone_duration_r <= 10) {
            tone_set(10, 0, 0);
            tone_duration_r = 0;
        } else {
            tone_duration_r -= 10;
        }
    }
}

// === Protocol ===

void handle_command(uint8_t cmd) {
    switch (cmd) {
        case 'L': {
            // Wait for 2 bytes
            uint8_t r = usart_rx();
            uint8_t g = usart_rx();
            led_set(r, g);
            break;
        }

        case 'S': {
            // Wait for 5 bytes
            uint8_t pin = usart_rx();
            uint16_t freq = ((uint16_t)usart_rx() << 8) | usart_rx();
            uint16_t dur = ((uint16_t)usart_rx() << 8) | usart_rx();
            tone_set(pin, freq, dur);
            break;
        }

        case 'X':
            led_set(0, 0);
            tone_stop_all();
            break;

        case '\n':
        case '\r':
            break;
    }
}

void send_sensors(void) {
    uint16_t light = adc_read(0);
    uint16_t noise = adc_read(1);
    uint8_t touch = touch_read();

    usart_print("S,");
    usart_print_int(light);
    usart_tx(',');
    usart_print_int(noise);
    usart_tx(',');
    usart_tx('0' + touch);
    usart_tx('\n');
}

// === Main ===

int main(void) {
    // Init
    usart_init();
    adc_init();
    pwm_led_init();
    pwm_speaker_init();
    gpio_init();
    timer2_init();

    // Enable interrupts
    sei();

    // Boot signal
    led_set(0, 100);
    tone_set(9, 440, 100);
    _delay_ms(150);
    led_set(0, 0);

    usart_print("VIVA_READY\n");

    uint16_t last_send = 0;

    while (1) {
        // Receive commands
        while (usart_rx_ready()) {
            handle_command(usart_rx());
        }

        // Send sensors every 50ms
        if (tick_ms - last_send >= 50) {
            send_sensors();
            last_send = tick_ms;
        }
    }

    return 0;
}
