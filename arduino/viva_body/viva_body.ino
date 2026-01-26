/*
 * VIVA Body - Arduino firmware for physical embodiment
 *
 * This sketch implements the Body side of VIVA-Link protocol.
 * It sends sensor data to the Soul (Gleam) and receives commands.
 *
 * Hardware:
 * - 2x Speakers on pins 9, 10 (PWM)
 * - Temperature sensor on A0
 * - Light sensor on A1
 * - Touch sensor on pin 2
 * - LED on pin 13
 * - Optional: Servo on pin 6
 */

#include <PacketSerial.h>
#include "viva_packets.h"

// ============================================================================
// Configuration
// ============================================================================

#define BAUD_RATE 115200  // 500000 not reliable on Uno
#define TELEMETRY_INTERVAL_MS 10  // 100Hz
#define HEARTBEAT_TIMEOUT_MS 2000

// Pin assignments
#define PIN_SPEAKER_L 9
#define PIN_SPEAKER_R 10
#define PIN_TEMP A0
#define PIN_LIGHT A1
#define PIN_TOUCH 2
#define PIN_LED 13
#define PIN_SERVO 6

// ============================================================================
// Global State
// ============================================================================

PacketSerial_<COBS> vivaSerial;

uint8_t currentSeq = 0;
unsigned long lastTelemetry = 0;
unsigned long lastHeartbeat = 0;
bool soulConnected = false;

// Audio state
uint16_t audioFreqL = 0;
uint16_t audioFreqR = 0;
unsigned long audioEndTime = 0;
uint8_t audioWaveform = 0;

// PAD state (emotional coloring for responses)
float padPleasure = 0.0;
float padArousal = 0.0;
float padDominance = 0.0;

// ============================================================================
// Setup
// ============================================================================

void setup() {
    // Initialize serial with COBS framing
    vivaSerial.begin(BAUD_RATE);
    vivaSerial.setPacketHandler(&onPacketReceived);

    // Configure pins
    pinMode(PIN_SPEAKER_L, OUTPUT);
    pinMode(PIN_SPEAKER_R, OUTPUT);
    pinMode(PIN_LED, OUTPUT);
    pinMode(PIN_TOUCH, INPUT_PULLUP);

    // Startup indication
    digitalWrite(PIN_LED, HIGH);
    delay(100);
    digitalWrite(PIN_LED, LOW);
}

// ============================================================================
// Main Loop
// ============================================================================

void loop() {
    // Process incoming packets
    vivaSerial.update();

    unsigned long now = millis();

    // Send telemetry at fixed interval
    if (now - lastTelemetry >= TELEMETRY_INTERVAL_MS) {
        sendSensorData();
        lastTelemetry = now;
    }

    // Check soul connection (heartbeat timeout)
    if (soulConnected && (now - lastHeartbeat > HEARTBEAT_TIMEOUT_MS)) {
        soulConnected = false;
        onSoulDisconnected();
    }

    // Update audio output
    updateAudio(now);
}

// ============================================================================
// Sensor Reading & Telemetry
// ============================================================================

void sendSensorData() {
    VivaSensorData pkt;
    pkt.seq = nextSeq();

    // Read sensors
    pkt.temperature = readTemperature();
    pkt.light = analogRead(PIN_LIGHT);
    pkt.touch = !digitalRead(PIN_TOUCH);  // Active low
    pkt.audio_level = readAudioLevel();

    // Send packet
    vivaSerial.send((uint8_t*)&pkt, sizeof(pkt));
}

float readTemperature() {
    // Simple voltage to temperature conversion (adjust for your sensor)
    int raw = analogRead(PIN_TEMP);
    float voltage = raw * (5.0 / 1024.0);
    return (voltage - 0.5) * 100.0;  // LM35 formula
}

uint16_t readAudioLevel() {
    // Simple peak detection over a few samples
    uint16_t peak = 0;
    for (int i = 0; i < 10; i++) {
        uint16_t sample = abs(analogRead(A2) - 512);
        if (sample > peak) peak = sample;
    }
    return peak;
}

// ============================================================================
// Packet Handling
// ============================================================================

void onPacketReceived(const uint8_t* buffer, size_t size) {
    if (size < 2) return;  // Minimum: type + seq

    // TODO: Verify CRC here in production

    int8_t type = vivaParsePacket(buffer, size);

    switch (type) {
        case VIVA_TYPE_HEARTBEAT:
            handleHeartbeat();
            break;

        case VIVA_TYPE_COMMAND:
            handleCommand();
            break;

        case VIVA_TYPE_PADSTATE:
            handlePadState();
            break;

        case VIVA_TYPE_AUDIOCOMMAND:
            handleAudioCommand();
            break;

        default:
            // Unknown packet, send error
            sendError(0x01);  // Unknown type
            break;
    }
}

void handleHeartbeat() {
    lastHeartbeat = millis();

    if (!soulConnected) {
        soulConnected = true;
        onSoulConnected();
    }

    // Echo heartbeat back
    VivaHeartbeat ack;
    ack.seq = nextSeq();
    vivaSerial.send((uint8_t*)&ack, sizeof(ack));
}

void handleCommand() {
    // Apply physical commands
    // pkt_command was filled by vivaParsePacket

    // Servo control (if attached)
    // servo.write(pkt_command.servo_angle);

    // LED state
    digitalWrite(PIN_LED, pkt_command.led_state ? HIGH : LOW);

    // Vibration (PWM on separate motor pin if available)
    // analogWrite(PIN_VIBRATION, pkt_command.vibration);

    // Send acknowledgment
    sendAck(pkt_command.seq);
}

void handlePadState() {
    // Update emotional coloring
    padPleasure = pkt_padstate.pleasure;
    padArousal = pkt_padstate.arousal;
    padDominance = pkt_padstate.dominance;

    // Could modulate LED brightness based on arousal, etc.
    int brightness = map(padArousal * 100, -100, 100, 0, 255);
    analogWrite(PIN_LED, constrain(brightness, 0, 255));
}

void handleAudioCommand() {
    audioFreqL = pkt_audiocommand.freq_left;
    audioFreqR = pkt_audiocommand.freq_right;
    audioWaveform = pkt_audiocommand.waveform;
    audioEndTime = millis() + pkt_audiocommand.duration_ms;

    // Start audio immediately
    if (audioFreqL > 0) tone(PIN_SPEAKER_L, audioFreqL);
    if (audioFreqR > 0) tone(PIN_SPEAKER_R, audioFreqR);
}

// ============================================================================
// Audio Generation
// ============================================================================

void updateAudio(unsigned long now) {
    if (now >= audioEndTime) {
        // Stop audio
        noTone(PIN_SPEAKER_L);
        noTone(PIN_SPEAKER_R);
        audioFreqL = 0;
        audioFreqR = 0;
    }
    // For more complex waveforms, implement DDS here
}

// Binaural beat helper
void playBinauralBeat(uint16_t baseFreq, uint16_t beatFreq, uint16_t durationMs) {
    audioFreqL = baseFreq;
    audioFreqR = baseFreq + beatFreq;
    audioEndTime = millis() + durationMs;
    tone(PIN_SPEAKER_L, audioFreqL);
    tone(PIN_SPEAKER_R, audioFreqR);
}

// ============================================================================
// Response Packets
// ============================================================================

void sendAck(uint8_t ackedSeq) {
    VivaAck pkt;
    pkt.seq = nextSeq();
    pkt.acked_seq = ackedSeq;
    vivaSerial.send((uint8_t*)&pkt, sizeof(pkt));
}

void sendError(uint8_t errorCode) {
    VivaErrorPacket pkt;
    pkt.seq = nextSeq();
    pkt.error_code = errorCode;
    vivaSerial.send((uint8_t*)&pkt, sizeof(pkt));
}

// ============================================================================
// Connection Events
// ============================================================================

void onSoulConnected() {
    // Soul has connected! Celebrate with a brief tone
    tone(PIN_SPEAKER_L, 880, 100);  // A5
    delay(100);
    tone(PIN_SPEAKER_R, 1760, 100); // A6
}

void onSoulDisconnected() {
    // Soul disconnected, enter safe mode
    digitalWrite(PIN_LED, LOW);
    noTone(PIN_SPEAKER_L);
    noTone(PIN_SPEAKER_R);

    // Sad tone
    tone(PIN_SPEAKER_L, 220, 500);  // A3
}

// ============================================================================
// Utilities
// ============================================================================

uint8_t nextSeq() {
    return currentSeq++;
}
