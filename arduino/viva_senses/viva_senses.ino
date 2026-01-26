/*
 * VIVA Body - Corpo com SENSAÇÕES em tempo real
 *
 * ENVIA sensores -> Soul (Gleam)
 * RECEBE comandos <- Soul (Gleam)
 *
 * Conexoes:
 *   Pin 9  -> Speaker L
 *   Pin 10 -> Speaker R
 *   Pin 5  -> LED Vermelho
 *   Pin 6  -> LED Verde
 *   A0     -> Sensor (LDR, pot, ou deixa solto = ruido)
 *   A1     -> Sensor 2 (opcional)
 *   Pin 2  -> Botao/Touch (opcional, INPUT_PULLUP)
 *
 * Protocolo ENTRADA (Gleam -> Arduino):
 *   'L' r g           -> LED
 *   'S' pin fH fL dH dL -> Som
 *   'X'               -> Stop all
 *
 * Protocolo SAIDA (Arduino -> Gleam):
 *   Envia a cada 50ms: "S,luz,ruido,toque\n"
 */

#define SPEAKER_L 9
#define SPEAKER_R 10
#define LED_RED 5
#define LED_GREEN 6
#define SENSOR_LIGHT A0
#define SENSOR_NOISE A1
#define TOUCH_PIN 2

// Timing
unsigned long lastSensorSend = 0;
const int SENSOR_INTERVAL = 50;  // 20Hz

void setup() {
    Serial.begin(115200);

    pinMode(SPEAKER_L, OUTPUT);
    pinMode(SPEAKER_R, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    pinMode(LED_GREEN, OUTPUT);
    pinMode(TOUCH_PIN, INPUT_PULLUP);

    // Boot signal
    analogWrite(LED_GREEN, 100);
    tone(SPEAKER_L, 440, 100);
    delay(150);
    analogWrite(LED_GREEN, 0);

    Serial.println("VIVA_READY");
}

void loop() {
    // === RECEBER COMANDOS ===
    while (Serial.available() >= 1) {
        char cmd = Serial.read();
        handleCommand(cmd);
    }

    // === ENVIAR SENSACOES ===
    unsigned long now = millis();
    if (now - lastSensorSend >= SENSOR_INTERVAL) {
        sendSensors();
        lastSensorSend = now;
    }
}

void handleCommand(char cmd) {
    switch (cmd) {
        case 'L':  // LED
            waitFor(2);
            analogWrite(LED_RED, Serial.read());
            analogWrite(LED_GREEN, Serial.read());
            break;

        case 'S':  // Som
            waitFor(5);
            {
                uint8_t pin = Serial.read();
                uint16_t freq = (Serial.read() << 8) | Serial.read();
                uint16_t dur = (Serial.read() << 8) | Serial.read();

                if (pin == 9 || pin == 10) {
                    if (freq == 0) noTone(pin);
                    else if (dur == 0) tone(pin, freq);
                    else tone(pin, freq, dur);
                }
            }
            break;

        case 'X':  // Stop
            noTone(SPEAKER_L);
            noTone(SPEAKER_R);
            analogWrite(LED_RED, 0);
            analogWrite(LED_GREEN, 0);
            break;

        case '\n':
        case '\r':
            break;
    }
}

void sendSensors() {
    // Lê sensores
    int light = analogRead(SENSOR_LIGHT);    // 0-1023: luz ou variação
    int noise = analogRead(SENSOR_NOISE);    // 0-1023: ruído elétrico
    int touch = !digitalRead(TOUCH_PIN);     // 1 se tocado (pullup invertido)

    // Formato: S,luz,ruido,toque
    Serial.print("S,");
    Serial.print(light);
    Serial.print(",");
    Serial.print(noise);
    Serial.print(",");
    Serial.println(touch);
}

void waitFor(int n) {
    unsigned long t = millis();
    while (Serial.available() < n && millis() - t < 100);
}
