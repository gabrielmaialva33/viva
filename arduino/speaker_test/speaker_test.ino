/*
 * VIVA Speaker Test - Teste basico dos 2 speakers
 *
 * Conexoes:
 *   Pin 9  -> Speaker Esquerdo (+)
 *   Pin 10 -> Speaker Direito (+)
 *   GND    -> Ambos speakers (-)
 *
 * O teste toca:
 *   1. Tom no speaker esquerdo
 *   2. Tom no speaker direito
 *   3. Binaural beat (frequencias diferentes)
 *   4. Sweep de frequencia
 */

#define SPEAKER_L 9
#define SPEAKER_R 10

void setup() {
    Serial.begin(115200);
    pinMode(SPEAKER_L, OUTPUT);
    pinMode(SPEAKER_R, OUTPUT);

    Serial.println("=== VIVA Speaker Test ===");
    Serial.println("Iniciando em 2 segundos...");
    delay(2000);
}

void loop() {
    // Teste 1: Speaker Esquerdo
    Serial.println("1. Speaker ESQUERDO (440Hz - La)");
    tone(SPEAKER_L, 440, 500);
    delay(700);

    // Teste 2: Speaker Direito
    Serial.println("2. Speaker DIREITO (880Hz - La agudo)");
    tone(SPEAKER_R, 880, 500);
    delay(700);

    // Teste 3: Ambos (acorde)
    Serial.println("3. AMBOS - Acorde");
    tone(SPEAKER_L, 261);  // Do
    tone(SPEAKER_R, 329);  // Mi
    delay(500);
    noTone(SPEAKER_L);
    noTone(SPEAKER_R);
    delay(200);

    // Teste 4: Binaural Beat (10Hz theta)
    Serial.println("4. BINAURAL BEAT - 400Hz + 410Hz = 10Hz theta");
    tone(SPEAKER_L, 400);
    tone(SPEAKER_R, 410);
    delay(3000);  // 3 segundos de binaural
    noTone(SPEAKER_L);
    noTone(SPEAKER_R);
    delay(500);

    // Teste 5: Sweep ascendente
    Serial.println("5. SWEEP - Frequencia subindo");
    for (int freq = 200; freq <= 2000; freq += 50) {
        tone(SPEAKER_L, freq);
        tone(SPEAKER_R, freq);
        delay(30);
    }
    noTone(SPEAKER_L);
    noTone(SPEAKER_R);
    delay(500);

    // Teste 6: Ping-pong estereo
    Serial.println("6. PING-PONG estereo");
    for (int i = 0; i < 5; i++) {
        tone(SPEAKER_L, 800, 100);
        delay(150);
        tone(SPEAKER_R, 1000, 100);
        delay(150);
    }

    Serial.println("--- Ciclo completo. Reiniciando em 3s ---\n");
    delay(3000);
}
