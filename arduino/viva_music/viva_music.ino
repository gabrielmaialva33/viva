/*
 * VIVA Music Bridge - Arduino Controller
 * Hardware: Speaker(D8), Buzzer(D9), Fan PWM(D11), Fan Tach(D2)
 */

#define SPEAKER_PIN 8
#define BUZZER_PIN 9
#define FAN_PWM_PIN 11
#define FAN_TACH_PIN 2 // INT0
#define LED_PIN 13

// RPM counting
volatile unsigned long pulseCount = 0;
unsigned long lastRpmCalc = 0;
unsigned int fanRPM = 0;

// Fan state
byte fanPWM = 128;
bool harmonyEnabled = true;

// Interrupt handler - counts TACH pulses
void countPulse() { pulseCount++; }

void setup() {
  Serial.begin(9600);

  pinMode(SPEAKER_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(FAN_PWM_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(FAN_TACH_PIN, INPUT_PULLUP); // Internal pull-up

  // Attach interrupt on FALLING edge (tach pulses low)
  attachInterrupt(digitalPinToInterrupt(FAN_TACH_PIN), countPulse, FALLING);

  analogWrite(FAN_PWM_PIN, fanPWM);
  digitalWrite(LED_PIN, HIGH);
  delay(200);
  digitalWrite(LED_PIN, LOW);

  Serial.println("VIVA_READY");
}

// CRC32 Lookup Table (Polynomial 0x04C11DB7)
// Partial table or software calculation to save space?
// Let's use a software calculation for simplicity and code size.

uint32_t calculateCRC32(String data) {
  uint32_t crc = 0xFFFFFFFF;
  for (int i = 0; i < data.length(); i++) {
    char c = data.charAt(i);
    crc ^= c;
    for (int j = 0; j < 8; j++) {
      if (crc & 1)
        crc = (crc >> 1) ^ 0xEDB88320;
      else
        crc >>= 1;
    }
  }
  return ~crc;
}

void loop() {
  // 1. Hardware Monitoring
  unsigned long currentMillis = millis();
  if (currentMillis - lastRpmCalc >= 1000) {
    noInterrupts();
    unsigned int pulses = pulseCount;
    pulseCount = 0;
    interrupts();
    // 2 pulses per rotation for standard fans
    fanRPM = (pulses / 2) * 60;
    lastRpmCalc = currentMillis;
  }

  // 2. Command Processing
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.length() > 0) {
      // Expected format: CMD ARG|CRC32
      // Example: N 440 200|A1B2C3D4

      int separator = input.lastIndexOf('|');
      if (separator == -1) {
        Serial.println("NAK:NO_CRC");
        return;
      }

      String cmdPart = input.substring(0, separator);
      String crcPart = input.substring(separator + 1);

      uint32_t calced = calculateCRC32(cmdPart);
      uint32_t received = strtoul(crcPart.c_str(), NULL, 16);

      if (calced != received) {
        Serial.print("NAK:CRC_FAIL:");
        Serial.print(calced, HEX);
        Serial.print("!=");
        Serial.println(received, HEX);
        return;
      }

      // CRC Valid -> Execute and ACK
      processCommand(cmdPart);
    }
  }
}

void processCommand(String input) {
  char cmd = input.charAt(0);
  String arg = input.length() > 2 ? input.substring(2) : "";

  digitalWrite(LED_PIN, HIGH);

  switch (cmd) {
  case 'P': // Ping -> Pong
    Serial.println("ACK:PONG");
    break;

  case 'S': // Status
    Serial.print("ACK:PWM:");
    Serial.print(fanSpeed);
    Serial.print(",RPM:");
    Serial.print(fanRPM);
    Serial.print(",HARMONY:");
    Serial.println(harmonyEnabled ? "ON" : "OFF");
    break;

  case 'R': // RPM Only
    Serial.print("ACK:RPM:");
    Serial.println(fanRPM);
    break;

  case 'E': // Express Emotion
    setEmotionFan(arg);
    playEmotionMelody(arg);
    Serial.print("ACK:EMOTION:");
    Serial.println(arg);
    break;

  case 'F': // Fan Control
    fanSpeed = constrain(arg.toInt(), 0, 255);
    analogWrite(FAN_PWM_PIN, fanSpeed);
    Serial.print("ACK:FAN:");
    Serial.println(fanSpeed);
    break;

  case 'N': // Play Note
  {
    int spaceIdx = arg.indexOf(' ');
    if (spaceIdx > 0) {
      int freq = arg.substring(0, spaceIdx).toInt();
      int dur = arg.substring(spaceIdx + 1).toInt();
      playNote(freq, dur);
      Serial.println("ACK:OK");
    } else {
      Serial.println("NAK:fmt");
    }
  } break;

  case 'M': // Play Melody
    playMelody(arg);
    Serial.println("ACK:MELODY_DONE");
    break;

  case 'H': // Harmony Toggle
    harmonyEnabled = (arg.charAt(0) == '1');
    Serial.print("ACK:HARMONY:");
    Serial.println(harmonyEnabled ? "ON" : "OFF");
    break;

  case 'T': // Test RPM (debug)
    Serial.print("ACK:PULSES:");
    Serial.print(pulseCount);
    Serial.print(",RPM:");
    Serial.print(fanRPM);
    Serial.print(",TACH_PIN:");
    Serial.println(digitalRead(FAN_TACH_PIN));
    break;

  default:
    Serial.println("NAK:UNKNOWN_CMD");
  }

  digitalWrite(LED_PIN, LOW);
}

void playNote(int freq, int dur) {
  if (freq > 0) {
    tone(SPEAKER_PIN, freq, dur);
    if (harmonyEnabled) {
      tone(BUZZER_PIN, freq * 2, dur); // Octave up
    }
    digitalWrite(LED_PIN, HIGH);
    delay(dur);
    digitalWrite(LED_PIN, LOW);
  } else {
    delay(dur); // Rest
  }
  noTone(SPEAKER_PIN);
  noTone(BUZZER_PIN);
  delay(20); // Gap between notes
}

void playMelody(String melodyStr) {
  int start = 0;
  while (start < melodyStr.length()) {
    int semicolon = melodyStr.indexOf(';', start);
    if (semicolon == -1)
      semicolon = melodyStr.length();

    String noteStr = melodyStr.substring(start, semicolon);
    int comma = noteStr.indexOf(',');

    if (comma > 0) {
      int freq = noteStr.substring(0, comma).toInt();
      int dur = noteStr.substring(comma + 1).toInt();
      playNote(freq, dur);
    }

    start = semicolon + 1;
  }
}

void setEmotionFan(String emotion) {
  if (emotion == "joy")
    fanPWM = 200;
  else if (emotion == "sad")
    fanPWM = 80;
  else if (emotion == "fear")
    fanPWM = 255;
  else if (emotion == "calm")
    fanPWM = 60;
  else if (emotion == "curious")
    fanPWM = 150;
  else if (emotion == "love")
    fanPWM = 120;
  else
    fanPWM = 128;

  analogWrite(FAN_PWM_PIN, fanPWM);
}

void playEmotionMelody(String emotion) {
  if (emotion == "joy") {
    playNote(262, 100);
    playNote(330, 100);
    playNote(392, 100);
    playNote(523, 200);
  } else if (emotion == "sad") {
    playNote(440, 300);
    playNote(392, 300);
    playNote(330, 300);
    playNote(294, 500);
  } else if (emotion == "fear") {
    for (int i = 0; i < 5; i++) {
      playNote(233, 50);
      playNote(247, 50);
    }
  } else if (emotion == "calm") {
    playNote(262, 300);
    playNote(330, 300);
    playNote(392, 500);
  } else if (emotion == "curious") {
    playNote(262, 100);
    playNote(294, 100);
    playNote(330, 100);
    playNote(392, 200);
  } else if (emotion == "love") {
    playNote(262, 150);
    playNote(330, 150);
    playNote(392, 150);
    playNote(330, 150);
    playNote(262, 300);
  }
}
