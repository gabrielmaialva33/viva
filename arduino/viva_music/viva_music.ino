/*
 * VIVA Music Bridge - Arduino Controller
 * Hardware: Speaker(D8), Buzzer(D9), Fan PWM(D10), Fan Tach(D2)
 *
 * CHANGE: Fan moved from D11 to D10 to use Timer1 @ 25kHz
 * Intel 4-pin fans require 25kHz PWM
 */

#define SPEAKER_PIN 8
#define BUZZER_PIN 9
#define FAN_PWM_PIN 10 // Moved from D11 to D10 (Timer1 OC1B)
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

// Set fan speed using Timer1 (25kHz PWM)
void setFanSpeed(byte pwm) {
  fanPWM = pwm;
  // Map 0-255 to 0-639 (ICR1 = 639)
  OCR1B = map(pwm, 0, 255, 0, ICR1);
}

void setup() {
  Serial.begin(9600);

  pinMode(SPEAKER_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(FAN_PWM_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(FAN_TACH_PIN, INPUT_PULLUP);

  // Timer1: 25kHz PWM on pin D10 (OC1B)
  // Intel 4-pin fans require 25kHz
  // Fast PWM, TOP = ICR1
  // f_PWM = 16MHz / (1 * (1 + 639)) = 25kHz
  TCCR1A = _BV(COM1B1) | _BV(WGM11);            // Clear OC1B on match, WGM11
  TCCR1B = _BV(WGM13) | _BV(WGM12) | _BV(CS10); // WGM13:12, no prescaler
  ICR1 = 639;                                   // TOP = 639 for 25kHz

  // Set initial speed (50%)
  setFanSpeed(fanPWM);

  // Attach interrupt on FALLING edge (tach pulses low)
  attachInterrupt(digitalPinToInterrupt(FAN_TACH_PIN), countPulse, FALLING);

  // Startup blink
  digitalWrite(LED_PIN, HIGH);
  delay(200);
  digitalWrite(LED_PIN, LOW);

  Serial.println("VIVA_READY");
  Serial.setTimeout(50); // Don't block loop for long (>1s would break RPM calc)
}

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
  // 1. RPM Calculation (every 1 second)
  unsigned long currentMillis = millis();
  unsigned long dt = currentMillis - lastRpmCalc;

  if (dt >= 1000) {
    noInterrupts();
    unsigned int pulses = pulseCount;
    pulseCount = 0;
    interrupts();

    // Normalize by actual time elapsed (in seconds) to handle loop jitter
    // RPM = (Pulses / 2) * (60 / dt_seconds)
    // RPM = (Pulses / 2) * 60000 / dt_ms
    fanRPM = (unsigned int)(((unsigned long)pulses * 30000) / dt);

    lastRpmCalc = currentMillis;
  }

  // 2. Command Processing
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.length() > 0) {
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

      processCommand(cmdPart);
    }
  }
}

void processCommand(String input) {
  char cmd = input.charAt(0);
  String arg = input.length() > 2 ? input.substring(2) : "";

  digitalWrite(LED_PIN, HIGH);

  switch (cmd) {
  case 'P': // Ping
    Serial.println("ACK:PONG");
    break;

  case 'S': // Status
    Serial.print("ACK:PWM:");
    Serial.print(fanPWM);
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

  case 'F': // Fan Control (0-255)
    setFanSpeed(constrain(arg.toInt(), 0, 255));
    Serial.print("ACK:FAN:");
    Serial.println(fanPWM);
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
      Serial.println("NAK:FMT");
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

  case 'T': // Test/Debug
    Serial.print("ACK:PULSES:");
    Serial.print(pulseCount);
    Serial.print(",RPM:");
    Serial.print(fanRPM);
    Serial.print(",PWM:");
    Serial.print(fanPWM);
    Serial.print(",OCR1B:");
    Serial.print(OCR1B);
    Serial.print(",TACH:");
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
  delay(20);
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
  byte pwm;
  if (emotion == "joy")
    pwm = 200;
  else if (emotion == "sad")
    pwm = 80;
  else if (emotion == "fear")
    pwm = 255;
  else if (emotion == "calm")
    pwm = 60;
  else if (emotion == "curious")
    pwm = 150;
  else if (emotion == "love")
    pwm = 120;
  else
    pwm = 128;

  setFanSpeed(pwm);
}

void playEmotionMelody(String emotion) {
  if (emotion == "joy") {
    playNote(262, 100); // C4
    playNote(330, 100); // E4
    playNote(392, 100); // G4
    playNote(523, 200); // C5
  } else if (emotion == "sad") {
    playNote(440, 300); // A4
    playNote(392, 300); // G4
    playNote(330, 300); // E4
    playNote(294, 500); // D4
  } else if (emotion == "fear") {
    for (int i = 0; i < 5; i++) {
      playNote(233, 50); // Bb3
      playNote(247, 50); // B3
    }
  } else if (emotion == "calm") {
    playNote(262, 300); // C4
    playNote(330, 300); // E4
    playNote(392, 500); // G4
  } else if (emotion == "curious") {
    playNote(262, 100); // C4
    playNote(294, 100); // D4
    playNote(330, 100); // E4
    playNote(392, 200); // G4
  } else if (emotion == "love") {
    playNote(262, 150); // C4
    playNote(330, 150); // E4
    playNote(392, 150); // G4
    playNote(330, 150); // E4
    playNote(262, 300); // C4
  }
}
