// Arduino Uno trainer-port PPM test
// pin 3  -> trainer tip
// GND    -> trainer sleeve
// pin 2  -> optional marker toggle for analyser alignment

#include <Arduino.h>

// =============================================================================
// SECTION MAP
// =============================================================================
// 1) Constants, Pin Map, and PPM Timing
// 2) Fast Pin Helpers and Channel State
// 3) Timer Setup and PPM ISR
// 4) Setup and Bench-Test Loop
// =============================================================================

// =============================================================================
// 1) Constants, Pin Map, and PPM Timing
// =============================================================================
static const uint8_t PPM_PIN = 3;
static const uint8_t DBG_PIN = 2;

// Trainer-port bench signal uses eight RC channels; only TEST_CHANNEL_INDEX
// moves while the other channels stay neutral for receiver compatibility.
static const uint8_t CHANNEL_COUNT = 8;
static const uint16_t FRAME_US = 22000;
static const uint16_t PULSE_US = 300;
static const uint8_t TEST_CHANNEL_INDEX = 0;

// Bench bring-up pattern:
// keep every channel centered at 1500 us except one selected test channel.
volatile uint16_t g_channels[CHANNEL_COUNT] = {
  1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500
};

volatile uint16_t g_frame_sum_us = 12000;  // sum of 8 channel intervals
volatile uint8_t g_interval_index = 0;     // 0..7 = channels, 8 = sync
volatile bool g_pulse_active = false;

// true  = low baseline, short high pulse  (paper-style figure)
// false = high baseline, short low pulse
static const bool ACTIVE_HIGH_PULSE = true;

void setChannelsAtomic(const uint16_t new_channels[CHANNEL_COUNT]);
void loadAllNeutralAtomic();
void loadTestStep(uint8_t step_index);

// =============================================================================
// 2) Fast Pin Helpers and Channel State
// =============================================================================
// Uno direct-port mapping: pin 3 = PD3, pin 2 = PD2.
inline void ppmPulseOn() {
  if (ACTIVE_HIGH_PULSE) {
    PORTD |= _BV(PD3);
  } else {
    PORTD &= ~_BV(PD3);
  }
}

inline void ppmPulseOff() {
  if (ACTIVE_HIGH_PULSE) {
    PORTD &= ~_BV(PD3);
  } else {
    PORTD |= _BV(PD3);
  }
}

inline uint16_t usToTimerTicks(uint16_t us) {
  // Timer1 runs at 0.5 us/tick with prescaler 8 on the 16 MHz Uno.
  // OCR1A is zero-based, so subtract one after clamping the tick count.
  uint32_t ticks = (uint32_t)us * 2U;
  if (ticks == 0U) {
    ticks = 1U;
  }
  if (ticks > 65535U) {
    ticks = 65535U;
  }
  return (uint16_t)(ticks - 1U);
}

void loadAllNeutralAtomic() {
  const uint16_t neutral[CHANNEL_COUNT] = {
    1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500
  };
  setChannelsAtomic(neutral);
}

void setChannelsAtomic(const uint16_t new_channels[CHANNEL_COUNT]) {
  // Channel values are clamped before the ISR reads them so no measured
  // frame can contain invalid PPM pulse widths.
  noInterrupts();

  uint32_t sum_us = 0U;
  for (uint8_t i = 0; i < CHANNEL_COUNT; ++i) {
    uint16_t v = new_channels[i];

    if (v < 1000U) {
      v = 1000U;
    }
    if (v > 2000U) {
      v = 2000U;
    }

    g_channels[i] = v;
    sum_us += v;
  }

  g_frame_sum_us = (uint16_t)sum_us;
  interrupts();
}

void loadNeutralAter() {
  loadAllNeutralAtomic();
}

void loadTestStep(uint8_t step_index) {
  uint16_t ch[CHANNEL_COUNT] = {
    1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500
  };

  // The four-state pattern isolates one channel transition at a time for
  // logic-analyser bring-up.
  switch (step_index & 0x03U) {
    case 0:
      ch[TEST_CHANNEL_INDEX] = 1000;
      break;
    case 1:
      ch[TEST_CHANNEL_INDEX] = 1500;
      break;
    case 2:
      ch[TEST_CHANNEL_INDEX] = 2000;
      break;
    case 3:
      ch[TEST_CHANNEL_INDEX] = 1500;
      break;
    default:
      break;
  }

  setChannelsAtomic(ch);
}

// =============================================================================
// 3) Timer Setup and PPM ISR
// =============================================================================
void setupTimer1() {
  noInterrupts();

  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 0;

  // CTC mode keeps each PPM mark/gap duration scheduled by OCR1A.
  TCCR1B |= _BV(WGM12);

  // Prescaler 8 preserves sub-microsecond resolution without overflow here.
  TCCR1B |= _BV(CS11);

  // Delayed first compare avoids a boot-edge transient in analyser captures.
  OCR1A = usToTimerTicks(1000);

  TIMSK1 |= _BV(OCIE1A);

  interrupts();
}

ISR(TIMER1_COMPA_vect) {
  if (!g_pulse_active) {
    ppmPulseOn();
    OCR1A = usToTimerTicks(PULSE_US);
    g_pulse_active = true;
    return;
  }

  ppmPulseOff();

  uint16_t next_gap_us = 0;

  if (g_interval_index < CHANNEL_COUNT) {
    uint16_t interval_us = g_channels[g_interval_index];
    if (interval_us < PULSE_US + 50U) {
      interval_us = PULSE_US + 50U;
    }
    next_gap_us = (uint16_t)(interval_us - PULSE_US);
    g_interval_index++;
  } else {
    // Sync gap. This conventional formulation keeps one short pulse
    // between frames and makes total frame length approximately FRAME_US.
    uint16_t sum_us = g_frame_sum_us;
    uint16_t sync_us = (sum_us + PULSE_US < FRAME_US)
      ? (uint16_t)(FRAME_US - sum_us - PULSE_US)
      : 4000U;

    next_gap_us = sync_us;
    g_interval_index = 0;
  }

  OCR1A = usToTimerTicks(next_gap_us);
  g_pulse_active = false;
}

// =============================================================================
// 4) Setup and Bench-Test Loop
// =============================================================================
void setup() {
  pinMode(PPM_PIN, OUTPUT);
  pinMode(DBG_PIN, OUTPUT);

  // Baseline matches the inactive trainer level before Timer1 owns D3.
  ppmPulseOff();
  digitalWrite(DBG_PIN, LOW);

  loadAllNeutralAtomic();
  setupTimer1();

  Serial.begin(115200);
  Serial.println(F("Uno trainer-port PPM test"));
  Serial.print(F("Test channel index: "));
  Serial.println(TEST_CHANNEL_INDEX + 1U);
  Serial.println(F("Pattern: 1000 -> 1500 -> 2000 -> 1500 us"));
  Serial.println(F("All other channels: 1500 us"));
}

void loop() {
  static uint32_t last_update_ms = 0;
  static uint8_t step_index = 0;
  static bool dbg_state = false;

  // Change one test state every 1000 ms
  if (millis() - last_update_ms >= 1000UL) {
    last_update_ms = millis();

    loadTestStep(step_index);
    step_index++;

    dbg_state = !dbg_state;
    digitalWrite(DBG_PIN, dbg_state ? HIGH : LOW);
  }
}
