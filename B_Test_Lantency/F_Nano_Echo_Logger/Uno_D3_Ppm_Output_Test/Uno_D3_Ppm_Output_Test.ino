// Arduino Uno D3 PPM output bench test
// D3  -> analyser probe only for the first validation pass
// D2  -> optional marker pulse for state-change alignment
// GND -> analyser ground

#include <Arduino.h>

// =============================================================================
// SECTION MAP
// =============================================================================
// 1) Constants, Pin Map, and PPM Timing
// 2) Pin Helpers and Timer Scheduling
// 3) Timer ISR and State Loading
// 4) Setup and Bench-Test Loop
// =============================================================================

// =============================================================================
// 1) Constants, Pin Map, and PPM Timing
// =============================================================================
namespace {

constexpr uint8_t kPpmPin = 3;
constexpr uint8_t kMarkerPin = 2;

// D3 carries the trainer PPM signal and D2 is a short analyser marker used
// to verify command-state timing independently of the PPM decoder.
constexpr bool kActiveHighPulse = true;
constexpr uint8_t kChannelCount = 8;
// Bench test uses a 20 ms PPM frame with 300 us marks to match the
// transmitter latency scripts.
constexpr uint16_t kFrameLengthUs = 20000;
constexpr uint16_t kMarkWidthUs = 300;
constexpr uint16_t kMinimumPulseUs = 1000;
constexpr uint16_t kNeutralPulseUs = 1500;
constexpr uint16_t kMaximumPulseUs = 2000;
constexpr uint16_t kMarkerPulseUs = 50;

constexpr uint8_t kTestChannelIndex = 0;
constexpr uint32_t kStateHoldMs = 600;

volatile uint16_t gActivePpmUs[kChannelCount] = {
  kNeutralPulseUs, kNeutralPulseUs, kNeutralPulseUs, kNeutralPulseUs,
  kNeutralPulseUs, kNeutralPulseUs, kNeutralPulseUs, kNeutralPulseUs
};

volatile uint8_t gPpmChannelIndex = 0;
volatile bool gPulseActive = false;

uint16_t gStateSequenceUs[] = {
  kNeutralPulseUs,
  1750,
  kNeutralPulseUs,
  1250,
  kNeutralPulseUs
};

// =============================================================================
// 2) Pin Helpers and Timer Scheduling
// =============================================================================
inline void writePpmHigh() {
  PORTD |= _BV(PD3);
}

inline void writePpmLow() {
  PORTD &= ~_BV(PD3);
}

inline void setPpmMarkLevel() {
  if (kActiveHighPulse) {
    writePpmHigh();
  } else {
    writePpmLow();
  }
}

inline void setPpmIdleLevel() {
  if (kActiveHighPulse) {
    writePpmLow();
  } else {
    writePpmHigh();
  }
}

inline void writeMarkerHigh() {
  PORTD |= _BV(PD2);
}

inline void writeMarkerLow() {
  PORTD &= ~_BV(PD2);
}

uint16_t clampPulseUs(uint16_t pulseUs) {
  if (pulseUs < kMinimumPulseUs) {
    return kMinimumPulseUs;
  }
  if (pulseUs > kMaximumPulseUs) {
    return kMaximumPulseUs;
  }
  return pulseUs;
}

uint16_t usToTimerTicks(uint16_t durationUs) {
  uint32_t ticks = static_cast<uint32_t>(durationUs) * 2U;
  if (ticks == 0U) {
    ticks = 1U;
  }
  if (ticks > 65535U) {
    ticks = 65535U;
  }
  return static_cast<uint16_t>(ticks - 1U);
}

void scheduleTimerUs(uint16_t durationUs) {
  OCR1A = usToTimerTicks(durationUs);
}

uint16_t computeSyncGapUs() {
  // The sync gap fills the remainder of the frame after all channel slots.
  uint32_t slotSumUs = 0U;
  for (uint8_t channelIndex = 0; channelIndex < kChannelCount; ++channelIndex) {
    slotSumUs += gActivePpmUs[channelIndex];
  }

  if (slotSumUs >= kFrameLengthUs) {
    return kMarkWidthUs;
  }
  return static_cast<uint16_t>(kFrameLengthUs - slotSumUs);
}

void configureTimer1() {
  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 0;
  TCCR1B |= _BV(WGM12);
  TCCR1B |= _BV(CS11);
  scheduleTimerUs(1000);
  TIMSK1 |= _BV(OCIE1A);
  interrupts();
}

// =============================================================================
// 3) Timer ISR and State Loading
// =============================================================================
void loadTestState(uint16_t commandedPulseUs) {
  uint16_t pulseUs = clampPulseUs(commandedPulseUs);

  // Only one channel changes during this bench test, making analyser edges
  // attributable to the selected trainer channel.
  noInterrupts();
  for (uint8_t channelIndex = 0; channelIndex < kChannelCount; ++channelIndex) {
    gActivePpmUs[channelIndex] = kNeutralPulseUs;
  }
  gActivePpmUs[kTestChannelIndex] = pulseUs;
  interrupts();

  writeMarkerHigh();
  delayMicroseconds(kMarkerPulseUs);
  writeMarkerLow();

  Serial.print(F("STATE,channel="));
  Serial.print(kTestChannelIndex + 1U);
  Serial.print(F(",pulse_us="));
  Serial.println(pulseUs);
}

}  // namespace

ISR(TIMER1_COMPA_vect) {
  if (!gPulseActive) {
    setPpmMarkLevel();
    scheduleTimerUs(kMarkWidthUs);
    gPulseActive = true;
    return;
  }

  setPpmIdleLevel();

  if (gPpmChannelIndex < kChannelCount) {
    uint16_t slotUs = gActivePpmUs[gPpmChannelIndex];
    uint16_t gapUs = (slotUs > kMarkWidthUs)
      ? static_cast<uint16_t>(slotUs - kMarkWidthUs)
      : 1U;
    ++gPpmChannelIndex;
    scheduleTimerUs(gapUs);
  } else {
    gPpmChannelIndex = 0;
    scheduleTimerUs(computeSyncGapUs());
  }

  gPulseActive = false;
}

// =============================================================================
// 4) Setup and Bench-Test Loop
// =============================================================================
void setup() {
  pinMode(kPpmPin, OUTPUT);
  pinMode(kMarkerPin, OUTPUT);

  // Establish inactive levels before Timer1 starts so boot edges are not
  // mistaken for the first measured PPM transition.
  setPpmIdleLevel();
  writeMarkerLow();

  Serial.begin(115200);
  delay(100);
  Serial.println(F("Uno D3 PPM output test"));
  Serial.println(F("Observe D3 with analyser only for the first pass."));
  Serial.println(F("Pattern: 1500 -> 1750 -> 1500 -> 1250 -> 1500 us on channel 1."));
  Serial.println(F("Frame: 20 ms, mark width: 300 us, marker pulse on D2 at each change."));

  configureTimer1();
  loadTestState(gStateSequenceUs[0]);
}

void loop() {
  static uint32_t lastStateChangeMs = millis();
  static uint8_t stateIndex = 0;

  uint32_t nowMs = millis();
  if (nowMs - lastStateChangeMs < kStateHoldMs) {
    return;
  }

  lastStateChangeMs = nowMs;
  stateIndex = static_cast<uint8_t>((stateIndex + 1U) % (sizeof(gStateSequenceUs) / sizeof(gStateSequenceUs[0])));
  loadTestState(gStateSequenceUs[stateIndex]);
}
