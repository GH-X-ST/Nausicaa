// Nano33IoT_Command_Path_V1
// Serial vector command to trainer PPM output.
//
// Binary packet, 15 bytes:
//   byte 0      : 'V'
//   byte 1      : surface_count = 4
//   byte 2      : active_surface_mask, metadata only
//   byte 3-6    : sample_sequence, uint32 little endian
//   byte 7-14   : four uint16 codes, little endian
//
// MATLAB applies servoSigns before encoding:
//   code = round((clamp(packetSurfaceNorm, -1, 1) + 1) * 0.5 * 65535)
//
// Firmware decodes and maps directly:
//   surfaceNorm = double(code) / 65535.0 * 2.0 - 1.0
//   pulseUs = neutralUs + surfaceNorm * pulseRangeUs
//   pulseUs saturated to [1000, 2000] us.

#include <Arduino.h>
#include <string.h>

namespace Config {
constexpr char kFirmwareVersion[] = "Nano33IoT_Command_Path_V1";
constexpr uint32_t kSerialBaud = 1000000;
constexpr size_t kSurfaceCount = 4;
constexpr size_t kPpmChannelCount = 8;
constexpr size_t kBinaryPacketLength = 15;
constexpr size_t kCommandBufferLength = 80;
constexpr uint8_t kTrainerPin = 3;
constexpr uint8_t kReferencePin = 2;
constexpr uint16_t kFrameLengthUs = 20000;
constexpr uint16_t kMarkWidthUs = 300;
constexpr uint16_t kMinimumPulseUs = 1000;
constexpr uint16_t kNeutralPulseUs = 1500;
constexpr uint16_t kMaximumPulseUs = 2000;
constexpr uint16_t kPulseRangeUs = 500;
constexpr uint32_t kCommandTimeoutUs = 250000;
constexpr uint16_t kTimerTicksPerUs = 3;
}

struct PendingCommand {
  bool valid;
  uint32_t sampleSequence;
  uint8_t activeMask;
  uint32_t rxUs;
  uint16_t codes[Config::kSurfaceCount];
  uint16_t ppmUs[Config::kPpmChannelCount];
};

struct CommitTelemetry {
  uint32_t sampleSequence;
  uint8_t activeMask;
  uint32_t rxUs;
  uint32_t commitUs;
  uint32_t frameIndex;
  uint16_t ppmUs[Config::kPpmChannelCount];
};

char gCommandBuffer[Config::kCommandBufferLength];
size_t gCommandLength = 0;
bool gCommandOverflow = false;

volatile uint16_t gActivePpmUs[Config::kPpmChannelCount] = {
  Config::kNeutralPulseUs, Config::kNeutralPulseUs, Config::kNeutralPulseUs, Config::kNeutralPulseUs,
  Config::kNeutralPulseUs, Config::kNeutralPulseUs, Config::kNeutralPulseUs, Config::kNeutralPulseUs
};
volatile uint16_t gPendingPpmUs[Config::kPpmChannelCount] = {
  Config::kNeutralPulseUs, Config::kNeutralPulseUs, Config::kNeutralPulseUs, Config::kNeutralPulseUs,
  Config::kNeutralPulseUs, Config::kNeutralPulseUs, Config::kNeutralPulseUs, Config::kNeutralPulseUs
};

volatile bool gPendingValid = false;
volatile uint32_t gPendingSequence = 0;
volatile uint8_t gPendingMask = 0;
volatile uint32_t gPendingRxUs = 0;
volatile bool gPulseActive = false;
volatile uint8_t gIntervalIndex = 0;
volatile uint32_t gFrameIndex = 0;
volatile bool gReferenceHigh = false;

volatile uint32_t gRxCount = 0;
volatile uint32_t gCommitCount = 0;
volatile uint32_t gErrorCount = 0;
volatile uint32_t gLatestSequence = 0;
volatile uint32_t gLastCommandRxUs = 0;
volatile bool gTimeoutNeutralQueued = true;
volatile bool gCommitTelemetryPending = false;
volatile CommitTelemetry gCommitTelemetry = {};

void configurePins();
void configureTimer3();
void waitForTimer3Sync();
void scheduleTimerUs(uint16_t durationUs);
void serviceSerialInput();
void finalizeCommandLine();
void resetCommandBuffer();
void handleTextCommand(char* commandBuffer, uint32_t rxUs);
void handleBinaryPacket(const uint8_t* packetBytes, uint32_t rxUs);
bool parseBinaryPacket(const uint8_t* packetBytes, uint32_t rxUs, PendingCommand& command);
void queuePendingCommand(const PendingCommand& command);
void queueNeutralCommand(uint32_t sequence, uint32_t rxUs);
void buildNeutralCommand(PendingCommand& command, uint32_t sequence, uint32_t rxUs);
void serviceCommandTimeout();
void serviceCommitTelemetry();
void emitRxEvent(const PendingCommand& command);
void emitHelloEvent();
void emitStatusEvent();
void emitOkEvent(const __FlashStringHelper* message);
void emitErrorEvent(const __FlashStringHelper* message);
uint16_t codeToPulseUs(uint16_t code);
uint16_t decodeUint16LE(const uint8_t* data);
uint32_t decodeUint32LE(const uint8_t* data);
uint16_t computeSyncGapUs();
inline void setIdleLevel();
inline void setMarkLevel();
inline void referenceHigh();
inline void referenceLow();
inline uint16_t usToTimerTicks(uint16_t durationUs);
inline PortGroup& portGroupForPin(uint8_t pin);
inline uint32_t portMaskForPin(uint8_t pin);

void setup() {
  Serial.begin(Config::kSerialBaud);
  while (!Serial && millis() < 3000UL) {
  }
  Serial.setTimeout(1);
  configurePins();

  PendingCommand neutral = {};
  buildNeutralCommand(neutral, 0U, micros());
  queuePendingCommand(neutral);

  configureTimer3();
  gLastCommandRxUs = micros();
}

void loop() {
  serviceSerialInput();
  serviceCommandTimeout();
  serviceCommitTelemetry();
}

void configurePins() {
  pinMode(Config::kTrainerPin, OUTPUT);
  pinMode(Config::kReferencePin, OUTPUT);
  setIdleLevel();
  referenceLow();
}

void configureTimer3() {
  PM->APBCMASK.reg |= PM_APBCMASK_TC3;
  GCLK->CLKCTRL.reg = static_cast<uint16_t>(
    GCLK_CLKCTRL_CLKEN |
    GCLK_CLKCTRL_GEN_GCLK0 |
    GCLK_CLKCTRL_ID(GCM_TCC2_TC3));
  while (GCLK->STATUS.bit.SYNCBUSY) {
  }

  TC3->COUNT16.CTRLA.reg &= ~TC_CTRLA_ENABLE;
  waitForTimer3Sync();
  TC3->COUNT16.CTRLA.reg =
    TC_CTRLA_MODE_COUNT16 |
    TC_CTRLA_WAVEGEN_MFRQ |
    TC_CTRLA_PRESCALER_DIV16 |
    TC_CTRLA_PRESCSYNC_PRESC;
  waitForTimer3Sync();
  TC3->COUNT16.COUNT.reg = 0;
  waitForTimer3Sync();
  scheduleTimerUs(1000U);
  TC3->COUNT16.INTFLAG.reg = TC_INTFLAG_MC0;
  TC3->COUNT16.INTENSET.reg = TC_INTENSET_MC0;
  NVIC_ClearPendingIRQ(TC3_IRQn);
  NVIC_SetPriority(TC3_IRQn, 0);
  NVIC_EnableIRQ(TC3_IRQn);
  TC3->COUNT16.CTRLA.reg |= TC_CTRLA_ENABLE;
  waitForTimer3Sync();
}

void waitForTimer3Sync() {
  while (TC3->COUNT16.STATUS.bit.SYNCBUSY) {
  }
}

void serviceSerialInput() {
  while (Serial.available() > 0) {
    int nextByte = Serial.peek();
    if (nextByte < 0) {
      return;
    }

    if (static_cast<uint8_t>(nextByte) == static_cast<uint8_t>('V')) {
      if (Serial.available() < static_cast<int>(Config::kBinaryPacketLength)) {
        return;
      }
      uint8_t packetBytes[Config::kBinaryPacketLength];
      for (size_t byteIndex = 0; byteIndex < Config::kBinaryPacketLength; ++byteIndex) {
        int packetByte = Serial.read();
        if (packetByte < 0) {
          emitErrorEvent(F("SHORT_BINARY_PACKET"));
          return;
        }
        packetBytes[byteIndex] = static_cast<uint8_t>(packetByte);
      }
      handleBinaryPacket(packetBytes, micros());
      continue;
    }

    char nextChar = static_cast<char>(Serial.read());
    if (nextChar == '\r') {
      continue;
    }
    if (nextChar == '\n') {
      finalizeCommandLine();
      continue;
    }
    if (gCommandOverflow) {
      continue;
    }
    if (gCommandLength + 1U < Config::kCommandBufferLength) {
      gCommandBuffer[gCommandLength++] = nextChar;
    } else {
      gCommandOverflow = true;
    }
  }
}

void finalizeCommandLine() {
  if (gCommandOverflow) {
    resetCommandBuffer();
    emitErrorEvent(F("COMMAND_TOO_LONG"));
    return;
  }
  if (gCommandLength == 0U) {
    return;
  }
  gCommandBuffer[gCommandLength] = '\0';
  handleTextCommand(gCommandBuffer, micros());
  resetCommandBuffer();
}

void resetCommandBuffer() {
  gCommandLength = 0;
  gCommandOverflow = false;
}

void handleTextCommand(char* commandBuffer, uint32_t rxUs) {
  if (strcmp(commandBuffer, "HELLO") == 0) {
    emitHelloEvent();
    return;
  }
  if (strcmp(commandBuffer, "STATUS") == 0) {
    emitStatusEvent();
    return;
  }
  if (strcmp(commandBuffer, "SET_NEUTRAL") == 0) {
    queueNeutralCommand(0U, rxUs);
    emitOkEvent(F("SET_NEUTRAL"));
    return;
  }
  emitErrorEvent(F("UNKNOWN_COMMAND"));
}

void handleBinaryPacket(const uint8_t* packetBytes, uint32_t rxUs) {
  PendingCommand command = {};
  if (!parseBinaryPacket(packetBytes, rxUs, command)) {
    return;
  }
  queuePendingCommand(command);
}

bool parseBinaryPacket(const uint8_t* packetBytes, uint32_t rxUs, PendingCommand& command) {
  if (packetBytes[0] != static_cast<uint8_t>('V')) {
    emitErrorEvent(F("BAD_PACKET_HEADER"));
    return false;
  }
  if (packetBytes[1] != Config::kSurfaceCount) {
    emitErrorEvent(F("BAD_SURFACE_COUNT"));
    return false;
  }

  command.valid = true;
  command.sampleSequence = decodeUint32LE(packetBytes + 3);
  command.activeMask = packetBytes[2];
  command.rxUs = rxUs;

  size_t readIndex = 7U;
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    command.codes[surfaceIndex] = decodeUint16LE(packetBytes + readIndex);
    command.ppmUs[surfaceIndex] = codeToPulseUs(command.codes[surfaceIndex]);
    readIndex += 2U;
  }
  for (size_t channelIndex = Config::kSurfaceCount; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    command.ppmUs[channelIndex] = Config::kNeutralPulseUs;
  }
  return true;
}

void buildNeutralCommand(PendingCommand& command, uint32_t sequence, uint32_t rxUs) {
  command = {};
  command.valid = true;
  command.sampleSequence = sequence;
  command.activeMask = 0U;
  command.rxUs = rxUs;
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    command.codes[surfaceIndex] = 32768U;
    command.ppmUs[surfaceIndex] = Config::kNeutralPulseUs;
  }
  for (size_t channelIndex = Config::kSurfaceCount; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    command.ppmUs[channelIndex] = Config::kNeutralPulseUs;
  }
}

void queuePendingCommand(const PendingCommand& command) {
  noInterrupts();
  for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    gPendingPpmUs[channelIndex] = command.ppmUs[channelIndex];
  }
  gPendingSequence = command.sampleSequence;
  gPendingMask = command.activeMask;
  gPendingRxUs = command.rxUs;
  gPendingValid = true;
  gLatestSequence = command.sampleSequence;
  gLastCommandRxUs = command.rxUs;
  gTimeoutNeutralQueued = false;
  ++gRxCount;
  interrupts();
  emitRxEvent(command);
}

void queueNeutralCommand(uint32_t sequence, uint32_t rxUs) {
  PendingCommand neutral = {};
  buildNeutralCommand(neutral, sequence, rxUs);
  noInterrupts();
  gTimeoutNeutralQueued = true;
  interrupts();
  queuePendingCommand(neutral);
  noInterrupts();
  gTimeoutNeutralQueued = true;
  interrupts();
}

void serviceCommandTimeout() {
  if (gTimeoutNeutralQueued) {
    return;
  }
  uint32_t nowUs = micros();
  if (static_cast<uint32_t>(nowUs - gLastCommandRxUs) >= Config::kCommandTimeoutUs) {
    queueNeutralCommand(0U, nowUs);
  }
}

void serviceCommitTelemetry() {
  if (!gCommitTelemetryPending) {
    return;
  }
  CommitTelemetry telemetry = {};
  noInterrupts();
  if (!gCommitTelemetryPending) {
    interrupts();
    return;
  }
  telemetry.sampleSequence = gCommitTelemetry.sampleSequence;
  telemetry.activeMask = gCommitTelemetry.activeMask;
  telemetry.rxUs = gCommitTelemetry.rxUs;
  telemetry.commitUs = gCommitTelemetry.commitUs;
  telemetry.frameIndex = gCommitTelemetry.frameIndex;
  for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    telemetry.ppmUs[channelIndex] = gCommitTelemetry.ppmUs[channelIndex];
  }
  gCommitTelemetryPending = false;
  interrupts();

  Serial.print(F("COMMIT_EVENT,"));
  Serial.print(telemetry.sampleSequence);
  Serial.print(',');
  Serial.print(static_cast<uint32_t>(telemetry.activeMask));
  Serial.print(',');
  Serial.print(telemetry.rxUs);
  Serial.print(',');
  Serial.print(telemetry.commitUs);
  Serial.print(',');
  Serial.print(telemetry.frameIndex);
  for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    Serial.print(',');
    Serial.print(telemetry.ppmUs[channelIndex]);
  }
  Serial.println();
}

void emitRxEvent(const PendingCommand& command) {
  Serial.print(F("RX_EVENT,"));
  Serial.print(command.sampleSequence);
  Serial.print(',');
  Serial.print(static_cast<uint32_t>(command.activeMask));
  Serial.print(',');
  Serial.print(command.rxUs);
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    Serial.print(',');
    Serial.print(command.codes[surfaceIndex]);
  }
  Serial.println();
}

void emitHelloEvent() {
  Serial.print(F("HELLO_EVENT,"));
  Serial.print(Config::kFirmwareVersion);
  Serial.print(',');
  Serial.print(Config::kSerialBaud);
  Serial.print(',');
  Serial.println(micros());
}

void emitStatusEvent() {
  noInterrupts();
  uint32_t rxCount = gRxCount;
  uint32_t commitCount = gCommitCount;
  uint32_t errorCount = gErrorCount;
  uint32_t latestSequence = gLatestSequence;
  interrupts();

  Serial.print(F("STATUS_EVENT,rx_count="));
  Serial.print(rxCount);
  Serial.print(F(",commit_count="));
  Serial.print(commitCount);
  Serial.print(F(",error_count="));
  Serial.print(errorCount);
  Serial.print(F(",latest_sequence="));
  Serial.print(latestSequence);
  Serial.print(F(",servo_signs=MATLAB_APPLIED"));
  Serial.print(F(",frame_length_us="));
  Serial.println(Config::kFrameLengthUs);
}

void emitOkEvent(const __FlashStringHelper* message) {
  Serial.print(F("OK_EVENT,"));
  Serial.println(message);
}

void emitErrorEvent(const __FlashStringHelper* message) {
  ++gErrorCount;
  Serial.print(F("ERR_EVENT,"));
  Serial.println(message);
}

uint16_t codeToPulseUs(uint16_t code) {
  double surfaceNorm = (static_cast<double>(code) / 65535.0) * 2.0 - 1.0;
  int32_t pulseUs = static_cast<int32_t>(
    static_cast<double>(Config::kNeutralPulseUs) +
    surfaceNorm * static_cast<double>(Config::kPulseRangeUs) +
    0.5);
  if (pulseUs < static_cast<int32_t>(Config::kMinimumPulseUs)) {
    pulseUs = Config::kMinimumPulseUs;
  } else if (pulseUs > static_cast<int32_t>(Config::kMaximumPulseUs)) {
    pulseUs = Config::kMaximumPulseUs;
  }
  return static_cast<uint16_t>(pulseUs);
}

uint16_t decodeUint16LE(const uint8_t* data) {
  return static_cast<uint16_t>(data[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(data[1]) << 8);
}

uint32_t decodeUint32LE(const uint8_t* data) {
  return static_cast<uint32_t>(data[0]) |
         (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) |
         (static_cast<uint32_t>(data[3]) << 24);
}

inline PortGroup& portGroupForPin(uint8_t pin) {
  return PORT->Group[g_APinDescription[pin].ulPort];
}

inline uint32_t portMaskForPin(uint8_t pin) {
  return (1ul << g_APinDescription[pin].ulPin);
}

inline void setIdleLevel() {
  PortGroup& group = portGroupForPin(Config::kTrainerPin);
  uint32_t mask = portMaskForPin(Config::kTrainerPin);
  group.OUTCLR.reg = mask;
  group.DIRSET.reg = mask;
}

inline void setMarkLevel() {
  PortGroup& group = portGroupForPin(Config::kTrainerPin);
  uint32_t mask = portMaskForPin(Config::kTrainerPin);
  group.OUTSET.reg = mask;
  group.DIRSET.reg = mask;
}

inline void referenceHigh() {
  portGroupForPin(Config::kReferencePin).OUTSET.reg = portMaskForPin(Config::kReferencePin);
}

inline void referenceLow() {
  portGroupForPin(Config::kReferencePin).OUTCLR.reg = portMaskForPin(Config::kReferencePin);
}

inline uint16_t usToTimerTicks(uint16_t durationUs) {
  uint32_t ticks = static_cast<uint32_t>(durationUs) * Config::kTimerTicksPerUs;
  if (ticks == 0U) {
    ticks = 1U;
  }
  if (ticks > 65535U) {
    ticks = 65535U;
  }
  return static_cast<uint16_t>(ticks - 1U);
}

void scheduleTimerUs(uint16_t durationUs) {
  TC3->COUNT16.CC[0].reg = usToTimerTicks(durationUs);
  waitForTimer3Sync();
}

uint16_t computeSyncGapUs() {
  uint32_t slotSumUs = 0U;
  for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    slotSumUs += gActivePpmUs[channelIndex];
  }
  if (slotSumUs + Config::kMarkWidthUs >= Config::kFrameLengthUs) {
    return 4000U;
  }
  return static_cast<uint16_t>(Config::kFrameLengthUs - slotSumUs - Config::kMarkWidthUs);
}

void TC3_Handler() {
  if ((TC3->COUNT16.INTFLAG.reg & TC_INTFLAG_MC0) == 0U) {
    return;
  }
  TC3->COUNT16.INTFLAG.reg = TC_INTFLAG_MC0;

  if (!gPulseActive) {
    if (gIntervalIndex == 0U && gPendingValid) {
      uint32_t commitUs = micros();
      for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
        gActivePpmUs[channelIndex] = gPendingPpmUs[channelIndex];
        gCommitTelemetry.ppmUs[channelIndex] = gPendingPpmUs[channelIndex];
      }
      gCommitTelemetry.sampleSequence = gPendingSequence;
      gCommitTelemetry.activeMask = gPendingMask;
      gCommitTelemetry.rxUs = gPendingRxUs;
      gCommitTelemetry.commitUs = commitUs;
      gCommitTelemetry.frameIndex = gFrameIndex;
      gCommitTelemetryPending = true;
      gPendingValid = false;
      ++gCommitCount;
      referenceHigh();
      gReferenceHigh = true;
    }
    setMarkLevel();
    scheduleTimerUs(Config::kMarkWidthUs);
    gPulseActive = true;
    return;
  }

  if (gReferenceHigh) {
    referenceLow();
    gReferenceHigh = false;
  }

  setIdleLevel();
  if (gIntervalIndex < Config::kPpmChannelCount) {
    uint16_t slotUs = gActivePpmUs[gIntervalIndex];
    uint16_t gapUs = (slotUs > Config::kMarkWidthUs)
      ? static_cast<uint16_t>(slotUs - Config::kMarkWidthUs)
      : 1U;
    ++gIntervalIndex;
    scheduleTimerUs(gapUs);
  } else {
    ++gFrameIndex;
    gIntervalIndex = 0U;
    scheduleTimerUs(computeSyncGapUs());
  }
  gPulseActive = false;
}
