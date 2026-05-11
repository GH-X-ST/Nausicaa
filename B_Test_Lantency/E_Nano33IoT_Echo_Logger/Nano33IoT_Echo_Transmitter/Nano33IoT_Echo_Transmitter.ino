// Arduino Nano 33 IoT live PPM transmitter
// Minimal real-time architecture for trainer-port control.
// D3 -> trainer PPM output
// D2 -> diagnostic/reference marker pulse

#include <Arduino.h>
#include <stdlib.h>
#include <string.h>

// =============================================================================
// SECTION MAP
// =============================================================================
// 1) Constants, Pin Map, and Packet Layout
// 2) Setup and Service Loop
// 3) Serial Command Parsing and Queueing
// 4) Telemetry and Counters
// 5) Pin/Timer Helpers and ISR
// =============================================================================

// =============================================================================
// 1) Constants, Pin Map, and Packet Layout
// =============================================================================
namespace Config {
constexpr char kFirmwareVersion[] = "Nano33IoT_Echo_Transmitter_V1_RT";
// 1 Mbaud keeps serial transfer below the 20 ms PPM frame budget.
constexpr uint32_t kSerialBaud = 1000000;

constexpr size_t kSurfaceCount = 4;
// Trainer output uses eight RC channels; channels 1-4 are the active
// surfaces and channels 5-8 remain neutral to preserve receiver framing.
constexpr size_t kPpmChannelCount = 8;
// Binary vector packet: 'V', surface count, active mask, uint32 sequence,
// then four little-endian uint16 normalized positions.
constexpr size_t kBinaryVectorPacketLength = 15;
constexpr size_t kCommandBufferLength = 192;

constexpr uint8_t kTrainerPin = 3;
constexpr uint8_t kReferencePin = 2;

// Polarity and output mode must match the trainer-port wiring and analyser
// probe assumptions used by Transmitter_Test.m.
constexpr bool kUseOpenDrainTrainerOutput = false;
constexpr bool kUseTrainerInputPullupWhenReleased = false;
constexpr bool kPpmActiveHighPulse = true;

constexpr uint16_t kMinimumPulseUs = 1000;
constexpr uint16_t kMaximumPulseUs = 2000;
constexpr uint16_t kNeutralPulseUs = 1500;
constexpr int16_t kSurfacePulseTrimUs[kSurfaceCount] = {-2, -2, -12, -2};

// PPM timing follows the receiver trainer-port convention: 300 us mark
// pulses in a 20 ms frame, with failsafe neutral after command timeout.
constexpr uint16_t kFrameLengthUs = 20000;
constexpr uint16_t kMarkWidthUs = 300;
constexpr uint32_t kCommandTimeoutUs = 250000;
constexpr bool kUseInternalDiagnosticPattern = false;
constexpr uint8_t kDiagnosticSurfaceIndex = 0;
constexpr uint32_t kDiagnosticHoldUs = 600000;
constexpr uint16_t kDiagnosticPulseSequenceUs[] = {
  kNeutralPulseUs,
  1750,
  kNeutralPulseUs,
  1250,
  kNeutralPulseUs
};
constexpr uint16_t kDiagnosticMarkerPulseUs = 50;

constexpr uint16_t kTimerTicksPerUs = 3;

// Surface order is shared with MATLAB dispatch rows and logic-analyser
// matching; trims are applied before channel values enter the ISR.
constexpr char kSurfaceNames[kSurfaceCount][16] = {
  "Aileron_L",
  "Aileron_R",
  "Rudder",
  "Elevator"
};
}

struct PendingCommand {
  bool isValid;
  uint32_t sampleSequence;
  uint8_t activeMask;
  uint32_t rxUs;
  uint16_t positionCodes[Config::kSurfaceCount];
  uint16_t ppmUs[Config::kPpmChannelCount];
};

struct CommitTelemetry {
  uint32_t sampleSequence;
  uint8_t activeMask;
  uint32_t boardRxUs;
  uint32_t boardCommitUs;
  uint32_t receiveToCommitUs;
  uint32_t strobeUs;
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

volatile bool gPendingVectorValid = false;
volatile uint32_t gPendingSampleSequence = 0;
volatile uint8_t gPendingActiveMask = 0;
volatile uint32_t gPendingRxUs = 0;

volatile bool gPpmPulseActive = false;
volatile uint8_t gPpmIntervalIndex = 0;
volatile uint32_t gFrameIndex = 0;
volatile bool gReferencePulseActive = false;

volatile uint32_t gRxEventCount = 0;
volatile uint32_t gAppliedUpdateCount = 0;
volatile uint32_t gOverwriteCount = 0;
volatile uint32_t gTimeoutNeutralCount = 0;
volatile uint32_t gErrorCount = 0;
volatile uint32_t gLatestSequence = 0;
volatile uint32_t gLastCommandRxUs = 0;
volatile bool gTimeoutNeutralQueued = true;
volatile bool gCommitTelemetryPending = false;
volatile CommitTelemetry gCommitTelemetry = {};

uint32_t gDiagnosticLastUpdateUs = 0;
uint8_t gDiagnosticStateIndex = 0;

void configurePins();
void configureTimer3();
void waitForTimer3Sync();
void scheduleTimerUs(uint16_t durationUs);
uint16_t computeSyncGapUs();
void buildDiagnosticCommand(PendingCommand& candidate, uint8_t stateIndex, uint32_t boardRxUs);
void serviceInternalDiagnosticPattern();
void serviceSerialInput();
void finalizeCommandLine();
void resetCommandBuffer();
void handleTextCommand(char* commandBuffer, uint32_t boardRxUs);
void handleBinaryVectorPacket(const uint8_t* packetBytes, uint32_t boardRxUs);
bool tryParseBinaryVectorCommand(const uint8_t* packetBytes, uint32_t boardRxUs, PendingCommand& candidate);
bool tryHandleSetAllCommand(char* commandBuffer, uint32_t boardRxUs);
void finalizeVectorCandidate(PendingCommand& candidate);
void buildNeutralCommand(PendingCommand& candidate, uint32_t sampleSequence, uint32_t boardRxUs);
void queuePendingVector(const PendingCommand& command);
void queueNeutralVector(uint32_t sampleSequence, uint32_t boardRxUs);
void serviceCommandTimeout();
void handleSyncCommand(char* context, uint32_t boardRxUs);
void emitRxEvent(const PendingCommand& command);
void serviceCommitTelemetry();
void sendHelloEvent();
void sendStatusEvent();
void sendOkEvent(const __FlashStringHelper* message);
void sendErrorEvent(const __FlashStringHelper* message);
void clearCounters();
int findSurfaceIndex(const char* surfaceName);
bool parseUnsigned32(const char* token, uint32_t& value);
bool parseFloat(const char* token, float& value);
uint16_t decodeUint16LittleEndian(const uint8_t* data);
uint32_t decodeUint32LittleEndian(const uint8_t* data);
uint16_t positionNormToCode(float positionNorm);
uint16_t positionCodeToPulseUs(uint16_t positionCode);
uint16_t applySurfacePulseTrimUs(size_t surfaceIndex, uint16_t pulseUs);
inline void driveTrainerLow();
inline void releaseTrainerOutput();
inline void setPpmIdleLevel();
inline void setPpmMarkLevel();
inline void writeReferenceHigh();
inline void writeReferenceLow();
inline uint16_t usToTimerTicks(uint16_t durationUs);
inline PortGroup& portGroupForPin(uint8_t pin);
inline uint32_t portMaskForPin(uint8_t pin);

// =============================================================================
// 2) Setup and Service Loop
// =============================================================================
void setup() {
  Serial.begin(Config::kSerialBaud);
  while (!Serial && millis() < 3000UL) {
  }

  Serial.setTimeout(1);
  configurePins();

  PendingCommand neutralCommand = {};
  // Start with a queued neutral vector so the first committed PPM frame is
  // safe even if MATLAB connects after the timer is running.
  buildNeutralCommand(neutralCommand, 0U, micros());
  queuePendingVector(neutralCommand);

  configureTimer3();
  gLastCommandRxUs = micros();
  gDiagnosticLastUpdateUs = micros();
}

void loop() {
  if (Config::kUseInternalDiagnosticPattern) {
    serviceInternalDiagnosticPattern();
    return;
  }

  serviceSerialInput();
  serviceCommandTimeout();
  serviceCommitTelemetry();
}

inline PortGroup& portGroupForPin(uint8_t pin) {
  return PORT->Group[g_APinDescription[pin].ulPort];
}

inline uint32_t portMaskForPin(uint8_t pin) {
  return (1ul << g_APinDescription[pin].ulPin);
}

void configurePins() {
  pinMode(Config::kReferencePin, OUTPUT);
  pinMode(Config::kTrainerPin, OUTPUT);
  setPpmIdleLevel();
  writeReferenceLow();
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

// =============================================================================
// 3) Serial Command Parsing and Queueing
// =============================================================================
void serviceSerialInput() {
  while (Serial.available() > 0) {
    int nextByte = Serial.peek();
    if (nextByte < 0) {
      return;
    }

    if (static_cast<uint8_t>(nextByte) == static_cast<uint8_t>('V')) {
      if (Serial.available() < static_cast<int>(Config::kBinaryVectorPacketLength)) {
        return;
      }

      uint8_t packetBytes[Config::kBinaryVectorPacketLength];
      Serial.readBytes(reinterpret_cast<char*>(packetBytes), Config::kBinaryVectorPacketLength);
      handleBinaryVectorPacket(packetBytes, micros());
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
      gCommandBuffer[gCommandLength] = nextChar;
      ++gCommandLength;
    } else {
      gCommandOverflow = true;
    }
  }
}

void finalizeCommandLine() {
  if (gCommandOverflow) {
    resetCommandBuffer();
    sendErrorEvent(F("COMMAND_TOO_LONG"));
    return;
  }

  if (gCommandLength == 0U) {
    return;
  }

  gCommandBuffer[gCommandLength] = '\0';
  // boardRxUs is captured at packet completion and becomes RX_EVENT
  // telemetry for MATLAB's board-micros to host-time conversion.
  handleTextCommand(gCommandBuffer, micros());
  resetCommandBuffer();
}

void resetCommandBuffer() {
  gCommandLength = 0;
  gCommandOverflow = false;
}

void handleTextCommand(char* commandBuffer, uint32_t boardRxUs) {
  if (tryHandleSetAllCommand(commandBuffer, boardRxUs)) {
    return;
  }

  char* context = nullptr;
  char* commandName = strtok_r(commandBuffer, ",", &context);
  if (commandName == nullptr) {
    return;
  }

  if (strcmp(commandName, "HELLO") == 0) {
    sendHelloEvent();
    return;
  }

  if (strcmp(commandName, "STATUS") == 0) {
    sendStatusEvent();
    return;
  }

  if (strcmp(commandName, "CLEAR_LOGS") == 0) {
    clearCounters();
    sendOkEvent(F("CLEAR_LOGS"));
    return;
  }

  if (strcmp(commandName, "SET_NEUTRAL") == 0) {
    queueNeutralVector(0U, boardRxUs);
    sendOkEvent(F("SET_NEUTRAL"));
    return;
  }

  if (strcmp(commandName, "SYNC") == 0) {
    handleSyncCommand(context, boardRxUs);
    return;
  }

  sendErrorEvent(F("UNKNOWN_COMMAND"));
}

void handleBinaryVectorPacket(const uint8_t* packetBytes, uint32_t boardRxUs) {
  PendingCommand candidate = {};
  if (!tryParseBinaryVectorCommand(packetBytes, boardRxUs, candidate)) {
    return;
  }

  queuePendingVector(candidate);
}

bool tryParseBinaryVectorCommand(const uint8_t* packetBytes, uint32_t boardRxUs, PendingCommand& candidate) {
  if (packetBytes[0] != static_cast<uint8_t>('V')) {
    return false;
  }

  // Surface-count validation protects the channel order contract between
  // MATLAB, firmware, and the receiver PPM decoder.
  if (packetBytes[1] != Config::kSurfaceCount) {
    sendErrorEvent(F("BINARY_VECTOR_SURFACE_COUNT_ERROR"));
    return false;
  }

  candidate = {};
  candidate.isValid = true;
  candidate.sampleSequence = decodeUint32LittleEndian(packetBytes + 3);
  candidate.activeMask = packetBytes[2];
  candidate.rxUs = boardRxUs;

  size_t readIndex = 7U;
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    candidate.positionCodes[surfaceIndex] = decodeUint16LittleEndian(packetBytes + readIndex);
    readIndex += 2U;
  }

  finalizeVectorCandidate(candidate);
  return true;
}

bool tryHandleSetAllCommand(char* commandBuffer, uint32_t boardRxUs) {
  if (strncmp(commandBuffer, "SET_ALL", 7) != 0 ||
      (commandBuffer[7] != ',' && commandBuffer[7] != '\0')) {
    return false;
  }

  char parseBuffer[Config::kCommandBufferLength];
  strncpy(parseBuffer, commandBuffer, Config::kCommandBufferLength - 1);
  parseBuffer[Config::kCommandBufferLength - 1] = '\0';

  char* context = nullptr;
  strtok_r(parseBuffer, ",", &context);
  char* sampleSequenceToken = strtok_r(nullptr, ",", &context);
  char* surfaceCountToken = strtok_r(nullptr, ",", &context);

  uint32_t sampleSequence = 0;
  uint32_t surfaceCount = 0;
  if (!parseUnsigned32(sampleSequenceToken, sampleSequence) ||
      !parseUnsigned32(surfaceCountToken, surfaceCount)) {
    sendErrorEvent(F("SET_ALL_PARSE_ERROR"));
    return true;
  }

  if (surfaceCount == 0U || surfaceCount > Config::kSurfaceCount) {
    sendErrorEvent(F("SET_ALL_SURFACE_COUNT_ERROR"));
    return true;
  }

  PendingCommand candidate = {};
  candidate.isValid = true;
  candidate.sampleSequence = sampleSequence;
  candidate.activeMask = 0U;
  candidate.rxUs = boardRxUs;
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    candidate.positionCodes[surfaceIndex] = positionNormToCode(0.5f);
  }

  for (uint32_t elementIndex = 0; elementIndex < surfaceCount; ++elementIndex) {
    char* surfaceNameToken = strtok_r(nullptr, ",", &context);
    char* sequenceToken = strtok_r(nullptr, ",", &context);
    char* positionToken = strtok_r(nullptr, ",", &context);

    int surfaceIndex = findSurfaceIndex(surfaceNameToken);
    uint32_t commandSequence = 0;
    float positionNorm = 0.0f;
    if (surfaceIndex < 0 ||
        !parseUnsigned32(sequenceToken, commandSequence) ||
        !parseFloat(positionToken, positionNorm)) {
      sendErrorEvent(F("SET_ALL_ITEM_PARSE_ERROR"));
      return true;
    }

    (void)commandSequence;
    candidate.activeMask |= static_cast<uint8_t>(1U << static_cast<uint8_t>(surfaceIndex));
    candidate.positionCodes[surfaceIndex] = positionNormToCode(positionNorm);
  }

  finalizeVectorCandidate(candidate);
  queuePendingVector(candidate);
  return true;
}

void finalizeVectorCandidate(PendingCommand& candidate) {
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    candidate.ppmUs[surfaceIndex] = applySurfacePulseTrimUs(
      surfaceIndex,
      positionCodeToPulseUs(candidate.positionCodes[surfaceIndex]));
  }

  for (size_t channelIndex = Config::kSurfaceCount; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    candidate.ppmUs[channelIndex] = Config::kNeutralPulseUs;
  }
}

void buildNeutralCommand(PendingCommand& candidate, uint32_t sampleSequence, uint32_t boardRxUs) {
  candidate = {};
  candidate.isValid = true;
  candidate.sampleSequence = sampleSequence;
  candidate.activeMask = 0U;
  candidate.rxUs = boardRxUs;

  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    candidate.positionCodes[surfaceIndex] = positionNormToCode(0.5f);
  }

  finalizeVectorCandidate(candidate);
}

void queuePendingVector(const PendingCommand& command) {
  noInterrupts();
  if (gPendingVectorValid) {
    ++gOverwriteCount;
  }
  for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    gPendingPpmUs[channelIndex] = command.ppmUs[channelIndex];
  }
  gPendingSampleSequence = command.sampleSequence;
  gPendingActiveMask = command.activeMask;
  gPendingRxUs = command.rxUs;
  gPendingVectorValid = true;
  gLatestSequence = command.sampleSequence;
  gLastCommandRxUs = command.rxUs;
  gTimeoutNeutralQueued = false;
  ++gRxEventCount;
  interrupts();

  emitRxEvent(command);
}

void queueNeutralVector(uint32_t sampleSequence, uint32_t boardRxUs) {
  PendingCommand neutralCommand = {};
  buildNeutralCommand(neutralCommand, sampleSequence, boardRxUs);

  noInterrupts();
  for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    gPendingPpmUs[channelIndex] = neutralCommand.ppmUs[channelIndex];
  }
  gPendingSampleSequence = neutralCommand.sampleSequence;
  gPendingActiveMask = neutralCommand.activeMask;
  gPendingRxUs = neutralCommand.rxUs;
  gPendingVectorValid = true;
  gLastCommandRxUs = boardRxUs;
  gTimeoutNeutralQueued = true;
  interrupts();

  emitRxEvent(neutralCommand);
}

void serviceCommandTimeout() {
  if (gTimeoutNeutralQueued) {
    return;
  }

  uint32_t nowUs = micros();
  if (static_cast<uint32_t>(nowUs - gLastCommandRxUs) < Config::kCommandTimeoutUs) {
    return;
  }

  ++gTimeoutNeutralCount;
  // Timeout drives all channels to neutral to prevent a stale command from
  // persisting when MATLAB or the serial link stops.
  queueNeutralVector(0U, nowUs);
}

void handleSyncCommand(char* context, uint32_t boardRxUs) {
  char* syncIdToken = strtok_r(nullptr, ",", &context);
  char* hostTxUsToken = strtok_r(nullptr, ",", &context);

  uint32_t syncId = 0;
  uint32_t hostTxUs = 0;
  if (!parseUnsigned32(syncIdToken, syncId) || !parseUnsigned32(hostTxUsToken, hostTxUs)) {
    sendErrorEvent(F("SYNC_PARSE_ERROR"));
    return;
  }

  uint32_t boardTxUs = micros();
  // SYNC_EVENT is the clock-map probe: host TX micros from MATLAB plus
  // board RX/TX micros from this firmware.
  Serial.print(F("SYNC_EVENT,"));
  Serial.print(syncId);
  Serial.print(',');
  Serial.print(hostTxUs);
  Serial.print(',');
  Serial.print(boardRxUs);
  Serial.print(',');
  Serial.println(boardTxUs);
}

// =============================================================================
// 4) Telemetry and Counters
// =============================================================================
void emitRxEvent(const PendingCommand& command) {
  // RX_EVENT records accepted commands before the ISR commits them to PPM.
  Serial.print(F("RX_EVENT,"));
  Serial.print(command.sampleSequence);
  Serial.print(',');
  Serial.print(static_cast<uint32_t>(command.activeMask));
  Serial.print(',');
  Serial.print(command.rxUs);
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    Serial.print(',');
    Serial.print(command.positionCodes[surfaceIndex]);
  }
  Serial.println();
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
  telemetry.boardRxUs = gCommitTelemetry.boardRxUs;
  telemetry.boardCommitUs = gCommitTelemetry.boardCommitUs;
  telemetry.receiveToCommitUs = gCommitTelemetry.receiveToCommitUs;
  telemetry.strobeUs = gCommitTelemetry.strobeUs;
  telemetry.frameIndex = gCommitTelemetry.frameIndex;
  for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    telemetry.ppmUs[channelIndex] = gCommitTelemetry.ppmUs[channelIndex];
  }
  gCommitTelemetryPending = false;
  interrupts();

  // COMMIT_EVENT is emitted outside the ISR but carries ISR-captured commit
  // micros, frame index, reference strobe time, and all eight PPM channels.
  Serial.print(F("COMMIT_EVENT,"));
  Serial.print(telemetry.sampleSequence);
  Serial.print(',');
  Serial.print(static_cast<uint32_t>(telemetry.activeMask));
  Serial.print(',');
  Serial.print(telemetry.boardRxUs);
  Serial.print(',');
  Serial.print(telemetry.boardCommitUs);
  Serial.print(',');
  Serial.print(telemetry.strobeUs);
  Serial.print(',');
  Serial.print(telemetry.frameIndex);
  for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    Serial.print(',');
    Serial.print(telemetry.ppmUs[channelIndex]);
  }
  Serial.print(',');
  Serial.print(telemetry.receiveToCommitUs);
  Serial.println();
}

void sendHelloEvent() {
  Serial.print(F("HELLO_EVENT,"));
  Serial.print(Config::kFirmwareVersion);
  Serial.print(',');
  Serial.print(Config::kSerialBaud);
  Serial.print(',');
  Serial.println(micros());
}

void sendStatusEvent() {
  noInterrupts();
  uint32_t rxEventCount = gRxEventCount;
  uint32_t appliedUpdateCount = gAppliedUpdateCount;
  uint32_t overwriteCount = gOverwriteCount;
  uint32_t timeoutNeutralCount = gTimeoutNeutralCount;
  uint32_t errorCount = gErrorCount;
  uint32_t latestSequence = gLatestSequence;
  bool pendingVectorValid = gPendingVectorValid;
  interrupts();

  Serial.print(F("STATUS_EVENT,rx_event_count="));
  Serial.print(rxEventCount);
  Serial.print(F(",applied_update_count="));
  Serial.print(appliedUpdateCount);
  Serial.print(F(",overwrite_count="));
  Serial.print(overwriteCount);
  Serial.print(F(",timeout_neutral_count="));
  Serial.print(timeoutNeutralCount);
  Serial.print(F(",error_count="));
  Serial.print(errorCount);
  Serial.print(F(",latest_sequence="));
  Serial.print(latestSequence);
  Serial.print(F(",pending_vector="));
  Serial.print(pendingVectorValid ? 1 : 0);
  Serial.print(F(",telemetry_mode=event_stream"));
  Serial.print(F(",frame_length_us="));
  Serial.print(Config::kFrameLengthUs);
  Serial.print(F(",mark_width_us="));
  Serial.println(Config::kMarkWidthUs);
}

void sendOkEvent(const __FlashStringHelper* message) {
  Serial.print(F("OK_EVENT,"));
  Serial.println(message);
}

void sendErrorEvent(const __FlashStringHelper* message) {
  ++gErrorCount;
  Serial.print(F("ERR_EVENT,"));
  Serial.println(message);
}

void clearCounters() {
  noInterrupts();
  gRxEventCount = 0U;
  gAppliedUpdateCount = 0U;
  gOverwriteCount = 0U;
  gTimeoutNeutralCount = 0U;
  gErrorCount = 0U;
  gLatestSequence = 0U;
  interrupts();
}

int findSurfaceIndex(const char* surfaceName) {
  if (surfaceName == nullptr) {
    return -1;
  }

  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    if (strcmp(surfaceName, Config::kSurfaceNames[surfaceIndex]) == 0) {
      return static_cast<int>(surfaceIndex);
    }
  }

  return -1;
}

bool parseUnsigned32(const char* token, uint32_t& value) {
  if (token == nullptr) {
    return false;
  }

  char* parseEnd = nullptr;
  unsigned long parsedValue = strtoul(token, &parseEnd, 10);
  if (parseEnd == token || *parseEnd != '\0') {
    return false;
  }

  value = static_cast<uint32_t>(parsedValue);
  return true;
}

bool parseFloat(const char* token, float& value) {
  if (token == nullptr) {
    return false;
  }

  char* parseEnd = nullptr;
  double parsedValue = strtod(token, &parseEnd);
  if (parseEnd == token || *parseEnd != '\0') {
    return false;
  }

  value = static_cast<float>(parsedValue);
  return true;
}

uint16_t decodeUint16LittleEndian(const uint8_t* data) {
  return static_cast<uint16_t>(data[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(data[1]) << 8);
}

uint32_t decodeUint32LittleEndian(const uint8_t* data) {
  return static_cast<uint32_t>(data[0]) |
         (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) |
         (static_cast<uint32_t>(data[3]) << 24);
}

uint16_t positionNormToCode(float positionNorm) {
  positionNorm = constrain(positionNorm, 0.0f, 1.0f);
  return static_cast<uint16_t>(positionNorm * 65535.0f + 0.5f);
}

uint16_t positionCodeToPulseUs(uint16_t positionCode) {
  uint32_t spanUs = static_cast<uint32_t>(Config::kMaximumPulseUs - Config::kMinimumPulseUs);
  return static_cast<uint16_t>(Config::kMinimumPulseUs +
    ((static_cast<uint32_t>(positionCode) * spanUs + 32767U) / 65535U));
}

uint16_t applySurfacePulseTrimUs(size_t surfaceIndex, uint16_t pulseUs) {
  int32_t trimmedPulseUs = static_cast<int32_t>(pulseUs);
  if (surfaceIndex < Config::kSurfaceCount) {
    trimmedPulseUs += static_cast<int32_t>(Config::kSurfacePulseTrimUs[surfaceIndex]);
  }

  if (trimmedPulseUs < static_cast<int32_t>(Config::kMinimumPulseUs)) {
    trimmedPulseUs = static_cast<int32_t>(Config::kMinimumPulseUs);
  } else if (trimmedPulseUs > static_cast<int32_t>(Config::kMaximumPulseUs)) {
    trimmedPulseUs = static_cast<int32_t>(Config::kMaximumPulseUs);
  }

  return static_cast<uint16_t>(trimmedPulseUs);
}

// =============================================================================
// 5) Pin/Timer Helpers and ISR
// =============================================================================
inline void driveTrainerLow() {
  PortGroup& group = portGroupForPin(Config::kTrainerPin);
  const uint32_t mask = portMaskForPin(Config::kTrainerPin);
  group.OUTCLR.reg = mask;
  group.DIRSET.reg = mask;
}

inline void releaseTrainerOutput() {
  PortGroup& group = portGroupForPin(Config::kTrainerPin);
  const uint32_t mask = portMaskForPin(Config::kTrainerPin);
  group.DIRCLR.reg = mask;
  if (Config::kUseTrainerInputPullupWhenReleased) {
    group.OUTSET.reg = mask;
  } else {
    group.OUTCLR.reg = mask;
  }
}

inline void setPpmIdleLevel() {
  PortGroup& group = portGroupForPin(Config::kTrainerPin);
  const uint32_t mask = portMaskForPin(Config::kTrainerPin);

  if (Config::kUseOpenDrainTrainerOutput) {
    if (Config::kPpmActiveHighPulse) {
      driveTrainerLow();
    } else {
      releaseTrainerOutput();
    }
    return;
  }

  if (Config::kPpmActiveHighPulse) {
    group.OUTCLR.reg = mask;
  } else {
    group.OUTSET.reg = mask;
  }
  group.DIRSET.reg = mask;
}

inline void setPpmMarkLevel() {
  PortGroup& group = portGroupForPin(Config::kTrainerPin);
  const uint32_t mask = portMaskForPin(Config::kTrainerPin);

  if (Config::kUseOpenDrainTrainerOutput) {
    if (Config::kPpmActiveHighPulse) {
      releaseTrainerOutput();
    } else {
      driveTrainerLow();
    }
    return;
  }

  if (Config::kPpmActiveHighPulse) {
    group.OUTSET.reg = mask;
  } else {
    group.OUTCLR.reg = mask;
  }
  group.DIRSET.reg = mask;
}

inline void writeReferenceHigh() {
  portGroupForPin(Config::kReferencePin).OUTSET.reg = portMaskForPin(Config::kReferencePin);
}

inline void writeReferenceLow() {
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

void buildDiagnosticCommand(PendingCommand& candidate, uint8_t stateIndex, uint32_t boardRxUs) {
  buildNeutralCommand(candidate, stateIndex, boardRxUs);
  candidate.activeMask = static_cast<uint8_t>(1U << Config::kDiagnosticSurfaceIndex);
  candidate.ppmUs[Config::kDiagnosticSurfaceIndex] =
    applySurfacePulseTrimUs(
      Config::kDiagnosticSurfaceIndex,
      Config::kDiagnosticPulseSequenceUs[stateIndex]);
}

void serviceInternalDiagnosticPattern() {
  uint32_t nowUs = micros();
  if (static_cast<uint32_t>(nowUs - gDiagnosticLastUpdateUs) < Config::kDiagnosticHoldUs) {
    return;
  }

  gDiagnosticLastUpdateUs = nowUs;
  gDiagnosticStateIndex = static_cast<uint8_t>(
    (gDiagnosticStateIndex + 1U) %
    (sizeof(Config::kDiagnosticPulseSequenceUs) / sizeof(Config::kDiagnosticPulseSequenceUs[0])));

  PendingCommand candidate = {};
  buildDiagnosticCommand(candidate, gDiagnosticStateIndex, nowUs);
  queuePendingVector(candidate);

  writeReferenceHigh();
  delayMicroseconds(Config::kDiagnosticMarkerPulseUs);
  writeReferenceLow();
}

void TC3_Handler() {
  if ((TC3->COUNT16.INTFLAG.reg & TC_INTFLAG_MC0) == 0U) {
    return;
  }
  TC3->COUNT16.INTFLAG.reg = TC_INTFLAG_MC0;

  if (!gPpmPulseActive) {
    if (gPpmIntervalIndex == 0U && gPendingVectorValid) {
      uint32_t boardCommitUs = micros();
      for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
        gActivePpmUs[channelIndex] = gPendingPpmUs[channelIndex];
        gCommitTelemetry.ppmUs[channelIndex] = gPendingPpmUs[channelIndex];
      }
      gCommitTelemetry.sampleSequence = gPendingSampleSequence;
      gCommitTelemetry.activeMask = gPendingActiveMask;
      gCommitTelemetry.boardRxUs = gPendingRxUs;
      gCommitTelemetry.boardCommitUs = boardCommitUs;
      gCommitTelemetry.receiveToCommitUs = boardCommitUs - gPendingRxUs;
      gCommitTelemetry.strobeUs = boardCommitUs;
      gCommitTelemetry.frameIndex = gFrameIndex;
      gCommitTelemetryPending = true;
      gPendingVectorValid = false;
      ++gAppliedUpdateCount;
      writeReferenceHigh();
      gReferencePulseActive = true;
    }

    setPpmMarkLevel();
    scheduleTimerUs(Config::kMarkWidthUs);
    gPpmPulseActive = true;
    return;
  }

  if (gReferencePulseActive) {
    writeReferenceLow();
    gReferencePulseActive = false;
  }

  setPpmIdleLevel();

  if (gPpmIntervalIndex < Config::kPpmChannelCount) {
    uint16_t slotUs = gActivePpmUs[gPpmIntervalIndex];
    uint16_t gapUs = (slotUs > Config::kMarkWidthUs)
      ? static_cast<uint16_t>(slotUs - Config::kMarkWidthUs)
      : 1U;
    ++gPpmIntervalIndex;
    scheduleTimerUs(gapUs);
  } else {
    ++gFrameIndex;
    gPpmIntervalIndex = 0U;
    scheduleTimerUs(computeSyncGapUs());
  }

  gPpmPulseActive = false;
}
