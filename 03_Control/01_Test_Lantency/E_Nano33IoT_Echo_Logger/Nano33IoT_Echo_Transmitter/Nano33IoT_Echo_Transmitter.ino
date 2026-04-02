#include <stdlib.h>
#include <string.h>

namespace Config {
constexpr char kFirmwareVersion[] = "Nano33IoT_Echo_Transmitter_V1";
constexpr uint32_t kSerialBaud = 460800;

constexpr size_t kSurfaceCount = 4;
constexpr size_t kPpmChannelCount = 8;
constexpr size_t kBinaryVectorPacketLength = 15;
constexpr size_t kCommandBufferLength = 192;
constexpr size_t kCommitQueueLength = 32;

constexpr uint8_t kTrainerPin = 3;
constexpr uint8_t kReferencePin = 2;

constexpr uint16_t kMinimumPulseUs = 1000;
constexpr uint16_t kMaximumPulseUs = 2000;
constexpr uint16_t kNeutralPulseUs = 1500;
constexpr uint16_t kFrameLengthUs = 22000;
constexpr uint16_t kMarkWidthUs = 300;
constexpr uint32_t kCommandTimeoutUs = 250000;

constexpr char kSurfaceNames[kSurfaceCount][16] = {
  "Aileron_L",
  "Aileron_R",
  "Rudder",
  "Elevator"
};
}

enum class ReferencePulseMode : uint8_t {
  CommitOnly,
  EveryFrame
};

namespace Config {
constexpr uint16_t kReferencePulseWidthUs = 50;
constexpr bool kReferenceStrobeEnabled = true;
constexpr ReferencePulseMode kReferencePulseMode = ReferencePulseMode::CommitOnly;
}

struct VectorCommand {
  bool isValid;
  uint32_t sampleSequence;
  uint8_t activeMask;
  uint32_t rxUs;
  uint16_t positionCodes[Config::kSurfaceCount];
  uint16_t ppmUs[Config::kPpmChannelCount];
};

struct CommitLogEntry {
  uint32_t sampleSequence;
  uint8_t activeMask;
  uint32_t boardRxUs;
  uint32_t boardCommitUs;
  uint32_t strobeUs;
  uint32_t frameIndex;
  uint16_t ppmUs[Config::kPpmChannelCount];
};

enum class PpmPhase : uint8_t {
  Mark,
  Space,
  SyncGap
};

const __FlashStringHelper* referencePulseModeLabel();

char gCommandBuffer[Config::kCommandBufferLength];
size_t gCommandLength = 0;
bool gCommandOverflow = false;

volatile PortGroup* gTrainerPort = nullptr;
volatile PortGroup* gReferencePort = nullptr;
uint32_t gTrainerMask = 0;
uint32_t gReferenceMask = 0;

VectorCommand gPendingVector = {};
volatile bool gPendingVectorValid = false;
volatile uint16_t gActivePpmUs[Config::kPpmChannelCount] = {
  Config::kNeutralPulseUs, Config::kNeutralPulseUs, Config::kNeutralPulseUs, Config::kNeutralPulseUs,
  Config::kNeutralPulseUs, Config::kNeutralPulseUs, Config::kNeutralPulseUs, Config::kNeutralPulseUs
};

volatile CommitLogEntry gCommitQueue[Config::kCommitQueueLength];
volatile uint8_t gCommitQueueHead = 0;
volatile uint8_t gCommitQueueTail = 0;

volatile PpmPhase gPpmPhase = PpmPhase::SyncGap;
volatile uint8_t gPpmChannelIndex = 0;
volatile uint32_t gFrameIndex = 0;
volatile bool gReferencePulseActive = false;
volatile uint32_t gReferencePulseStartUs = 0;

volatile uint32_t gRxEventCount = 0;
volatile uint32_t gCommitEventCount = 0;
volatile uint32_t gOverwriteCount = 0;
volatile uint32_t gSyncEventCount = 0;
volatile uint32_t gTimeoutNeutralCount = 0;
volatile uint32_t gErrorCount = 0;
volatile uint32_t gLatestSequence = 0;
volatile uint32_t gLastCommandRxUs = 0;
volatile bool gTimeoutNeutralQueued = true;

void setup() {
  Serial.begin(Config::kSerialBaud);
  while (!Serial && millis() < 3000) {
  }

  Serial.setTimeout(1);
  configureOutputPin(Config::kTrainerPin, gTrainerPort, gTrainerMask);
  configureOutputPin(Config::kReferencePin, gReferencePort, gReferenceMask);
  writeTrainerHigh();
  writeReferenceLow();

  VectorCommand neutralCommand = {};
  buildNeutralCommand(neutralCommand, 0, micros());
  loadActiveVector(neutralCommand);
  gLastCommandRxUs = micros();

  configureTimer();
  scheduleTimerUs(computeSyncGapUs());
}

void loop() {
  serviceSerialInput();
  serviceReferencePulse();
  serviceCommandTimeout();
  flushCommitQueue();
}

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

  if (gCommandLength == 0) {
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
    queueNeutralVector(0, micros());
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
  VectorCommand candidate = {};
  if (!tryParseBinaryVectorCommand(packetBytes, boardRxUs, candidate)) {
    return;
  }

  queuePendingVector(candidate);
  sendRxEvent(candidate);
}

bool tryParseBinaryVectorCommand(const uint8_t* packetBytes, uint32_t boardRxUs, VectorCommand& candidate) {
  if (packetBytes[0] != static_cast<uint8_t>('V')) {
    return false;
  }

  if (packetBytes[1] != Config::kSurfaceCount) {
    sendErrorEvent(F("BINARY_VECTOR_SURFACE_COUNT_ERROR"));
    return false;
  }

  candidate = {};
  candidate.isValid = true;
  candidate.sampleSequence = decodeUint32LittleEndian(packetBytes + 3);
  candidate.activeMask = packetBytes[2];
  candidate.rxUs = boardRxUs;

  size_t readIndex = 7;
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    candidate.positionCodes[surfaceIndex] = decodeUint16LittleEndian(packetBytes + readIndex);
    readIndex += 2;
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

  if (surfaceCount == 0 || surfaceCount > Config::kSurfaceCount) {
    sendErrorEvent(F("SET_ALL_SURFACE_COUNT_ERROR"));
    return true;
  }

  VectorCommand candidate = {};
  candidate.isValid = true;
  candidate.sampleSequence = sampleSequence;
  candidate.activeMask = 0;
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

    candidate.activeMask |= static_cast<uint8_t>(1U << static_cast<uint8_t>(surfaceIndex));
    candidate.positionCodes[surfaceIndex] = positionNormToCode(positionNorm);
  }

  finalizeVectorCandidate(candidate);
  queuePendingVector(candidate);
  sendRxEvent(candidate);
  return true;
}

void finalizeVectorCandidate(VectorCommand& candidate) {
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    candidate.ppmUs[surfaceIndex] = positionCodeToPulseUs(candidate.positionCodes[surfaceIndex]);
  }
  for (size_t channelIndex = Config::kSurfaceCount; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    candidate.ppmUs[channelIndex] = Config::kNeutralPulseUs;
  }
}

void buildNeutralCommand(VectorCommand& candidate, uint32_t sampleSequence, uint32_t boardRxUs) {
  candidate = {};
  candidate.isValid = true;
  candidate.sampleSequence = sampleSequence;
  candidate.activeMask = 0;
  candidate.rxUs = boardRxUs;
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    candidate.positionCodes[surfaceIndex] = positionNormToCode(0.5f);
  }
  finalizeVectorCandidate(candidate);
}

void queuePendingVector(const VectorCommand& command) {
  noInterrupts();
  if (gPendingVectorValid) {
    ++gOverwriteCount;
  }
  gPendingVector = command;
  gPendingVectorValid = true;
  gLastCommandRxUs = command.rxUs;
  gLatestSequence = command.sampleSequence;
  gTimeoutNeutralQueued = false;
  interrupts();
}

void queueNeutralVector(uint32_t sampleSequence, uint32_t boardRxUs) {
  VectorCommand neutralCommand = {};
  buildNeutralCommand(neutralCommand, sampleSequence, boardRxUs);
  noInterrupts();
  if (gPendingVectorValid) {
    ++gOverwriteCount;
  }
  gPendingVector = neutralCommand;
  gPendingVectorValid = true;
  gLastCommandRxUs = boardRxUs;
  gTimeoutNeutralQueued = true;
  interrupts();
}

void serviceCommandTimeout() {
  if (gTimeoutNeutralQueued) {
    return;
  }

  const uint32_t nowUs = micros();
  if (static_cast<uint32_t>(nowUs - gLastCommandRxUs) < Config::kCommandTimeoutUs) {
    return;
  }

  ++gTimeoutNeutralCount;
  queueNeutralVector(0, nowUs);
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

  emitReferencePulse();
  uint32_t boardTxUs = micros();
  Serial.print(F("SYNC_EVENT,"));
  Serial.print(syncId);
  Serial.print(',');
  Serial.print(hostTxUs);
  Serial.print(',');
  Serial.print(boardRxUs);
  Serial.print(',');
  Serial.println(boardTxUs);
  ++gSyncEventCount;
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
  Serial.print(F("STATUS_EVENT,rx_event_count="));
  Serial.print(gRxEventCount);
  Serial.print(F(",commit_event_count="));
  Serial.print(gCommitEventCount);
  Serial.print(F(",overwrite_count="));
  Serial.print(gOverwriteCount);
  Serial.print(F(",sync_event_count="));
  Serial.print(gSyncEventCount);
  Serial.print(F(",timeout_neutral_count="));
  Serial.print(gTimeoutNeutralCount);
  Serial.print(F(",error_count="));
  Serial.print(gErrorCount);
  Serial.print(F(",latest_sequence="));
  Serial.print(gLatestSequence);
  Serial.print(F(",reference_mode="));
  Serial.print(referencePulseModeLabel());
  Serial.print(F(",reference_pulse_us="));
  Serial.println(Config::kReferencePulseWidthUs);
}

void sendRxEvent(const VectorCommand& command) {
  Serial.print(F("RX_EVENT,"));
  Serial.print(command.sampleSequence);
  Serial.print(',');
  Serial.print(command.activeMask);
  Serial.print(',');
  Serial.print(command.rxUs);
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    Serial.print(',');
    Serial.print(command.positionCodes[surfaceIndex]);
  }
  Serial.println();
  ++gRxEventCount;
}

void flushCommitQueue() {
  while (gCommitQueueTail != gCommitQueueHead) {
    CommitLogEntry entry = {};
    noInterrupts();
    const volatile CommitLogEntry& queuedEntry = gCommitQueue[gCommitQueueTail];
    entry.sampleSequence = queuedEntry.sampleSequence;
    entry.activeMask = queuedEntry.activeMask;
    entry.boardRxUs = queuedEntry.boardRxUs;
    entry.boardCommitUs = queuedEntry.boardCommitUs;
    entry.strobeUs = queuedEntry.strobeUs;
    entry.frameIndex = queuedEntry.frameIndex;
    for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
      entry.ppmUs[channelIndex] = queuedEntry.ppmUs[channelIndex];
    }
    gCommitQueueTail = static_cast<uint8_t>((gCommitQueueTail + 1U) % Config::kCommitQueueLength);
    interrupts();

    Serial.print(F("COMMIT_EVENT,"));
    Serial.print(entry.sampleSequence);
    Serial.print(',');
    Serial.print(entry.activeMask);
    Serial.print(',');
    Serial.print(entry.boardRxUs);
    Serial.print(',');
    Serial.print(entry.boardCommitUs);
    Serial.print(',');
    Serial.print(entry.strobeUs);
    Serial.print(',');
    Serial.print(entry.frameIndex);
    for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
      Serial.print(',');
      Serial.print(entry.ppmUs[channelIndex]);
    }
    Serial.println();
  }
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
  gRxEventCount = 0;
  gCommitEventCount = 0;
  gOverwriteCount = 0;
  gSyncEventCount = 0;
  gTimeoutNeutralCount = 0;
  gErrorCount = 0;
  gLatestSequence = 0;
  gCommitQueueHead = 0;
  gCommitQueueTail = 0;
  interrupts();
}

void loadActiveVector(const VectorCommand& command) {
  for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    gActivePpmUs[channelIndex] = command.ppmUs[channelIndex];
  }
}

void configureOutputPin(uint8_t pin, volatile PortGroup*& portGroup, uint32_t& bitMask) {
  pinMode(pin, OUTPUT);
  const PinDescription& pinDescription = g_APinDescription[pin];
  portGroup = &(PORT->Group[pinDescription.ulPort]);
  bitMask = (1UL << pinDescription.ulPin);
}

inline void writeTrainerHigh() {
  gTrainerPort->OUTSET.reg = gTrainerMask;
}

inline void writeTrainerLow() {
  gTrainerPort->OUTCLR.reg = gTrainerMask;
}

inline void writeReferenceHigh() {
  gReferencePort->OUTSET.reg = gReferenceMask;
}

inline void writeReferenceLow() {
  gReferencePort->OUTCLR.reg = gReferenceMask;
}

void emitReferencePulse() {
  if (!Config::kReferenceStrobeEnabled) {
    return;
  }

  noInterrupts();
  writeReferenceHigh();
  gReferencePulseStartUs = micros();
  gReferencePulseActive = true;
  interrupts();
}

uint32_t emitReferencePulseIsr() {
  if (!Config::kReferenceStrobeEnabled) {
    return 0;
  }

  writeReferenceHigh();
  uint32_t nowUs = micros();
  gReferencePulseStartUs = nowUs;
  gReferencePulseActive = true;
  return nowUs;
}

const __FlashStringHelper* referencePulseModeLabel() {
  switch (Config::kReferencePulseMode) {
    case ReferencePulseMode::CommitOnly:
      return F("commit_only");

    case ReferencePulseMode::EveryFrame:
      return F("every_frame");
  }

  return F("commit_only");
}

void serviceReferencePulse() {
  if (!gReferencePulseActive) {
    return;
  }

  const uint32_t nowUs = micros();
  if (static_cast<uint32_t>(nowUs - gReferencePulseStartUs) < Config::kReferencePulseWidthUs) {
    return;
  }

  noInterrupts();
  if (gReferencePulseActive &&
      static_cast<uint32_t>(micros() - gReferencePulseStartUs) >= Config::kReferencePulseWidthUs) {
    writeReferenceLow();
    gReferencePulseActive = false;
  }
  interrupts();
}

void configureTimer() {
  PM->APBCMASK.reg |= PM_APBCMASK_TC5;
  GCLK->CLKCTRL.reg = GCLK_CLKCTRL_CLKEN | GCLK_CLKCTRL_GEN_GCLK0 | GCLK_CLKCTRL_ID_TC4_TC5;
  while (GCLK->STATUS.bit.SYNCBUSY) {
  }

  TC5->COUNT16.CTRLA.reg = TC_CTRLA_SWRST;
  while (TC5->COUNT16.STATUS.bit.SYNCBUSY || TC5->COUNT16.CTRLA.bit.SWRST) {
  }

  TC5->COUNT16.CTRLA.reg =
    TC_CTRLA_MODE_COUNT16 |
    TC_CTRLA_WAVEGEN_MFRQ |
    TC_CTRLA_PRESCALER_DIV16;
  while (TC5->COUNT16.STATUS.bit.SYNCBUSY) {
  }

  TC5->COUNT16.CC[0].reg = 3000;
  while (TC5->COUNT16.STATUS.bit.SYNCBUSY) {
  }

  NVIC_SetPriority(TC5_IRQn, 0);
  NVIC_EnableIRQ(TC5_IRQn);
  TC5->COUNT16.INTENSET.reg = TC_INTENSET_MC0;
  TC5->COUNT16.CTRLA.bit.ENABLE = 1;
  while (TC5->COUNT16.STATUS.bit.SYNCBUSY) {
  }
}

void scheduleTimerUs(uint32_t intervalUs) {
  uint32_t ticks = intervalUs * 3U;
  if (ticks < 1U) {
    ticks = 1U;
  } else if (ticks > 65535U) {
    ticks = 65535U;
  }
  TC5->COUNT16.COUNT.reg = 0;
  while (TC5->COUNT16.STATUS.bit.SYNCBUSY) {
  }
  TC5->COUNT16.CC[0].reg = static_cast<uint16_t>(ticks);
  while (TC5->COUNT16.STATUS.bit.SYNCBUSY) {
  }
}

uint16_t computeSyncGapUs() {
  uint32_t totalSlotUs = 0;
  for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    totalSlotUs += gActivePpmUs[channelIndex];
  }

  if (Config::kFrameLengthUs <= totalSlotUs) {
    return Config::kMarkWidthUs;
  }
  return static_cast<uint16_t>(Config::kFrameLengthUs - totalSlotUs);
}

void enqueueCommitEventIsr(const VectorCommand& command, uint32_t boardCommitUs, uint32_t strobeUs) {
  const uint8_t nextHead = static_cast<uint8_t>((gCommitQueueHead + 1U) % Config::kCommitQueueLength);
  if (nextHead == gCommitQueueTail) {
    ++gErrorCount;
    return;
  }

  volatile CommitLogEntry& entry = gCommitQueue[gCommitQueueHead];
  entry.sampleSequence = command.sampleSequence;
  entry.activeMask = command.activeMask;
  entry.boardRxUs = command.rxUs;
  entry.boardCommitUs = boardCommitUs;
  entry.strobeUs = strobeUs;
  entry.frameIndex = gFrameIndex;
  for (size_t channelIndex = 0; channelIndex < Config::kPpmChannelCount; ++channelIndex) {
    entry.ppmUs[channelIndex] = command.ppmUs[channelIndex];
  }

  gCommitQueueHead = nextHead;
  ++gCommitEventCount;
}

void startNextFrameIsr() {
  ++gFrameIndex;
  uint32_t frameStrobeUs = 0;

  if (Config::kReferencePulseMode == ReferencePulseMode::EveryFrame) {
    frameStrobeUs = emitReferencePulseIsr();
  }

  if (gPendingVectorValid) {
    VectorCommand committed = gPendingVector;
    gPendingVectorValid = false;
    loadActiveVector(committed);
    uint32_t strobeUs = frameStrobeUs;
    if (Config::kReferencePulseMode == ReferencePulseMode::CommitOnly) {
      strobeUs = emitReferencePulseIsr();
    }
    const uint32_t boardCommitUs = micros();
    enqueueCommitEventIsr(committed, boardCommitUs, strobeUs);
  }

  gPpmChannelIndex = 0;
  writeTrainerLow();
  scheduleTimerUs(Config::kMarkWidthUs);
  gPpmPhase = PpmPhase::Mark;
}

void advancePpmStateIsr() {
  switch (gPpmPhase) {
    case PpmPhase::Mark: {
      writeTrainerHigh();
      const uint16_t slotUs = gActivePpmUs[gPpmChannelIndex];
      const uint16_t spaceUs = (slotUs > Config::kMarkWidthUs) ? static_cast<uint16_t>(slotUs - Config::kMarkWidthUs) : 1;
      scheduleTimerUs(spaceUs);
      gPpmPhase = PpmPhase::Space;
      break;
    }

    case PpmPhase::Space:
      ++gPpmChannelIndex;
      if (gPpmChannelIndex >= Config::kPpmChannelCount) {
        scheduleTimerUs(computeSyncGapUs());
        gPpmPhase = PpmPhase::SyncGap;
      } else {
        writeTrainerLow();
        scheduleTimerUs(Config::kMarkWidthUs);
        gPpmPhase = PpmPhase::Mark;
      }
      break;

    case PpmPhase::SyncGap:
      startNextFrameIsr();
      break;
  }
}

void TC5_Handler() {
  if (TC5->COUNT16.INTFLAG.bit.MC0 == 0) {
    return;
  }

  TC5->COUNT16.INTFLAG.reg = TC_INTFLAG_MC0;
  advancePpmStateIsr();
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
  float parsedValue = strtof(token, &parseEnd);
  if (parseEnd == token || *parseEnd != '\0') {
    return false;
  }

  value = parsedValue;
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
  const uint32_t spanUs = static_cast<uint32_t>(Config::kMaximumPulseUs - Config::kMinimumPulseUs);
  return static_cast<uint16_t>(
    Config::kMinimumPulseUs +
    ((static_cast<uint32_t>(positionCode) * spanUs + 32767U) / 65535U));
}
