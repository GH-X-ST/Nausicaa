#include <Servo.h>
#include <stdlib.h>
#include <string.h>

namespace Config {
constexpr char kFirmwareVersion[] = "Nano33IoT_Echo_Logger_Wire_V1";
constexpr uint32_t kSerialBaud = 115200;

constexpr size_t kSurfaceCount = 4;
constexpr char kSurfaceNames[kSurfaceCount][16] = {
  "Aileron_L",
  "Aileron_R",
  "Rudder",
  "Elevator"
};
constexpr uint8_t kServoPins[kSurfaceCount] = {9, 10, 11, 12};
constexpr uint16_t kMinPulseUs[kSurfaceCount] = {1000, 1000, 1000, 1000};
constexpr uint16_t kMaxPulseUs[kSurfaceCount] = {2000, 2000, 2000, 2000};
constexpr float kNeutralPositions[kSurfaceCount] = {0.5f, 0.5f, 0.5f, 0.5f};
constexpr size_t kCommandBufferLength = 96;
}

Servo gServos[Config::kSurfaceCount];

char gCommandBuffer[Config::kCommandBufferLength];
size_t gCommandLength = 0;
bool gCommandOverflow = false;

uint32_t gCommandEventCount = 0;
uint32_t gSyncEventCount = 0;
uint32_t gErrorCount = 0;

void setup() {
  Serial.begin(Config::kSerialBaud);
  while (!Serial && millis() < 3000) {
  }

  Serial.setTimeout(1);

  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    gServos[surfaceIndex].attach(
      Config::kServoPins[surfaceIndex],
      Config::kMinPulseUs[surfaceIndex],
      Config::kMaxPulseUs[surfaceIndex]);
    applySurfacePosition(surfaceIndex, Config::kNeutralPositions[surfaceIndex]);
  }

  Serial.println(F("Nano33IoT wire echo logger ready."));
}

void loop() {
  while (Serial.available() > 0) {
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
  handleCommand(gCommandBuffer, micros());
  resetCommandBuffer();
}

void resetCommandBuffer() {
  gCommandLength = 0;
  gCommandOverflow = false;
}

void handleCommand(char* commandBuffer, uint32_t boardRxUs) {
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
    applyNeutralToAllSurfaces();
    sendOkEvent(F("SET_NEUTRAL"));
    return;
  }

  if (strcmp(commandName, "SYNC") == 0) {
    handleSyncCommand(context, boardRxUs);
    return;
  }

  if (strcmp(commandName, "SET") == 0) {
    handleSetCommand(context, boardRxUs);
    return;
  }

  sendErrorEvent(F("UNKNOWN_COMMAND"));
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

void handleSetCommand(char* context, uint32_t boardRxUs) {
  char* surfaceNameToken = strtok_r(nullptr, ",", &context);
  char* sequenceToken = strtok_r(nullptr, ",", &context);
  char* positionToken = strtok_r(nullptr, ",", &context);

  int surfaceIndex = findSurfaceIndex(surfaceNameToken);
  uint32_t commandSequence = 0;
  float positionNorm = 0.0f;
  if (surfaceIndex < 0 || !parseUnsigned32(sequenceToken, commandSequence) || !parseFloat(positionToken, positionNorm)) {
    sendErrorEvent(F("SET_PARSE_ERROR"));
    return;
  }

  positionNorm = constrain(positionNorm, 0.0f, 1.0f);
  uint16_t pulseUs = applySurfacePosition(static_cast<size_t>(surfaceIndex), positionNorm);
  uint32_t applyUs = micros();

  Serial.print(F("COMMAND_EVENT,"));
  Serial.print(Config::kSurfaceNames[surfaceIndex]);
  Serial.print(',');
  Serial.print(commandSequence);
  Serial.print(',');
  Serial.print(boardRxUs);
  Serial.print(',');
  Serial.print(applyUs);
  Serial.print(',');
  Serial.print(positionNorm, 6);
  Serial.print(',');
  Serial.println(pulseUs);
  ++gCommandEventCount;
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
  Serial.print(F("STATUS_EVENT,command_event_count="));
  Serial.print(gCommandEventCount);
  Serial.print(F(",sync_event_count="));
  Serial.print(gSyncEventCount);
  Serial.print(F(",error_count="));
  Serial.println(gErrorCount);
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
  gCommandEventCount = 0;
  gSyncEventCount = 0;
  gErrorCount = 0;
}

void applyNeutralToAllSurfaces() {
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    applySurfacePosition(surfaceIndex, Config::kNeutralPositions[surfaceIndex]);
  }
}

uint16_t applySurfacePosition(size_t surfaceIndex, float positionNorm) {
  uint16_t pulseUs = positionToPulseUs(surfaceIndex, positionNorm);
  gServos[surfaceIndex].writeMicroseconds(pulseUs);
  return pulseUs;
}

uint16_t positionToPulseUs(size_t surfaceIndex, float positionNorm) {
  float clippedPosition = constrain(positionNorm, 0.0f, 1.0f);
  float pulseUs = static_cast<float>(Config::kMinPulseUs[surfaceIndex]) +
    clippedPosition * static_cast<float>(Config::kMaxPulseUs[surfaceIndex] - Config::kMinPulseUs[surfaceIndex]);
  return static_cast<uint16_t>(pulseUs + 0.5f);
}

int findSurfaceIndex(const char* surfaceNameToken) {
  if (surfaceNameToken == nullptr) {
    return -1;
  }

  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    if (strcmp(surfaceNameToken, Config::kSurfaceNames[surfaceIndex]) == 0) {
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
  if (token == parseEnd || *parseEnd != '\0') {
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
  if (token == parseEnd || *parseEnd != '\0') {
    return false;
  }

  value = parsedValue;
  return true;
}
