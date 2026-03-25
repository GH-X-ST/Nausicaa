#include <SPI.h>
#include <Servo.h>
#include <WiFiNINA.h>
#include <WiFiUdp.h>
#include <stdlib.h>
#include <string.h>

namespace Config {
constexpr char kFirmwareVersion[] = "Nano33IoT_Echo_Logger_V4_UDP";
constexpr char kWifiSsid[] = "FlightArena_2.4G";
constexpr char kWifiPassword[] = "R0b0t1c$";
constexpr bool kUseStaticIp = true;
constexpr uint8_t kStaticIp[4] = {192, 168, 0, 33};
constexpr uint8_t kStaticGateway[4] = {192, 168, 0, 1};
constexpr uint8_t kStaticSubnet[4] = {255, 255, 255, 0};
constexpr uint8_t kStaticDns[4] = {192, 168, 0, 1};
constexpr uint16_t kServerPort = 9500;
constexpr uint32_t kSerialBaud = 115200;
constexpr uint32_t kWifiRetryIntervalMs = 2000;

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

constexpr size_t kPacketBufferLength = 192;
constexpr size_t kBinaryVectorPacketLength = 7 + 2 * kSurfaceCount;
}

enum class TelemetryMode : uint8_t {
  Controller = 0,
  Instrumentation = 1
};

struct PendingVectorCommand {
  bool isValid;
  uint32_t sampleSequence;
  uint32_t rxUs;
  uint8_t activeMask;
  uint16_t positionCodes[Config::kSurfaceCount];
  IPAddress remoteIp;
  uint16_t remotePort;
};

WiFiUDP gUdp;
Servo gServos[Config::kSurfaceCount];

char gPacketBuffer[Config::kPacketBufferLength];
uint32_t gLastWifiAttemptMs = 0;
TelemetryMode gTelemetryMode = TelemetryMode::Controller;

uint32_t gVectorEventCount = 0;
uint32_t gCommandEventCount = 0;
uint32_t gSyncEventCount = 0;
uint32_t gErrorCount = 0;

void setup() {
  Serial.begin(Config::kSerialBaud);
  while (!Serial && millis() < 3000) {
  }

  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    gServos[surfaceIndex].attach(
      Config::kServoPins[surfaceIndex],
      Config::kMinPulseUs[surfaceIndex],
      Config::kMaxPulseUs[surfaceIndex]);
    applySurfacePosition(surfaceIndex, positionNormToCode(Config::kNeutralPositions[surfaceIndex]));
  }

  connectToWifi(true);
  gUdp.begin(Config::kServerPort);

  Serial.println(F("Nano33IoT UDP echo logger ready."));
}

void loop() {
  maintainWifiConnection();
  serviceUdpDatagrams();
}

void maintainWifiConnection() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  uint32_t nowMs = millis();
  if (nowMs - gLastWifiAttemptMs < Config::kWifiRetryIntervalMs) {
    return;
  }

  connectToWifi(false);
}

void connectToWifi(bool blockUntilConnected) {
  gLastWifiAttemptMs = millis();

  if (strlen(Config::kWifiSsid) == 0 || strcmp(Config::kWifiSsid, "YOUR_WIFI_SSID") == 0) {
    Serial.println(F("Set Config::kWifiSsid and Config::kWifiPassword before uploading."));
    return;
  }

  while (true) {
    WiFi.disconnect();
    if (Config::kUseStaticIp) {
      IPAddress localIp(
        Config::kStaticIp[0],
        Config::kStaticIp[1],
        Config::kStaticIp[2],
        Config::kStaticIp[3]);
      IPAddress gateway(
        Config::kStaticGateway[0],
        Config::kStaticGateway[1],
        Config::kStaticGateway[2],
        Config::kStaticGateway[3]);
      IPAddress subnet(
        Config::kStaticSubnet[0],
        Config::kStaticSubnet[1],
        Config::kStaticSubnet[2],
        Config::kStaticSubnet[3]);
      IPAddress dns(
        Config::kStaticDns[0],
        Config::kStaticDns[1],
        Config::kStaticDns[2],
        Config::kStaticDns[3]);

      Serial.print(F("Requesting static IP: "));
      Serial.println(localIp);
      WiFi.config(localIp, dns, gateway, subnet);
    }

    WiFi.begin(Config::kWifiSsid, Config::kWifiPassword);

    uint32_t waitStartMs = millis();
    while (millis() - waitStartMs < 10000) {
      if (WiFi.status() == WL_CONNECTED) {
        Serial.print(F("Connected to WiFi, IP: "));
        Serial.println(WiFi.localIP());
        return;
      }

      delay(100);
    }

    if (!blockUntilConnected) {
      return;
    }
  }
}

void serviceUdpDatagrams() {
  PendingVectorCommand latestVectorCommand = {};
  latestVectorCommand.isValid = false;

  int packetSize = gUdp.parsePacket();
  while (packetSize > 0) {
    IPAddress remoteIp = gUdp.remoteIP();
    uint16_t remotePort = static_cast<uint16_t>(gUdp.remotePort());

    int bytesRead = gUdp.read(reinterpret_cast<uint8_t*>(gPacketBuffer), Config::kPacketBufferLength - 1);
    if (bytesRead > 0) {
      uint32_t boardRxUs = micros();
      bool handled = tryParseBinaryVectorCommand(
        reinterpret_cast<const uint8_t*>(gPacketBuffer),
        static_cast<size_t>(bytesRead),
        boardRxUs,
        remoteIp,
        remotePort,
        latestVectorCommand);

      if (!handled) {
        gPacketBuffer[bytesRead] = '\0';
        handled = tryParseTextVectorCommand(
          gPacketBuffer,
          boardRxUs,
          remoteIp,
          remotePort,
          latestVectorCommand);
      }

      if (!handled) {
        handleTextCommand(gPacketBuffer, boardRxUs, remoteIp, remotePort);
      }
    }

    packetSize = gUdp.parsePacket();
  }

  if (latestVectorCommand.isValid) {
    applyLatestVectorCommand(latestVectorCommand);
  }
}

bool tryParseBinaryVectorCommand(
  const uint8_t* packetBytes,
  size_t packetLength,
  uint32_t boardRxUs,
  const IPAddress& remoteIp,
  uint16_t remotePort,
  PendingVectorCommand& pendingCommand) {
  if (packetLength == 0 || packetBytes[0] != static_cast<uint8_t>('V')) {
    return false;
  }

  if (packetLength != Config::kBinaryVectorPacketLength) {
    sendErrorEvent(remoteIp, remotePort, F("BINARY_VECTOR_LENGTH_ERROR"));
    return true;
  }

  const uint8_t surfaceCount = packetBytes[1];
  if (surfaceCount != Config::kSurfaceCount) {
    sendErrorEvent(remoteIp, remotePort, F("BINARY_VECTOR_SURFACE_COUNT_ERROR"));
    return true;
  }

  PendingVectorCommand candidate = {};
  candidate.isValid = true;
  candidate.sampleSequence = decodeUint32LittleEndian(packetBytes + 3);
  candidate.rxUs = boardRxUs;
  candidate.activeMask = packetBytes[2];
  candidate.remoteIp = remoteIp;
  candidate.remotePort = remotePort;

  size_t readIndex = 7;
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    candidate.positionCodes[surfaceIndex] = decodeUint16LittleEndian(packetBytes + readIndex);
    readIndex += 2;
  }

  pendingCommand = candidate;
  return true;
}

bool tryParseTextVectorCommand(
  char* packetBuffer,
  uint32_t boardRxUs,
  const IPAddress& remoteIp,
  uint16_t remotePort,
  PendingVectorCommand& pendingCommand) {
  if (strncmp(packetBuffer, "SET_ALL", 7) != 0 ||
      (packetBuffer[7] != ',' && packetBuffer[7] != '\0')) {
    return false;
  }

  char parseBuffer[Config::kPacketBufferLength];
  strncpy(parseBuffer, packetBuffer, Config::kPacketBufferLength - 1);
  parseBuffer[Config::kPacketBufferLength - 1] = '\0';

  char* context = nullptr;
  char* commandName = strtok_r(parseBuffer, ",", &context);
  if (commandName == nullptr) {
    return false;
  }

  char* sampleSequenceToken = strtok_r(nullptr, ",", &context);
  char* surfaceCountToken = strtok_r(nullptr, ",", &context);

  uint32_t sampleSequence = 0;
  uint32_t surfaceCount = 0;
  if (!parseUnsigned32(sampleSequenceToken, sampleSequence) ||
      !parseUnsigned32(surfaceCountToken, surfaceCount)) {
    sendErrorEvent(remoteIp, remotePort, F("SET_ALL_PARSE_ERROR"));
    return true;
  }

  if (surfaceCount == 0 || surfaceCount > Config::kSurfaceCount) {
    sendErrorEvent(remoteIp, remotePort, F("SET_ALL_SURFACE_COUNT_ERROR"));
    return true;
  }

  PendingVectorCommand candidate = {};
  candidate.isValid = true;
  candidate.sampleSequence = sampleSequence;
  candidate.rxUs = boardRxUs;
  candidate.activeMask = 0;
  candidate.remoteIp = remoteIp;
  candidate.remotePort = remotePort;

  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    candidate.positionCodes[surfaceIndex] = 0;
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
      sendErrorEvent(remoteIp, remotePort, F("SET_ALL_ITEM_PARSE_ERROR"));
      return true;
    }

    candidate.activeMask |= static_cast<uint8_t>(1U << static_cast<uint8_t>(surfaceIndex));
    candidate.positionCodes[surfaceIndex] = positionNormToCode(positionNorm);
  }

  pendingCommand = candidate;
  return true;
}

void handleTextCommand(char* packetBuffer, uint32_t boardRxUs, const IPAddress& remoteIp, uint16_t remotePort) {
  char* context = nullptr;
  char* commandName = strtok_r(packetBuffer, ",", &context);
  if (commandName == nullptr) {
    return;
  }

  if (strcmp(commandName, "HELLO") == 0) {
    sendHelloEvent(remoteIp, remotePort);
    return;
  }

  if (strcmp(commandName, "STATUS") == 0) {
    sendStatusEvent(remoteIp, remotePort);
    return;
  }

  if (strcmp(commandName, "MODE") == 0) {
    handleModeCommand(context, remoteIp, remotePort);
    return;
  }

  if (strcmp(commandName, "CLEAR_LOGS") == 0) {
    clearCounters();
    sendOkEvent(remoteIp, remotePort, F("CLEAR_LOGS"));
    return;
  }

  if (strcmp(commandName, "SET_NEUTRAL") == 0) {
    applyNeutralToAllSurfaces();
    sendOkEvent(remoteIp, remotePort, F("SET_NEUTRAL"));
    return;
  }

  if (strcmp(commandName, "SYNC") == 0) {
    handleSyncCommand(context, boardRxUs, remoteIp, remotePort);
    return;
  }

  if (strcmp(commandName, "SET") == 0) {
    handleSetCommand(context, boardRxUs, remoteIp, remotePort);
    return;
  }

  sendErrorEvent(remoteIp, remotePort, F("UNKNOWN_COMMAND"));
}

void handleModeCommand(char* context, const IPAddress& remoteIp, uint16_t remotePort) {
  char* modeToken = strtok_r(nullptr, ",", &context);
  if (modeToken == nullptr) {
    sendErrorEvent(remoteIp, remotePort, F("MODE_PARSE_ERROR"));
    return;
  }

  if (strcmp(modeToken, "CONTROLLER") == 0) {
    gTelemetryMode = TelemetryMode::Controller;
    sendOkEvent(remoteIp, remotePort, F("MODE_CONTROLLER"));
    return;
  }

  if (strcmp(modeToken, "INSTRUMENTATION") == 0) {
    gTelemetryMode = TelemetryMode::Instrumentation;
    sendOkEvent(remoteIp, remotePort, F("MODE_INSTRUMENTATION"));
    return;
  }

  sendErrorEvent(remoteIp, remotePort, F("MODE_VALUE_ERROR"));
}

void handleSyncCommand(char* context, uint32_t boardRxUs, const IPAddress& remoteIp, uint16_t remotePort) {
  char* syncIdToken = strtok_r(nullptr, ",", &context);
  char* hostTxUsToken = strtok_r(nullptr, ",", &context);

  uint32_t syncId = 0;
  uint32_t hostTxUs = 0;
  if (!parseUnsigned32(syncIdToken, syncId) || !parseUnsigned32(hostTxUsToken, hostTxUs)) {
    sendErrorEvent(remoteIp, remotePort, F("SYNC_PARSE_ERROR"));
    return;
  }

  uint32_t boardTxUs = micros();
  ++gSyncEventCount;

  if (!beginEventPacket(remoteIp, remotePort)) {
    return;
  }

  gUdp.print(F("SYNC_EVENT,"));
  gUdp.print(syncId);
  gUdp.print(',');
  gUdp.print(hostTxUs);
  gUdp.print(',');
  gUdp.print(boardRxUs);
  gUdp.print(',');
  gUdp.print(boardTxUs);
  gUdp.endPacket();
}

void handleSetCommand(char* context, uint32_t boardRxUs, const IPAddress& remoteIp, uint16_t remotePort) {
  char* surfaceNameToken = strtok_r(nullptr, ",", &context);
  char* sequenceToken = strtok_r(nullptr, ",", &context);
  char* positionToken = strtok_r(nullptr, ",", &context);

  int surfaceIndex = findSurfaceIndex(surfaceNameToken);
  uint32_t commandSequence = 0;
  float positionNorm = 0.0f;
  if (surfaceIndex < 0 || !parseUnsigned32(sequenceToken, commandSequence) || !parseFloat(positionToken, positionNorm)) {
    sendErrorEvent(remoteIp, remotePort, F("SET_PARSE_ERROR"));
    return;
  }

  const uint16_t positionCode = positionNormToCode(positionNorm);
  const uint16_t pulseUs = applySurfacePosition(static_cast<size_t>(surfaceIndex), positionCode);
  const uint32_t applyUs = micros();

  ++gCommandEventCount;

  if (!beginEventPacket(remoteIp, remotePort)) {
    return;
  }

  gUdp.print(F("COMMAND_EVENT,"));
  gUdp.print(Config::kSurfaceNames[surfaceIndex]);
  gUdp.print(',');
  gUdp.print(commandSequence);
  gUdp.print(',');
  gUdp.print(boardRxUs);
  gUdp.print(',');
  gUdp.print(applyUs);
  gUdp.print(',');
  gUdp.print(positionCodeToNorm(positionCode), 6);
  gUdp.print(',');
  gUdp.print(pulseUs);
  gUdp.endPacket();
}

void applyLatestVectorCommand(const PendingVectorCommand& command) {
  uint32_t applyUs[Config::kSurfaceCount] = {0, 0, 0, 0};
  uint16_t pulseUs[Config::kSurfaceCount] = {0, 0, 0, 0};

  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    if ((command.activeMask & static_cast<uint8_t>(1U << surfaceIndex)) == 0U) {
      continue;
    }

    pulseUs[surfaceIndex] = applySurfacePosition(surfaceIndex, command.positionCodes[surfaceIndex]);
    applyUs[surfaceIndex] = micros();
  }

  if (gTelemetryMode == TelemetryMode::Instrumentation) {
    sendCommandEvents(command, applyUs, pulseUs);
  } else {
    sendVectorEvent(command, applyUs, pulseUs);
  }
}

void sendCommandEvents(
  const PendingVectorCommand& command,
  const uint32_t* applyUs,
  const uint16_t* pulseUs) {
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    if ((command.activeMask & static_cast<uint8_t>(1U << surfaceIndex)) == 0U) {
      continue;
    }

    if (!beginEventPacket(command.remoteIp, command.remotePort)) {
      continue;
    }

    gUdp.print(F("COMMAND_EVENT,"));
    gUdp.print(Config::kSurfaceNames[surfaceIndex]);
    gUdp.print(',');
    gUdp.print(command.sampleSequence);
    gUdp.print(',');
    gUdp.print(command.rxUs);
    gUdp.print(',');
    gUdp.print(applyUs[surfaceIndex]);
    gUdp.print(',');
    gUdp.print(positionCodeToNorm(command.positionCodes[surfaceIndex]), 6);
    gUdp.print(',');
    gUdp.print(pulseUs[surfaceIndex]);
    gUdp.endPacket();
    ++gCommandEventCount;
  }
}

void sendVectorEvent(
  const PendingVectorCommand& command,
  const uint32_t* applyUs,
  const uint16_t* pulseUs) {
  if (!beginEventPacket(command.remoteIp, command.remotePort)) {
    return;
  }

  gUdp.print(F("VECTOR_EVENT,"));
  gUdp.print(command.sampleSequence);
  gUdp.print(',');
  gUdp.print(command.activeMask);
  gUdp.print(',');
  gUdp.print(command.rxUs);

  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    gUdp.print(',');
    gUdp.print(applyUs[surfaceIndex]);
    gUdp.print(',');
    gUdp.print(command.positionCodes[surfaceIndex]);
    gUdp.print(',');
    gUdp.print(pulseUs[surfaceIndex]);
  }

  gUdp.endPacket();
  ++gVectorEventCount;
}

bool beginEventPacket(const IPAddress& remoteIp, uint16_t remotePort) {
  if (remotePort == 0) {
    return false;
  }

  return gUdp.beginPacket(remoteIp, remotePort) == 1;
}

void sendHelloEvent(const IPAddress& remoteIp, uint16_t remotePort) {
  if (!beginEventPacket(remoteIp, remotePort)) {
    return;
  }

  gUdp.print(F("HELLO_EVENT,"));
  gUdp.print(Config::kFirmwareVersion);
  gUdp.print(',');
  gUdp.print(WiFi.localIP());
  gUdp.print(',');
  gUdp.print(Config::kServerPort);
  gUdp.print(',');
  gUdp.print(telemetryModeToText(gTelemetryMode));
  gUdp.print(',');
  gUdp.print(micros());
  gUdp.endPacket();
}

void sendStatusEvent(const IPAddress& remoteIp, uint16_t remotePort) {
  if (!beginEventPacket(remoteIp, remotePort)) {
    return;
  }

  gUdp.print(F("STATUS_EVENT,"));
  gUdp.print(F("telemetry_mode="));
  gUdp.print(telemetryModeToText(gTelemetryMode));
  gUdp.print(F(",vector_event_count="));
  gUdp.print(gVectorEventCount);
  gUdp.print(F(",command_event_count="));
  gUdp.print(gCommandEventCount);
  gUdp.print(F(",sync_event_count="));
  gUdp.print(gSyncEventCount);
  gUdp.print(F(",error_count="));
  gUdp.print(gErrorCount);
  gUdp.print(F(",wifi_status="));
  gUdp.print(WiFi.status());
  gUdp.endPacket();
}

void sendOkEvent(const IPAddress& remoteIp, uint16_t remotePort, const __FlashStringHelper* message) {
  if (!beginEventPacket(remoteIp, remotePort)) {
    return;
  }

  gUdp.print(F("OK_EVENT,"));
  gUdp.print(message);
  gUdp.endPacket();
}

void sendErrorEvent(const IPAddress& remoteIp, uint16_t remotePort, const __FlashStringHelper* message) {
  ++gErrorCount;

  if (!beginEventPacket(remoteIp, remotePort)) {
    return;
  }

  gUdp.print(F("ERR_EVENT,"));
  gUdp.print(message);
  gUdp.endPacket();
}

void clearCounters() {
  gVectorEventCount = 0;
  gCommandEventCount = 0;
  gSyncEventCount = 0;
  gErrorCount = 0;
}

void applyNeutralToAllSurfaces() {
  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    applySurfacePosition(surfaceIndex, positionNormToCode(Config::kNeutralPositions[surfaceIndex]));
  }
}

uint16_t applySurfacePosition(size_t surfaceIndex, uint16_t positionCode) {
  uint16_t pulseUs = positionCodeToPulseUs(surfaceIndex, positionCode);
  gServos[surfaceIndex].writeMicroseconds(pulseUs);
  return pulseUs;
}

uint16_t positionCodeToPulseUs(size_t surfaceIndex, uint16_t positionCode) {
  const uint32_t pulseRangeUs = static_cast<uint32_t>(Config::kMaxPulseUs[surfaceIndex] - Config::kMinPulseUs[surfaceIndex]);
  const uint32_t scaledPulseUs = static_cast<uint32_t>(positionCode) * pulseRangeUs + 32767U;
  return static_cast<uint16_t>(Config::kMinPulseUs[surfaceIndex] + scaledPulseUs / 65535U);
}

float positionCodeToNorm(uint16_t positionCode) {
  return static_cast<float>(positionCode) / 65535.0f;
}

uint16_t positionNormToCode(float positionNorm) {
  const float clippedPosition = constrain(positionNorm, 0.0f, 1.0f);
  return static_cast<uint16_t>(clippedPosition * 65535.0f + 0.5f);
}

const char* telemetryModeToText(TelemetryMode telemetryMode) {
  if (telemetryMode == TelemetryMode::Instrumentation) {
    return "INSTRUMENTATION";
  }

  return "CONTROLLER";
}

uint16_t decodeUint16LittleEndian(const uint8_t* valueBytes) {
  return static_cast<uint16_t>(valueBytes[0]) |
    static_cast<uint16_t>(static_cast<uint16_t>(valueBytes[1]) << 8);
}

uint32_t decodeUint32LittleEndian(const uint8_t* valueBytes) {
  return static_cast<uint32_t>(valueBytes[0]) |
    static_cast<uint32_t>(static_cast<uint32_t>(valueBytes[1]) << 8) |
    static_cast<uint32_t>(static_cast<uint32_t>(valueBytes[2]) << 16) |
    static_cast<uint32_t>(static_cast<uint32_t>(valueBytes[3]) << 24);
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
