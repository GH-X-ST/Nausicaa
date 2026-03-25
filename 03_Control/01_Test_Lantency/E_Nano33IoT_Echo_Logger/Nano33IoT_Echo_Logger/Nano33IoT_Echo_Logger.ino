#include <SPI.h>
#include <Servo.h>
#include <WiFiNINA.h>
#include <WiFiUdp.h>

namespace Config {
constexpr char kFirmwareVersion[] = "Nano33IoT_Echo_Logger_V3_UDP";
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

constexpr size_t kCommandLogCapacity = 1024;
constexpr size_t kSyncLogCapacity = 128;
constexpr size_t kPacketBufferLength = 160;
}

struct CommandLogEntry {
  uint32_t commandSequence;
  uint32_t rxUs;
  uint32_t applyUs;
  float appliedPosition;
  uint16_t pulseUs;
  uint8_t surfaceIndex;
};

struct SyncLogEntry {
  uint32_t syncId;
  uint32_t hostTxUs;
  uint32_t boardRxUs;
  uint32_t boardTxUs;
};

WiFiUDP gUdp;
Servo gServos[Config::kSurfaceCount];

CommandLogEntry gCommandLog[Config::kCommandLogCapacity];
SyncLogEntry gSyncLog[Config::kSyncLogCapacity];

size_t gCommandLogCount = 0;
size_t gCommandLogNextWriteIndex = 0;
uint32_t gCommandLogOverflowCount = 0;

size_t gSyncLogCount = 0;
size_t gSyncLogNextWriteIndex = 0;
uint32_t gSyncLogOverflowCount = 0;

char gPacketBuffer[Config::kPacketBufferLength];
uint32_t gLastWifiAttemptMs = 0;

void setup() {
  Serial.begin(Config::kSerialBaud);
  while (!Serial && millis() < 3000) {
  }

  for (size_t surfaceIndex = 0; surfaceIndex < Config::kSurfaceCount; ++surfaceIndex) {
    gServos[surfaceIndex].attach(
      Config::kServoPins[surfaceIndex],
      Config::kMinPulseUs[surfaceIndex],
      Config::kMaxPulseUs[surfaceIndex]);
    applySurfacePosition(surfaceIndex, Config::kNeutralPositions[surfaceIndex]);
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
  int packetSize = gUdp.parsePacket();
  while (packetSize > 0) {
    IPAddress remoteIp = gUdp.remoteIP();
    uint16_t remotePort = static_cast<uint16_t>(gUdp.remotePort());

    int bytesRead = gUdp.read(gPacketBuffer, Config::kPacketBufferLength - 1);
    if (bytesRead > 0) {
      gPacketBuffer[bytesRead] = '\0';
      handleCommand(gPacketBuffer, micros(), remoteIp, remotePort);
    }

    packetSize = gUdp.parsePacket();
  }
}

void handleCommand(char* packetBuffer, uint32_t boardRxUs, const IPAddress& remoteIp, uint16_t remotePort) {
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

  if (strcmp(commandName, "CLEAR_LOGS") == 0) {
    clearLogs();
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
    
  if (strcmp(commandName, "SET_ALL") == 0) {
    handleSetAllCommand(context, boardRxUs, remoteIp, remotePort);
    return;
  }

  sendErrorEvent(remoteIp, remotePort, F("UNKNOWN_COMMAND"));
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
  appendSyncLog(syncId, hostTxUs, boardRxUs, boardTxUs);

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

  positionNorm = constrain(positionNorm, 0.0f, 1.0f);
  uint16_t pulseUs = applySurfacePosition(static_cast<size_t>(surfaceIndex), positionNorm);
  uint32_t applyUs = micros();

  appendCommandLog(commandSequence, boardRxUs, applyUs, positionNorm, pulseUs, static_cast<uint8_t>(surfaceIndex));

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
  gUdp.print(positionNorm, 6);
  gUdp.print(',');
  gUdp.print(pulseUs);
  gUdp.endPacket();
}

void handleSetAllCommand(char* context, uint32_t boardRxUs, const IPAddress& remoteIp, uint16_t remotePort) {
  char* sampleSequenceToken = strtok_r(nullptr, ",", &context);
  char* surfaceCountToken = strtok_r(nullptr, ",", &context);

  uint32_t sampleSequence = 0;
  uint32_t surfaceCount = 0;

  if (!parseUnsigned32(sampleSequenceToken, sampleSequence) ||
      !parseUnsigned32(surfaceCountToken, surfaceCount)) {
    sendErrorEvent(remoteIp, remotePort, F("SET_ALL_PARSE_ERROR"));
    return;
  }

  if (surfaceCount == 0) {
    sendErrorEvent(remoteIp, remotePort, F("SET_ALL_EMPTY"));
    return;
  }

  if (surfaceCount > Config::kSurfaceCount) {
    sendErrorEvent(remoteIp, remotePort, F("SET_ALL_TOO_MANY_SURFACES"));
    return;
  }

  uint8_t surfaceIndices[Config::kSurfaceCount];
  uint32_t commandSequences[Config::kSurfaceCount];
  float positionNorms[Config::kSurfaceCount];
  uint16_t pulseUsValues[Config::kSurfaceCount];
  uint32_t applyUsValues[Config::kSurfaceCount];

  for (uint32_t k = 0; k < surfaceCount; ++k) {
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
      return;
    }

    surfaceIndices[k] = static_cast<uint8_t>(surfaceIndex);
    commandSequences[k] = commandSequence;
    positionNorms[k] = constrain(positionNorm, 0.0f, 1.0f);
  }

  // Apply the whole vector first so later surfaces are not delayed by telemetry output.
  for (uint32_t k = 0; k < surfaceCount; ++k) {
    pulseUsValues[k] = applySurfacePosition(surfaceIndices[k], positionNorms[k]);
    applyUsValues[k] = micros();
  }

  for (uint32_t k = 0; k < surfaceCount; ++k) {
    appendCommandLog(
      commandSequences[k],
      boardRxUs,
      applyUsValues[k],
      positionNorms[k],
      pulseUsValues[k],
      surfaceIndices[k]);

    if (!beginEventPacket(remoteIp, remotePort)) {
      continue;
    }

    gUdp.print(F("COMMAND_EVENT,"));
    gUdp.print(Config::kSurfaceNames[surfaceIndices[k]]);
    gUdp.print(',');
    gUdp.print(commandSequences[k]);
    gUdp.print(',');
    gUdp.print(boardRxUs);
    gUdp.print(',');
    gUdp.print(applyUsValues[k]);
    gUdp.print(',');
    gUdp.print(positionNorms[k], 6);
    gUdp.print(',');
    gUdp.print(pulseUsValues[k]);
    gUdp.endPacket();
  }
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
  gUdp.print(micros());
  gUdp.endPacket();
}

void sendStatusEvent(const IPAddress& remoteIp, uint16_t remotePort) {
  if (!beginEventPacket(remoteIp, remotePort)) {
    return;
  }

  gUdp.print(F("STATUS_EVENT,"));
  gUdp.print(F("command_log_count="));
  gUdp.print(gCommandLogCount);
  gUdp.print(F(",command_log_overflow="));
  gUdp.print(gCommandLogOverflowCount);
  gUdp.print(F(",sync_log_count="));
  gUdp.print(gSyncLogCount);
  gUdp.print(F(",sync_log_overflow="));
  gUdp.print(gSyncLogOverflowCount);
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
  if (!beginEventPacket(remoteIp, remotePort)) {
    return;
  }

  gUdp.print(F("ERR_EVENT,"));
  gUdp.print(message);
  gUdp.endPacket();
}

void clearLogs() {
  gCommandLogCount = 0;
  gCommandLogNextWriteIndex = 0;
  gCommandLogOverflowCount = 0;

  gSyncLogCount = 0;
  gSyncLogNextWriteIndex = 0;
  gSyncLogOverflowCount = 0;
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

void appendCommandLog(
  uint32_t commandSequence,
  uint32_t rxUs,
  uint32_t applyUs,
  float appliedPosition,
  uint16_t pulseUs,
  uint8_t surfaceIndex) {
  CommandLogEntry& entry = gCommandLog[gCommandLogNextWriteIndex];
  entry.commandSequence = commandSequence;
  entry.rxUs = rxUs;
  entry.applyUs = applyUs;
  entry.appliedPosition = appliedPosition;
  entry.pulseUs = pulseUs;
  entry.surfaceIndex = surfaceIndex;

  advanceRingBuffer(gCommandLogCount, gCommandLogNextWriteIndex, Config::kCommandLogCapacity, gCommandLogOverflowCount);
}

void appendSyncLog(uint32_t syncId, uint32_t hostTxUs, uint32_t boardRxUs, uint32_t boardTxUs) {
  SyncLogEntry& entry = gSyncLog[gSyncLogNextWriteIndex];
  entry.syncId = syncId;
  entry.hostTxUs = hostTxUs;
  entry.boardRxUs = boardRxUs;
  entry.boardTxUs = boardTxUs;

  advanceRingBuffer(gSyncLogCount, gSyncLogNextWriteIndex, Config::kSyncLogCapacity, gSyncLogOverflowCount);
}

void advanceRingBuffer(size_t& count, size_t& nextWriteIndex, size_t capacity, uint32_t& overflowCount) {
  nextWriteIndex = (nextWriteIndex + 1U) % capacity;

  if (count < capacity) {
    ++count;
  } else {
    ++overflowCount;
  }
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
