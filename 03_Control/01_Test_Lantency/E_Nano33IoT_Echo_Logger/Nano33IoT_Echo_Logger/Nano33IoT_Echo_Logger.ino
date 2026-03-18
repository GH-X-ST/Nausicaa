#include <SPI.h>
#include <Servo.h>
#include <WiFiNINA.h>

namespace Config {
constexpr char kFirmwareVersion[] = "Nano33IoT_Echo_Logger_V1";
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
constexpr size_t kLineBufferLength = 128;
constexpr bool kReplyToSetCommands = false;
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

WiFiServer gServer(Config::kServerPort);
WiFiClient gClient;
Servo gServos[Config::kSurfaceCount];

CommandLogEntry gCommandLog[Config::kCommandLogCapacity];
SyncLogEntry gSyncLog[Config::kSyncLogCapacity];

size_t gCommandLogCount = 0;
size_t gCommandLogNextWriteIndex = 0;
uint32_t gCommandLogOverflowCount = 0;

size_t gSyncLogCount = 0;
size_t gSyncLogNextWriteIndex = 0;
uint32_t gSyncLogOverflowCount = 0;

char gLineBuffer[Config::kLineBufferLength];
size_t gLineLength = 0;
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
  gServer.begin();

  Serial.println(F("Nano33IoT echo logger ready."));
}

void loop() {
  maintainWifiConnection();
  acceptClientIfNeeded();
  serviceClient();
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

void acceptClientIfNeeded() {
  if (gClient && gClient.connected()) {
    return;
  }

  if (gClient) {
    gClient.stop();
  }

  WiFiClient nextClient = gServer.available();
  if (!nextClient) {
    return;
  }

  gClient = nextClient;
  gLineLength = 0;
  sendHelloReply();
}

void serviceClient() {
  if (!gClient || !gClient.connected()) {
    return;
  }

  while (gClient.available() > 0) {
    char nextChar = static_cast<char>(gClient.read());
    if (nextChar == '\r') {
      continue;
    }

    if (nextChar == '\n') {
      if (gLineLength == 0) {
        continue;
      }

      gLineBuffer[gLineLength] = '\0';
      handleCommand(gLineBuffer, micros());
      gLineLength = 0;
      continue;
    }

    if (gLineLength + 1 >= Config::kLineBufferLength) {
      gLineLength = 0;
      sendError(F("LINE_TOO_LONG"));
      continue;
    }

    gLineBuffer[gLineLength++] = nextChar;
  }
}

void handleCommand(char* lineBuffer, uint32_t lineCompleteUs) {
  char* context = nullptr;
  char* commandName = strtok_r(lineBuffer, ",", &context);
  if (commandName == nullptr) {
    return;
  }

  if (strcmp(commandName, "HELLO") == 0) {
    sendHelloReply();
    return;
  }

  if (strcmp(commandName, "STATUS") == 0) {
    sendStatusReply();
    return;
  }

  if (strcmp(commandName, "CLEAR_LOGS") == 0) {
    clearLogs();
    sendOk(F("CLEAR_LOGS"));
    return;
  }

  if (strcmp(commandName, "SET_NEUTRAL") == 0) {
    applyNeutralToAllSurfaces();
    sendOk(F("SET_NEUTRAL"));
    return;
  }

  if (strcmp(commandName, "SYNC") == 0) {
    handleSyncCommand(context, lineCompleteUs);
    return;
  }

  if (strcmp(commandName, "SET") == 0) {
    handleSetCommand(context, lineCompleteUs);
    return;
  }

  if (strcmp(commandName, "DUMP_COMMAND_LOG") == 0) {
    dumpCommandLog();
    return;
  }

  if (strcmp(commandName, "DUMP_SYNC_LOG") == 0) {
    dumpSyncLog();
    return;
  }

  sendError(F("UNKNOWN_COMMAND"));
}

void handleSyncCommand(char* context, uint32_t boardRxUs) {
  char* syncIdToken = strtok_r(nullptr, ",", &context);
  char* hostTxUsToken = strtok_r(nullptr, ",", &context);

  uint32_t syncId = 0;
  uint32_t hostTxUs = 0;
  if (!parseUnsigned32(syncIdToken, syncId) || !parseUnsigned32(hostTxUsToken, hostTxUs)) {
    sendError(F("SYNC_PARSE_ERROR"));
    return;
  }

  uint32_t boardTxUs = micros();
  appendSyncLog(syncId, hostTxUs, boardRxUs, boardTxUs);

  gClient.print(F("SYNC_REPLY,"));
  gClient.print(syncId);
  gClient.print(',');
  gClient.print(hostTxUs);
  gClient.print(',');
  gClient.print(boardRxUs);
  gClient.print(',');
  gClient.println(boardTxUs);
}

void handleSetCommand(char* context, uint32_t boardRxUs) {
  char* surfaceNameToken = strtok_r(nullptr, ",", &context);
  char* sequenceToken = strtok_r(nullptr, ",", &context);
  char* positionToken = strtok_r(nullptr, ",", &context);

  int surfaceIndex = findSurfaceIndex(surfaceNameToken);
  uint32_t commandSequence = 0;
  float positionNorm = 0.0f;
  if (surfaceIndex < 0 || !parseUnsigned32(sequenceToken, commandSequence) || !parseFloat(positionToken, positionNorm)) {
    sendError(F("SET_PARSE_ERROR"));
    return;
  }

  positionNorm = constrain(positionNorm, 0.0f, 1.0f);
  uint16_t pulseUs = applySurfacePosition(static_cast<size_t>(surfaceIndex), positionNorm);
  uint32_t applyUs = micros();

  appendCommandLog(commandSequence, boardRxUs, applyUs, positionNorm, pulseUs, static_cast<uint8_t>(surfaceIndex));

  if (Config::kReplyToSetCommands) {
    gClient.print(F("SET_REPLY,"));
    gClient.print(Config::kSurfaceNames[surfaceIndex]);
    gClient.print(',');
    gClient.print(commandSequence);
    gClient.print(',');
    gClient.print(boardRxUs);
    gClient.print(',');
    gClient.println(applyUs);
  }
}

void sendHelloReply() {
  gClient.print(F("HELLO_REPLY,"));
  gClient.print(Config::kFirmwareVersion);
  gClient.print(',');
  gClient.print(WiFi.localIP());
  gClient.print(',');
  gClient.print(Config::kServerPort);
  gClient.print(',');
  gClient.println(micros());
}

void sendStatusReply() {
  gClient.print(F("STATUS_REPLY,"));
  gClient.print(F("command_log_count="));
  gClient.print(gCommandLogCount);
  gClient.print(F(",command_log_overflow="));
  gClient.print(gCommandLogOverflowCount);
  gClient.print(F(",sync_log_count="));
  gClient.print(gSyncLogCount);
  gClient.print(F(",sync_log_overflow="));
  gClient.print(gSyncLogOverflowCount);
  gClient.print(F(",wifi_status="));
  gClient.println(WiFi.status());
}

void sendOk(const __FlashStringHelper* message) {
  gClient.print(F("OK,"));
  gClient.println(message);
}

void sendError(const __FlashStringHelper* message) {
  if (!gClient || !gClient.connected()) {
    return;
  }

  gClient.print(F("ERR,"));
  gClient.println(message);
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

void dumpCommandLog() {
  gClient.println(F("#COMMAND_LOG_BEGIN,V1"));
  gClient.print(F("#overflow_count="));
  gClient.println(gCommandLogOverflowCount);
  gClient.println(F("surface_name,command_sequence,rx_us,apply_us,receive_to_apply_us,applied_position,pulse_us"));

  size_t startIndex = oldestCommandLogIndex();
  for (size_t rowIndex = 0; rowIndex < gCommandLogCount; ++rowIndex) {
    const CommandLogEntry& entry = gCommandLog[(startIndex + rowIndex) % Config::kCommandLogCapacity];
    gClient.print(Config::kSurfaceNames[entry.surfaceIndex]);
    gClient.print(',');
    gClient.print(entry.commandSequence);
    gClient.print(',');
    gClient.print(entry.rxUs);
    gClient.print(',');
    gClient.print(entry.applyUs);
    gClient.print(',');
    gClient.print(entry.applyUs - entry.rxUs);
    gClient.print(',');
    gClient.print(entry.appliedPosition, 6);
    gClient.print(',');
    gClient.println(entry.pulseUs);
  }

  gClient.println(F("#COMMAND_LOG_END"));
}

void dumpSyncLog() {
  gClient.println(F("#SYNC_LOG_BEGIN,V1"));
  gClient.print(F("#overflow_count="));
  gClient.println(gSyncLogOverflowCount);
  gClient.println(F("sync_id,host_tx_us,board_rx_us,board_tx_us,board_turnaround_us"));

  size_t startIndex = oldestSyncLogIndex();
  for (size_t rowIndex = 0; rowIndex < gSyncLogCount; ++rowIndex) {
    const SyncLogEntry& entry = gSyncLog[(startIndex + rowIndex) % Config::kSyncLogCapacity];
    gClient.print(entry.syncId);
    gClient.print(',');
    gClient.print(entry.hostTxUs);
    gClient.print(',');
    gClient.print(entry.boardRxUs);
    gClient.print(',');
    gClient.print(entry.boardTxUs);
    gClient.print(',');
    gClient.println(entry.boardTxUs - entry.boardRxUs);
  }

  gClient.println(F("#SYNC_LOG_END"));
}

size_t oldestCommandLogIndex() {
  if (gCommandLogCount < Config::kCommandLogCapacity) {
    return 0;
  }

  return gCommandLogNextWriteIndex;
}

size_t oldestSyncLogIndex() {
  if (gSyncLogCount < Config::kSyncLogCapacity) {
    return 0;
  }

  return gSyncLogNextWriteIndex;
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
