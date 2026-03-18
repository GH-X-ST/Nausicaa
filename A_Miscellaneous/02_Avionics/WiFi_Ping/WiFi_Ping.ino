#include <SPI.h>
#include <WiFiNINA.h>

char ssid[] = "FlightArena_2.4G";
char pass[] = "R0b0t1c$";

int status = WL_IDLE_STATUS;

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    while (true) {}
  }

  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to WPA SSID: ");
    Serial.println(ssid);
    status = WiFi.begin(ssid, pass);
    delay(5000);
  }

  Serial.println("Connected.");
  Serial.print("Board IP: ");
  Serial.println(WiFi.localIP());

  Serial.print("Subnet: ");
  Serial.println(WiFi.subnetMask());

  Serial.print("Gateway: ");
  Serial.println(WiFi.gatewayIP());
}

void loop() {
  IPAddress gateway = WiFi.gatewayIP();
  IPAddress host(192, 168, 0, 209);   // laptop/host IP

  int rttGw = WiFi.ping(gateway);
  Serial.print("Ping gateway ");
  Serial.print(gateway);
  Serial.print(" -> ");
  Serial.println(rttGw);

  int rttHost = WiFi.ping(host);
  Serial.print("Ping host ");
  Serial.print(host);
  Serial.print(" -> ");
  Serial.println(rttHost);

  delay(3000);
}