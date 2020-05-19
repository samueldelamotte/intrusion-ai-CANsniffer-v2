/****************************************************************************
                  (MCP2515) CAN Bus Read and Log to SD card.
                    
Author: Samuel De La Motte
Contact: samuel.delamotte1@gmail.com
Date: 18/05/2020
Version: 1.1

Based off the Arduino script found at the following link:
https://github.com/alexandreblin/arduino-can-reader
*****************************************************************************/
//*******************************LIBRARIES**********************************//
#include <SPI.h>
#include <SD.h>
#include "mcp_can.h"

//******************************DEFINITIONS*********************************//
// Serial port data rate
#define SERIAL_SPEED 115200

// CAN bus data rate
#define CAN_BITRATE 500

// Joystick click pin
#define CLICK A4

//Define LED pins
#define LED2 8
#define LED3 7

//*******************************CONSTANTS**********************************//
// Initialise chip select pins
const int CAN_CS_PIN = 10; // MCP2515 pin

// Set the CAN bus speed according to CAN_BITRATE
#if (CAN_BITRATE == 500)
  const byte CAN_SPEED = CAN_500KBPS;
#elif (CAN_BITRATE == 250)
  const byte CAN_SPEED = CAN_250KBPS;
#elif (CAN_BITRATE == 125)
  const byte CAN_SPEED = CAN_125KBPS;
#else
  const byte CAN_SPEED = CAN_500KBPS;
#endif

//********************************GLOBALS***********************************//
// CAN Object
MCP_CAN CAN(CAN_CS_PIN);

// CAN data
unsigned char len = 0;
byte buffer[8];

//*******************************SETUP LOOP*********************************//
void setup() {
  // Initialise pins
  pinMode(CLICK,INPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT);

  // Pull analog pins high to enable reading of joystick click
  digitalWrite(CLICK, HIGH);
  
  // Write LED pins low to turn them off by default
  digitalWrite(LED2, LOW);
  digitalWrite(LED3, LOW);

  // Initialise serial bus
  Serial.begin(SERIAL_SPEED);
  if (Serial) {
    Serial.print("[INFO] - SERIAL SPEED: "); Serial.println(SERIAL_SPEED);
  } else {
    while (1) {
      Serial.println("[ERROR] - CAN'T INITIALISE USB SERIAL");
      delay(100);
    }
  }

  // Initialise MCP2515 CAN controller at the specified speed
  Serial.print("[INFO] - CAN BITRATE: "); Serial.println(CAN_BITRATE);
  byte canSpeed = CAN_SPEED;
  while (CAN.begin(canSpeed) != CAN_OK) {
    Serial.println("[ERROR] - CAN'T INITIALISE CAN");
    delay(100);
  }

  // CAN reader and data logger is ready
  Serial.println("[INFO] - SYSTEM READY");
  Serial.println("Press the joystick to begin.");

  // Wait for user to click joystick to begin logging
  while (digitalRead(CLICK)==HIGH) {
    // Waiting...
  }
  
  Serial.println("[INFO] - SYSTEM STARTING...");
  Serial.println("3");
  delay(1000);
  Serial.println("2");
  delay(1000);
  Serial.println("1");
  delay(1000);
}

//********************************MAIN LOOP*********************************//
void loop() {
  while (digitalRead(CLICK)==HIGH) {
    if (CAN.checkReceive() == CAN_MSGAVAIL) {
      // Turn on LED to indicate CAN Bus traffic
      digitalWrite(LED3, HIGH);
      
      // Get CAN message
      CAN.readMsgBuf(&len, buffer);

      // Print CAN ID and LEN
      Serial.print(CAN.getCanId());
      Serial.print(",");
      Serial.print(len);

      // Print CAN data
      char tmp[3];
      for(int i = 0; i<len; i++) {
        Serial.print(",");
        snprintf(tmp, 3, "%02X", buffer[i]);
        Serial.print(tmp);
      }
      Serial.println();
    }
    
    // Turn off LED3
    digitalWrite(LED3, LOW);
  }
  // User pressed the joystick again to stop logging
  while (1) {
    // Stopped...
  }
}
