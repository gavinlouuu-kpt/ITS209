#include <Arduino.h>
#include <Wire.h>

#define SDA_PIN 21
#define SCL_PIN 22

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        delay(10);
    }
    
    Serial.println("=== I2C Scanner ===");
    
    // Initialize I2C
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(100000); // 100kHz
    
    Serial.println("I2C Scanner starting...");
    Serial.println("Scanning addresses 0x01 to 0x7F...");
}

void loop() {
    byte error, address;
    int nDevices = 0;
    
    Serial.println("Scanning...");
    
    for(address = 1; address < 127; address++) {
        Wire.beginTransmission(address);
        error = Wire.endTransmission();
        
        if (error == 0) {
            Serial.print("I2C device found at address 0x");
            if (address < 16) Serial.print("0");
            Serial.print(address, HEX);
            Serial.println(" !");
            nDevices++;
        }
        else if (error == 4) {
            Serial.print("Unknown error at address 0x");
            if (address < 16) Serial.print("0");
            Serial.println(address, HEX);
        }
    }
    
    if (nDevices == 0) {
        Serial.println("No I2C devices found");
        Serial.println("Check wiring:");
        Serial.println("- SDA (pin 21) connected between both ESP32s");
        Serial.println("- SCL (pin 22) connected between both ESP32s"); 
        Serial.println("- GND connected between both ESP32s");
        Serial.println("- Both ESP32s powered on");
        Serial.println("- Slave ESP32 running main.cpp");
    }
    else {
        Serial.print("Found ");
        Serial.print(nDevices);
        Serial.println(" device(s)");
    }
    
    Serial.println("--------------------------------");
    delay(5000); // Wait 5 seconds before next scan
}

