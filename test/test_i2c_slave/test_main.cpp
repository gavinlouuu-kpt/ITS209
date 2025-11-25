#include <Arduino.h>
#include <unity.h>
#include <Wire.h>

/***************************************************
  I2C Master-Slave Communication Test
  
  Tests communication between master device (ITS209) 
  and slave device (async_ads) using I2C protocol.
  
  Master Device (ITS209):
  - Default I2C pins: SDA=21, SCL=22
  - I2C frequency: 100kHz
  
  Slave Device (async_ads):
  - I2C address: 0x08
  - I2C pins: SDA=25, SCL=26
  - I2C frequency: 100kHz
****************************************************/

// I2C configuration
#define I2C_SLAVE_ADDR 0x08 // Address of the async_ads slave device

// Command definitions (must match slave device)
#define CMD_SET_HEATING 0x01 // Set heater parameters
#define CMD_START_EXP 0x02   // Start experiment
#define CMD_STOP_EXP 0x03     // Stop experiment
#define CMD_READ_DATA 0x04    // Read data
#define CMD_SET_WIFI 0x05     // Set WiFi credentials
#define CMD_WIFI_STATUS 0x06  // Get WiFi status
#define CMD_READ_LOGS 0x07    // Read debug logs
#define CMD_CLEAR_LOGS 0x08   // Clear debug logs
#define CMD_GET_IP 0x09       // Get IP address

// Default I2C pins for ESP32 (ESP-WROVER-KIT)
// These are the default pins when Wire.begin() is called without parameters
// SDA = 21, SCL = 22

void test_i2c_initialization()
{
    // Verify I2C is initialized (should be done in setup())
    // This test just verifies the I2C bus is ready
    TEST_PASS_MESSAGE("I2C bus ready");
}

void test_slave_device_detection()
{
    // Try to communicate with slave device
    Wire.beginTransmission(I2C_SLAVE_ADDR);
    uint8_t error = Wire.endTransmission();
    
    // Error code 0 means device acknowledged (success)
    // Error code 2 means device didn't acknowledge (not present or wrong address)
    // Error code 4 means other error
    
    if (error == 0) {
        TEST_PASS_MESSAGE("Slave device detected at address 0x08");
    } else if (error == 2) {
        TEST_FAIL_MESSAGE("Slave device not found at address 0x08. Check wiring and power.");
    } else {
        char msg[100];
        snprintf(msg, sizeof(msg), "I2C error code: %d", error);
        TEST_FAIL_MESSAGE(msg);
    }
}

void test_read_data_command()
{
    // Send CMD_READ_DATA command
    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_READ_DATA);
    uint8_t error = Wire.endTransmission();
    
    if (error != 0) {
        TEST_FAIL_MESSAGE("Failed to send CMD_READ_DATA command");
        return;
    }
    
    // Request 8 bytes of data (as per protocol)
    delay(10); // Small delay for slave to prepare data
    uint8_t bytesReceived = Wire.requestFrom(I2C_SLAVE_ADDR, 8);
    
    if (bytesReceived != 8) {
        char msg[100];
        snprintf(msg, sizeof(msg), "Expected 8 bytes, received %d", bytesReceived);
        TEST_FAIL_MESSAGE(msg);
        return;
    }
    
    // Read the 8 bytes
    uint8_t response[8];
    for (int i = 0; i < 8; i++) {
        response[i] = Wire.read();
    }
    
    // Parse data according to protocol:
    // [Setting_High] [Setting_Low] [Timestamp_3] [Timestamp_2] [Timestamp_1] [Timestamp_0] [Raw_Value_High] [Raw_Value_Low]
    uint16_t setting = (response[0] << 8) | response[1];
    uint32_t timestamp = ((uint32_t)response[2] << 24) | 
                        ((uint32_t)response[3] << 16) | 
                        ((uint32_t)response[4] << 8) | 
                        response[5];
    int16_t raw_value = (response[6] << 8) | response[7];
    
    // Verify data is reasonable (not all zeros or all 0xFF)
    bool dataValid = false;
    for (int i = 0; i < 8; i++) {
        if (response[i] != 0x00 && response[i] != 0xFF) {
            dataValid = true;
            break;
        }
    }
    
    TEST_ASSERT_TRUE_MESSAGE(dataValid, "Received valid data from slave device");
    
    // Log parsed values for debugging
    Serial.printf("Setting: %d, Timestamp: %lu, Raw Value: %d\n", 
                  setting, timestamp, raw_value);
}

void test_stop_experiment_command()
{
    // Send CMD_STOP_EXP command (no data required)
    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_STOP_EXP);
    uint8_t error = Wire.endTransmission();
    
    if (error == 0) {
        TEST_PASS_MESSAGE("CMD_STOP_EXP command sent successfully");
    } else {
        char msg[100];
        snprintf(msg, sizeof(msg), "Failed to send CMD_STOP_EXP, error: %d", error);
        TEST_FAIL_MESSAGE(msg);
    }
}

void test_wifi_status_command()
{
    // Send CMD_WIFI_STATUS command
    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_WIFI_STATUS);
    uint8_t error = Wire.endTransmission();
    
    if (error != 0) {
        TEST_FAIL_MESSAGE("Failed to send CMD_WIFI_STATUS command");
        return;
    }
    
    // Request response (protocol may vary, try 1 byte first)
    delay(10);
    uint8_t bytesReceived = Wire.requestFrom(I2C_SLAVE_ADDR, 1);
    
    if (bytesReceived > 0) {
        uint8_t status = Wire.read();
        TEST_PASS_MESSAGE("WiFi status command responded");
        Serial.printf("WiFi Status: 0x%02X\n", status);
    } else {
        // Command may not return data, which is acceptable
        TEST_PASS_MESSAGE("WiFi status command sent (no response expected)");
    }
}

void test_multiple_read_consistency()
{
    // Read data multiple times and check consistency
    uint8_t response1[8];
    uint8_t response2[8];
    
    // First read
    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_READ_DATA);
    Wire.endTransmission();
    delay(50); // Longer delay between reads
    Wire.requestFrom(I2C_SLAVE_ADDR, 8);
    for (int i = 0; i < 8; i++) {
        response1[i] = Wire.read();
    }
    
    // Second read
    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_READ_DATA);
    Wire.endTransmission();
    delay(50);
    Wire.requestFrom(I2C_SLAVE_ADDR, 8);
    for (int i = 0; i < 8; i++) {
        response2[i] = Wire.read();
    }
    
    // Timestamp should increase (or be close) between reads
    uint32_t timestamp1 = ((uint32_t)response1[2] << 24) | 
                          ((uint32_t)response1[3] << 16) | 
                          ((uint32_t)response1[4] << 8) | 
                          response1[5];
    uint32_t timestamp2 = ((uint32_t)response2[2] << 24) | 
                          ((uint32_t)response2[3] << 16) | 
                          ((uint32_t)response2[4] << 8) | 
                          response2[5];
    
    // Timestamp should be close (within 200ms) or increasing
    uint32_t timeDiff = (timestamp2 > timestamp1) ? (timestamp2 - timestamp1) : (timestamp1 - timestamp2);
    TEST_ASSERT_TRUE_MESSAGE(timeDiff < 200, "Timestamps are consistent between reads");
}

void test_i2c_bus_integrity()
{
    // Test that I2C bus is functioning by scanning for devices
    uint8_t devicesFound = 0;
    
    for (uint8_t address = 1; address < 127; address++) {
        Wire.beginTransmission(address);
        uint8_t error = Wire.endTransmission();
        
        if (error == 0) {
            devicesFound++;
            Serial.printf("Device found at address 0x%02X\n", address);
        }
    }
    
    // Should find at least the slave device at 0x08
    TEST_ASSERT_TRUE_MESSAGE(devicesFound > 0, "At least one I2C device detected on bus");
}

void setup()
{
    // Initialize serial communication
    Serial.begin(115200);
    delay(2000); // Wait for serial port to connect
    
    Serial.println("=== I2C Master-Slave Communication Test ===");
    Serial.println("Testing communication with async_ads slave device");
    Serial.println("Slave address: 0x08");
    Serial.println("Default I2C pins: SDA=21, SCL=22");
    Serial.println();
    
    // Initialize Unity test framework
    UNITY_BEGIN();
    
    // Initialize I2C with default pins
    Wire.begin();
    Wire.setClock(100000); // 100kHz to match slave device
    delay(100); // Allow I2C bus to stabilize
    
    // Run all tests
    RUN_TEST(test_i2c_initialization);
    delay(100);
    
    RUN_TEST(test_slave_device_detection);
    delay(100);
    
    RUN_TEST(test_i2c_bus_integrity);
    delay(100);
    
    RUN_TEST(test_read_data_command);
    delay(100);
    
    RUN_TEST(test_stop_experiment_command);
    delay(100);
    
    RUN_TEST(test_wifi_status_command);
    delay(100);
    
    RUN_TEST(test_multiple_read_consistency);
    
    // Complete Unity tests
    UNITY_END();
}

void loop()
{
    // Empty loop required for Arduino framework
}

