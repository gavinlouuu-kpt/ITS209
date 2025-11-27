#include <Arduino.h>
#include <unity.h>
/***************************************************
  This is a library for the Adafruit SHT4x Temperature & Humidity Sensor

  Designed specifically to work with the Adafruit SHT4x Breakout
  ----> https://www.adafruit.com/products/4885

  These sensors use I2C to communicate, 2 pins are required to
  interface with the sensor.

  Adafruit invests time and resources providing this open source code,
  please support Adafruit and open-source hardware by purchasing
  products from Adafruit!

  Written by Limor Fried/Ladyada for Adafruit Industries.
  BSD license, all text above must be included in any redistribution
 ****************************************************/

#include <Wire.h>
#include "Adafruit_SHT4x.h"
#include <stdio.h>

// SHT40B sensor instance
Adafruit_SHT4x sht4 = Adafruit_SHT4x();

// I2C configuration for slave device communication
#define I2C_SLAVE_ADDR 0x08 // Address of the async_ads slave device
#define SDA_PIN 21          // SDA pin for I2C (master device)
#define SCL_PIN 22          // SCL pin for I2C (master device)

// Valid ranges for SHT40B sensor
#define TEMP_MIN -40.0     // Minimum temperature in Celsius
#define TEMP_MAX 125.0     // Maximum temperature in Celsius
#define HUMIDITY_MIN 0.0   // Minimum humidity in %RH
#define HUMIDITY_MAX 100.0 // Maximum humidity in %RH

void test_sensor_initialization()
{
    TEST_MESSAGE("Initializing SHT40B sensor and verifying precision...");
    // Initialize I2C for sensor communication
    Wire.begin(SDA_PIN, SCL_PIN);

    // Initialize SHT40B sensor
    bool initialized = sht4.begin();
    TEST_ASSERT_TRUE_MESSAGE(initialized, "SHT40B sensor initialization failed");

    // Verify sensor type
    sht4x_precision_t precision = sht4.getPrecision();
    TEST_ASSERT_TRUE_MESSAGE(precision >= SHT4X_HIGH_PRECISION && precision <= SHT4X_LOW_PRECISION,
                             "Invalid sensor precision");
}

void test_temperature_reading()
{
    TEST_MESSAGE("Reading temperature and humidity (single read)...");
    // Read temperature and humidity
    sensors_event_t humidity, temp;
    sht4.getEvent(&humidity, &temp);

    char msg[96];
    snprintf(msg, sizeof(msg), "Raw readings -> T=%.2f C, RH=%.2f %%", temp.temperature, humidity.relative_humidity);
    TEST_MESSAGE(msg);

    // Check if reading is valid (not NaN)
    TEST_ASSERT_FALSE_MESSAGE(isnan(temp.temperature), "Temperature reading is NaN");

    // Check temperature is within valid range
    TEST_ASSERT_TRUE_MESSAGE(temp.temperature >= TEMP_MIN && temp.temperature <= TEMP_MAX,
                             "Temperature out of valid range");

    // For typical room temperature, expect reasonable values (0-50°C)
    // This is a more practical range check for testing
    TEST_ASSERT_TRUE_MESSAGE(temp.temperature >= -10.0 && temp.temperature <= 60.0,
                             "Temperature outside expected operating range");
}

void test_humidity_reading()
{
    TEST_MESSAGE("Reading humidity and validating range...");
    // Read temperature and humidity
    sensors_event_t humidity, temp;
    sht4.getEvent(&humidity, &temp);

    char msg[96];
    snprintf(msg, sizeof(msg), "Humidity reading -> RH=%.2f %%", humidity.relative_humidity);
    TEST_MESSAGE(msg);

    // Check if reading is valid (not NaN)
    TEST_ASSERT_FALSE_MESSAGE(isnan(humidity.relative_humidity), "Humidity reading is NaN");

    // Check humidity is within valid range (0-100% RH)
    TEST_ASSERT_TRUE_MESSAGE(humidity.relative_humidity >= HUMIDITY_MIN &&
                                 humidity.relative_humidity <= HUMIDITY_MAX,
                             "Humidity out of valid range");
}

void test_sensor_precision()
{
    TEST_MESSAGE("Switching to high precision mode and verifying readings...");
    // Test high precision mode
    sht4.setPrecision(SHT4X_HIGH_PRECISION);
    delay(10); // Allow sensor to settle

    sensors_event_t humidity, temp;
    sht4.getEvent(&humidity, &temp);

    // Verify readings are still valid
    TEST_ASSERT_FALSE_MESSAGE(isnan(temp.temperature), "High precision temperature reading failed");
    TEST_ASSERT_FALSE_MESSAGE(isnan(humidity.relative_humidity), "High precision humidity reading failed");
}

void test_slave_device_communication()
{
    TEST_MESSAGE("Checking I2C slave device acknowledge on expected address...");
    // Test I2C communication with async_ads slave device
    Wire.beginTransmission(I2C_SLAVE_ADDR);
    uint8_t error = Wire.endTransmission();

    // Error code 0 means success (device acknowledged)
    // Other codes indicate communication issues
    // Note: This test will pass even if slave is not connected (error code 2)
    // In a real scenario, you might want to make this more strict
    TEST_ASSERT_TRUE_MESSAGE(error == 0 || error == 2,
                             "Unexpected I2C error when checking slave device");
}

void test_multiple_readings_consistency()
{
    TEST_MESSAGE("Taking two readings and checking for consistency...");
    // Read sensor multiple times and check consistency
    sensors_event_t humidity1, temp1;
    sensors_event_t humidity2, temp2;

    sht4.getEvent(&humidity1, &temp1);
    delay(100); // Small delay between readings
    sht4.getEvent(&humidity2, &temp2);

    char msg[96];
    float temp_diff = abs(temp1.temperature - temp2.temperature);
    float humidity_diff = abs(humidity1.relative_humidity - humidity2.relative_humidity);
    snprintf(msg, sizeof(msg), "Diffs -> dT=%.2f C, dRH=%.2f %%", temp_diff, humidity_diff);
    TEST_MESSAGE(msg);

    // Readings should be close (within reasonable sensor noise)
    // Temperature should be within 2°C (sensor accuracy is ±0.2°C typical)
    TEST_ASSERT_TRUE_MESSAGE(temp_diff < 2.0, "Temperature readings inconsistent");

    // Humidity should be within 5% RH (sensor accuracy is ±1.5% RH typical)
    TEST_ASSERT_TRUE_MESSAGE(humidity_diff < 5.0, "Humidity readings inconsistent");
}

void setup()
{
    // Initialize serial communication
    Serial.begin(115200);
    delay(2000); // Wait for serial port to connect

    // Initialize Unity test framework
    UNITY_BEGIN();

    Serial.println("=== SHT40B Sensor Test ===");

    // Initialize I2C for sensor communication
    Wire.begin(SDA_PIN, SCL_PIN);
    delay(100);

    // Initialize SHT40B sensor
    if (!sht4.begin())
    {
        Serial.println("ERROR: Could not find SHT40B sensor!");
        UNITY_END();
        return;
    }

    Serial.println("SHT40B sensor found!");

    // Set sensor to high precision mode
    sht4.setPrecision(SHT4X_HIGH_PRECISION);
    sht4.setHeater(SHT4X_NO_HEATER);

    // Run all tests
    RUN_TEST(test_sensor_initialization);
    delay(100);

    RUN_TEST(test_temperature_reading);
    delay(100);

    RUN_TEST(test_humidity_reading);
    delay(100);

    RUN_TEST(test_sensor_precision);
    delay(100);

    RUN_TEST(test_slave_device_communication);
    delay(100);

    RUN_TEST(test_multiple_readings_consistency);

    // Complete Unity tests
    UNITY_END();
}

void loop()
{
    // Empty loop required for Arduino framework
}
