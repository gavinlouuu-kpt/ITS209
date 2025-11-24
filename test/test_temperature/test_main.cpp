#include <Arduino.h>
#include <unity.h>
/***************************************************
  This is a library for the Adafruit PT100/P1000 RTD Sensor w/MAX31865

  Designed specifically to work with the Adafruit RTD Sensor
  ----> https://www.adafruit.com/products/3328

  This sensor uses SPI to communicate, 4 pins are required to
  interface
  Adafruit invests time and resources providing this open source code,
  please support Adafruit and open-source hardware by purchasing
  products from Adafruit!

  Written by Limor Fried/Ladyada for Adafruit Industries.
  BSD license, all text above must be included in any redistribution
 ****************************************************/

#include <Adafruit_MAX31865.h>

// Use software SPI: CS, DI, DO, CLK
Adafruit_MAX31865 thermo = Adafruit_MAX31865(5, 23, 19, 18);
// use hardware SPI, just pass in the CS pin
// Adafruit_MAX31865 thermo = Adafruit_MAX31865(10);

// The value of the Rref resistor. Use 430.0 for PT100 and 4300.0 for PT1000
#define RREF 430.0
// The 'nominal' 0-degrees-C resistance of the sensor
// 100.0 for PT100, 1000.0 for PT1000
#define RNOMINAL 100.0

void test_temperature_range()
{
    // Read temperature from sensor
    float temperature = thermo.temperature(RNOMINAL, RREF);

    // Check for sensor faults
    uint8_t fault = thermo.readFault();
    if (fault)
    {
        TEST_FAIL_MESSAGE("Sensor fault detected");
        thermo.clearFault();
        return;
    }

    // Assert temperature is within reasonable range (0-60 degrees Celsius)
    TEST_ASSERT_TRUE(temperature >= 0.0 && temperature <= 60.0);
}

void setup()
{
    // Initialize serial communication
    Serial.begin(115200);
    delay(2000); // Wait for serial port to connect

    // Initialize Unity test framework
    UNITY_BEGIN();

    // Initialize MAX31865 sensor
    thermo.begin(MAX31865_2WIRE); // set to 2WIRE or 4WIRE as necessary

    // Run the temperature range test
    RUN_TEST(test_temperature_range);

    // Complete Unity tests
    UNITY_END();
}

void loop()
{
    // Empty loop required for Arduino framework
}
