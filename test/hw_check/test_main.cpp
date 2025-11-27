#include <Arduino.h>
#include <unity.h>
#include <Wire.h>
#include "hardware_config.h"
#include "Adafruit_SHT4x.h"
#include <Adafruit_MAX31865.h>

// Sensor instances for tests
static Adafruit_SHT4x sht4;
static Adafruit_MAX31865 thermo(MAX31865_CS, MAX31865_MOSI, MAX31865_MISO, MAX31865_SCLK);

static bool sht4Initialized = false;
static bool maxInitialized = false;

// Optional: common helpers
static void initI2C()
{
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(WIRE_CLOCK_HZ);
}

void test_i2c_device_present_at_sht40_addr()
{
    initI2C();
    Wire.beginTransmission(SHT4X_ADDR);
    uint8_t err = Wire.endTransmission();
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(0, err, "No device at SHT4X_ADDR");
}

void test_sht40_init_and_read()
{
    if (!sht4Initialized)
    {
        TEST_ASSERT_TRUE_MESSAGE(sht4.begin(SHT4X_ADDR), "SHT40 begin failed");
        sht4.setPrecision(SHT4X_HIGH_PRECISION);
        sht4.setHeater(SHT4X_NO_HEATER);
        sht4Initialized = true;
    }

    sensors_event_t humidity, temp;
    TEST_ASSERT_TRUE_MESSAGE(sht4.getEvent(&humidity, &temp), "SHT40 read failed");
    TEST_ASSERT_FALSE(isnan(temp.temperature));
    TEST_ASSERT_FALSE(isnan(humidity.relative_humidity));
    TEST_ASSERT_INT_WITHIN(200, 25, (int)temp.temperature);              // rough sanity: -175..225C window around 25C
    TEST_ASSERT_FLOAT_WITHIN(150.0f, 50.0f, humidity.relative_humidity); // -100..200% window
}

void test_max31865_init_and_read()
{
    if (!maxInitialized)
    {
        thermo.begin(MAX31865_WIRING);
        maxInitialized = true;
    }

    float temperature = thermo.temperature(RNOMINAL, RREF);
    uint8_t fault = thermo.readFault();
    if (fault)
    {
        // Print fault for diagnostics then fail the test
        Serial.print("MAX31865 fault: 0x");
        Serial.println(fault, HEX);
    }
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(0, fault, "MAX31865 fault present");
    TEST_ASSERT_FALSE(isnan(temperature));
    // Accept a broad range to avoid false negatives due to environment
    TEST_ASSERT_TRUE_MESSAGE(temperature > -100.0f && temperature < 300.0f, "MAX31865 temp out of plausible range");
}

void setup()
{
    delay(1000); // Give serial time to attach
    UNITY_BEGIN();

    RUN_TEST(test_i2c_device_present_at_sht40_addr);
    RUN_TEST(test_sht40_init_and_read);
    RUN_TEST(test_max31865_init_and_read);

    UNITY_END();
}

void loop()
{
    // not used
}
