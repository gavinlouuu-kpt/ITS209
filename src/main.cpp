#include <Arduino.h>
#include <Wire.h>
#include "Adafruit_SHT4x.h"
#include <Adafruit_MAX31865.h>

// pin definitions
#define SOL2_PIN 25
#define SOL1_PIN 26
#define HEAT_PIN 27
#define PUMP_PIN 13
#define HEAT_PWM_CHANNEL 0
#define PUMP_PWM_CHANNEL 1

// I2C pins for SHT40
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22

// I2C slave configuration (async_ads device)
#define I2C_SLAVE_ADDR 0x08 // Address of the async_ads slave device
#define CMD_READ_DATA 0x04  // Read data command

// MAX31865 (software SPI) pins and constants
#define MAX31865_CS 5
#define MAX31865_MOSI 23
#define MAX31865_MISO 19
#define MAX31865_SCLK 18
#define RNOMINAL 100.0 // PT100
#define RREF 430.0     // Reference resistor for PT100

// Sensor instances
Adafruit_SHT4x sht4 = Adafruit_SHT4x();
Adafruit_MAX31865 thermo = Adafruit_MAX31865(MAX31865_CS, MAX31865_MOSI, MAX31865_MISO, MAX31865_SCLK);

static void scanI2CBus()
{
  Serial.println("I2C scan start");
  uint8_t found = 0;
  for (uint8_t address = 1; address < 127; ++address)
  {
    Wire.beginTransmission(address);
    uint8_t error = Wire.endTransmission();
    if (error == 0)
    {
      Serial.printf(" - I2C device found at 0x%02X\n", address);
      found++;
    }
    else if (error == 4)
    {
      Serial.printf(" - Unknown error at 0x%02X\n", address);
    }
  }
  if (found == 0)
  {
    Serial.println("No I2C devices found");
  }
  Serial.println("I2C scan done");
}

// FreeRTOS tasks
static void taskMax31865Read(void *pv)
{
  // Initialize MAX31865 (2-wire as confirmed)
  thermo.begin(MAX31865_2WIRE);

  for (;;)
  {
    float temperature = thermo.temperature(RNOMINAL, RREF);
    uint8_t fault = thermo.readFault();
    if (fault)
    {
      Serial.print("MAX31865 fault: 0x");
      Serial.println(fault, HEX);
      thermo.clearFault();
    }
    else
    {
      Serial.printf("MAX31865: %.2f C\n", temperature);
    }
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}

static void taskSht40Read(void *pv)
{
  // Initialize I2C and SHT40
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(100000);
  bool sht4Initialized = false;

  if (sht4.begin(&Wire))
  {
    sht4.setPrecision(SHT4X_HIGH_PRECISION);
    sht4.setHeater(SHT4X_NO_HEATER);
    sht4Initialized = true;
  }
  else
  {
    Serial.println("SHT40 not found");
    scanI2CBus();
  }

  for (;;)
  {
    // If sensor wasn't found at start, try again occasionally
    static uint32_t retryCounter = 0;
    static bool scannedOnce = true; // already scanned above if init failed
    if (!sht4Initialized)
    {
      if ((retryCounter++ % 10) == 0)
      {
        if (sht4.begin(&Wire))
        {
          sht4.setPrecision(SHT4X_HIGH_PRECISION);
          sht4.setHeater(SHT4X_NO_HEATER);
          sht4Initialized = true;
          Serial.println("SHT40 initialized");
        }
        else if (!scannedOnce)
        {
          scanI2CBus();
          scannedOnce = true;
        }
      }
      vTaskDelay(pdMS_TO_TICKS(2000));
      continue;
    }

    sensors_event_t humidity, temp;
    sht4.getEvent(&humidity, &temp);
    if (!isnan(temp.temperature) && !isnan(humidity.relative_humidity))
    {
      Serial.printf("SHT40: T=%.2f C, RH=%.2f %%\n", temp.temperature, humidity.relative_humidity);
    }
    else
    {
      Serial.println("SHT40: read error");
    }
    vTaskDelay(pdMS_TO_TICKS(2000));
  }
}

static void taskI2CSlaveRead(void *pv)
{
  // I2C bus is already initialized by taskSht40Read, so we use the shared Wire instance
  // Wait a bit to ensure I2C is initialized
  vTaskDelay(pdMS_TO_TICKS(500));

  for (;;)
  {
    // Request data from slave
    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_READ_DATA);
    uint8_t result = Wire.endTransmission();

    if (result != 0)
    {
      Serial.printf("I2C Slave: Failed to request data (error: %d)\n", result);
      vTaskDelay(pdMS_TO_TICKS(2000));
      continue;
    }

    // Request 8 bytes of response data
    uint8_t bytesReceived = Wire.requestFrom(I2C_SLAVE_ADDR, 8);

    if (bytesReceived == 8)
    {
      uint8_t response[8];
      for (int i = 0; i < 8; i++)
      {
        response[i] = Wire.read();
      }

      // Parse the response according to slave's format
      uint16_t setting = (response[0] << 8) | response[1];
      uint32_t timestamp = (response[2] << 24) | (response[3] << 16) | (response[4] << 8) | response[5];
      int16_t raw_value = (response[6] << 8) | response[7];

      // Check if data is valid (non-zero)
      if (setting != 0 || timestamp != 0 || raw_value != 0)
      {
        Serial.printf("I2C Slave: Setting=%d, Timestamp=%lu, Raw=%d\n", setting, timestamp, raw_value);
      }
      else
      {
        Serial.println("I2C Slave: No new data available (experiment might be idle)");
      }
    }
    else
    {
      Serial.printf("I2C Slave: Expected 8 bytes, received %d bytes\n", bytesReceived);
    }

    vTaskDelay(pdMS_TO_TICKS(2000)); // Read every 2 seconds
  }
}

static void taskSol1Toggle(void *pv)
{
  for (;;)
  {
    digitalWrite(SOL1_PIN, HIGH);
    vTaskDelay(pdMS_TO_TICKS(5000));
    digitalWrite(SOL1_PIN, LOW);
    vTaskDelay(pdMS_TO_TICKS(5000));
  }
}

static void taskSol2Toggle(void *pv)
{
  for (;;)
  {
    digitalWrite(SOL2_PIN, HIGH);
    vTaskDelay(pdMS_TO_TICKS(5000));
    digitalWrite(SOL2_PIN, LOW);
    vTaskDelay(pdMS_TO_TICKS(5000));
  }
}

static void taskHeaterToggle(void *pv)
{
  for (;;)
  {
    ledcWrite(HEAT_PWM_CHANNEL, 255); // 100% duty
    vTaskDelay(pdMS_TO_TICKS(5000));
    ledcWrite(HEAT_PWM_CHANNEL, 0);
    vTaskDelay(pdMS_TO_TICKS(5000));
  }
}

static void taskPumpToggle(void *pv)
{
  for (;;)
  {
    ledcWrite(PUMP_PWM_CHANNEL, 255); // 100% duty
    vTaskDelay(pdMS_TO_TICKS(5000));
    ledcWrite(PUMP_PWM_CHANNEL, 0);
    vTaskDelay(pdMS_TO_TICKS(5000));
  }
}

void setup()
{
  Serial.begin(115200);

  // initialize pins
  pinMode(SOL2_PIN, OUTPUT);
  pinMode(SOL1_PIN, OUTPUT);
  pinMode(HEAT_PIN, OUTPUT);
  pinMode(PUMP_PIN, OUTPUT);

  // setup heater and pump with pwm
  ledcSetup(HEAT_PWM_CHANNEL, 1000, 8);
  ledcSetup(PUMP_PWM_CHANNEL, 1000, 8);
  ledcAttachPin(HEAT_PIN, HEAT_PWM_CHANNEL);
  ledcAttachPin(PUMP_PIN, PUMP_PWM_CHANNEL);

  // Create tasks
  xTaskCreate(taskMax31865Read, "tMAX", 4096, nullptr, 2, nullptr);
  xTaskCreate(taskSht40Read, "tSHT40", 4096, nullptr, 2, nullptr);
  xTaskCreate(taskI2CSlaveRead, "tI2CSlave", 4096, nullptr, 2, nullptr);
  xTaskCreate(taskSol1Toggle, "tSOL1", 2048, nullptr, 1, nullptr);
  xTaskCreate(taskSol2Toggle, "tSOL2", 2048, nullptr, 1, nullptr);
  xTaskCreate(taskHeaterToggle, "tHEAT", 2048, nullptr, 1, nullptr);
  xTaskCreate(taskPumpToggle, "tPUMP", 2048, nullptr, 1, nullptr);
}

void loop()
{
  vTaskDelay(pdMS_TO_TICKS(1000));
}
