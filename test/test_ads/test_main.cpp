#include <Arduino.h>
#include <Adafruit_ADS1X15.h>

Adafruit_ADS1115 ads;  /* Use this for the 16-bit version */
const int PWM_Heater = 19;  // PWM pin for heater control
const int PWM_CHANNEL = 0;  // PWM channel
const int PWM_FREQ = 10000; // PWM frequency
const int PWM_RESOLUTION = 8; // 8-bit resolution

// Pin connected to the ALERT/RDY signal for new sample notification.
constexpr int READY_PIN = 3;

// This is required on ESP32 to put the ISR in IRAM. Define as
// empty for other platforms. Be careful - other platforms may have
// other requirements.
#ifndef IRAM_ATTR
#define IRAM_ATTR
#endif

volatile bool new_data = false;
void IRAM_ATTR NewDataReadyISR() {
  new_data = true;
}

void setup(void)
{
  Serial.begin(230400);
  // Setup PWM for heater control
ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION);
ledcAttachPin(PWM_Heater, PWM_CHANNEL);
ledcWrite(PWM_CHANNEL, 200); 
  ads.setGain(GAIN_ONE);        // 1x gain   +/- 4.096V  1 bit = 2mV      0.125mV
  if (!ads.begin()) {
    Serial.println("Failed to initialize ADS.");
    while (1);
  }

  pinMode(READY_PIN, INPUT);
  // We get a falling edge every time a new sample is ready.
  attachInterrupt(digitalPinToInterrupt(READY_PIN), NewDataReadyISR, FALLING);

  // Start continuous conversions.
  ads.startADCReading(ADS1X15_REG_CONFIG_MUX_SINGLE_0, /*continuous=*/true);
}

void loop(void)
{
  // If we don't have new data, skip this iteration.
  if (!new_data) {
    return;
  }

  int16_t results = ads.getLastConversionResults();

  Serial.print("Differential: "); Serial.print(results); Serial.print("("); Serial.print(ads.computeVolts(results)); Serial.println("V)");

  new_data = false;

}

