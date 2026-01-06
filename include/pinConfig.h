#ifndef PIN_CONFIG_H
#define PIN_CONFIG_H

#include <Arduino.h>

//----------------------------------------------------------
// Master Board Pin Definitions
//----------------------------------------------------------

#define HW_SOL_1 25 // Pin for hardware solenoid 1
#define HW_SOL_2 26 // Pin for hardware solenoid 2
#define HW_FPC_HEATER 27 // Pin for hardware FPC heater controlled with PWM
#define HW_PUMP 13 // Pin for hardware pump controlled with PWM

// PWM Channels
#define PWM_CHANNEL_HW_FPC_HEATER 0 // PWM channel for hardware FPC heater
#define PWM_CHANNEL_HW_PUMP 1 // PWM channel for hardware pump

// PWM Frequencies
#define PWM_FREQ_HW_FPC_HEATER 10000 // PWM frequency for hardware FPC heater
#define PWM_FREQ_HW_PUMP 10000 // PWM frequency for hardware pump

// PWM Resolutions
#define PWM_RESOLUTION_HW_FPC_HEATER 8 // PWM resolution for hardware FPC heater
#define PWM_RESOLUTION_HW_PUMP 8 // PWM resolution for hardware pump

// MAX31865 Pt100 Module Pin Definitions 
#define MAX31865_CS 5 // CS pin for MAX31865
#define MAX31865_CLK 18 // CLK pin for MAX31865
#define MAX31865_DO 23 // MOSI pin for MAX31865
#define MAX31865_DI 19 // MISO pin for MAX31865

// SHT40B i2c addr 0x45

#endif // PIN_CONFIG_H

