// Hardware configuration defaults for tests and sketches.
// Override any of these via PlatformIO build_flags per environment.
#ifndef HARDWARE_CONFIG_H
#define HARDWARE_CONFIG_H

// I2C pins and clock
#ifndef I2C_SDA_PIN
#define I2C_SDA_PIN 21
#endif

#ifndef I2C_SCL_PIN
#define I2C_SCL_PIN 22
#endif

#ifndef WIRE_CLOCK_HZ
#define WIRE_CLOCK_HZ 100000
#endif

// SHT40 address (default 0x44; our board uses 0x45)
#ifndef SHT4X_ADDR
#define SHT4X_ADDR 0x45
#endif

// MAX31865 (software SPI) pins and constants
#ifndef MAX31865_CS
#define MAX31865_CS 5
#endif

#ifndef MAX31865_MOSI
#define MAX31865_MOSI 23
#endif

#ifndef MAX31865_MISO
#define MAX31865_MISO 19
#endif

#ifndef MAX31865_SCLK
#define MAX31865_SCLK 18
#endif

#ifndef RNOMINAL
#define RNOMINAL 100.0
#endif

#ifndef RREF
#define RREF 430.0
#endif

// MAX31865 wiring mode (2-wire default). You can override to MAX31865_3WIRE/4WIRE.
#ifndef MAX31865_WIRING
#define MAX31865_WIRING MAX31865_2WIRE
#endif

// Optional actuator pins (not required for basic sensor tests)
#ifndef SOL2_PIN
#define SOL2_PIN 25
#endif
#ifndef SOL1_PIN
#define SOL1_PIN 26
#endif
#ifndef HEAT_PIN
#define HEAT_PIN 27
#endif
#ifndef PUMP_PIN
#define PUMP_PIN 13
#endif
#ifndef HEAT_PWM_CHANNEL
#define HEAT_PWM_CHANNEL 0
#endif
#ifndef PUMP_PWM_CHANNEL
#define PUMP_PWM_CHANNEL 1
#endif

#endif // HARDWARE_CONFIG_H


