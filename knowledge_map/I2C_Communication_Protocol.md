# I2C Communication Protocol Documentation

## Overview

This document describes the I2C communication protocol between the master ESP32 device (`i2c_master_test.cpp`) and the slave ESP32 device (`main.cpp`) in the async_ads project. The protocol enables remote control of gas sensor experiments and data collection.

**Note**: File management (listing, reading, and deleting files) is now handled via FTP server rather than I2C commands. This provides better performance and reliability for large file transfers. The slave device runs an FTP server accessible for downloading experiment data files.

## Hardware Configuration

### Slave Device (main.cpp)
- **I2C Address**: `0x08`
- **I2C Bus**: Wire (I2C0)
- **SDA Pin**: 25
- **SCL Pin**: 26
- **Frequency**: 100kHz
- **Role**: Gas sensor controller with ADS1115 ADC

### Master Device (i2c_master_test.cpp)
- **I2C Bus**: Wire (default)
- **SDA Pin**: 21
- **SCL Pin**: 22
- **Frequency**: 100kHz
- **Role**: Remote controller and data reader

## Command Protocol

The I2C communication uses a command-based protocol with the following structure:

```
[Command Byte] [Data Bytes...]
```

### Available Commands

| Command | Value | Description |
|---------|--------|-------------|
| `CMD_SET_HEATING` | `0x01` | Configure heater parameters |
| `CMD_START_EXP` | `0x02` | Start experiment with name |
| `CMD_STOP_EXP` | `0x03` | Stop current experiment |
| `CMD_READ_DATA` | `0x04` | Request latest sensor data |

## Command Details

### 1. CMD_SET_HEATING (0x01)

This command has three different formats depending on the data being sent:

#### Format 1: Single PWM Value (Legacy)
```
[0x01] [PWM] [Duration_High] [Duration_Low]
```
- **PWM**: Single PWM value (0-255)
- **Duration**: Heating duration in milliseconds (16-bit, big-endian)

#### Format 2: Heater Settings List
```
[0x01] [0xFF] [heaterSettings:value1,value2,value3,...]
```
- **0xFF**: Special marker indicating heater settings list
- **String**: ASCII string in format "heaterSettings:100,120,140,160"

#### Format 3: Heating Time
```
[0x01] [0xFE] [heatingtime:duration]
```
- **0xFE**: Special marker indicating heating time setting
- **String**: ASCII string in format "heatingtime:5000"

### 2. CMD_START_EXP (0x02)
```
[0x02] [experiment_name...]
```
- **experiment_name**: ASCII string containing the experiment name
- Starts data recording with the specified experiment name
- All data for this experiment will be saved to a single file: `{experiment_name}.csv`

### 3. CMD_STOP_EXP (0x03)
```
[0x03]
```
- No additional data required
- Stops current experiment and saves remaining buffered data

### 4. CMD_READ_DATA (0x04)
```
[0x04]
```
- No additional data required
- Master should follow with `requestFrom()` to receive 8 bytes of data



## Data Response Formats

### Sensor Data Response (CMD_READ_DATA)

When the master requests data using `CMD_READ_DATA`, the slave responds with 8 bytes:

```
[Setting_High] [Setting_Low] [Timestamp_3] [Timestamp_2] [Timestamp_1] [Timestamp_0] [Raw_Value_High] [Raw_Value_Low]
```

### Response Fields
- **Setting** (16-bit): Current heater PWM setting
- **Timestamp** (32-bit): Timestamp in milliseconds since boot
- **Raw_Value** (16-bit): Raw ADC reading from ADS1115

### Data Conversion
The master can convert the raw ADC value to meaningful units:

```cpp
// Convert to voltage
float voltage = raw_value * 0.000125; // 0.125mV per bit

// Convert to resistance
float load_resistance = 10000.0; // 10kΩ
float input_voltage = 3.3; // 3.3V
float resistance = (load_resistance / voltage) * ((input_voltage - voltage) / voltage);
```



## Communication Flow Examples

### Example 1: Setting Multiple Heater Values

**Master sends:**
1. `[0x01] [0xFF] heaterSettings:100,120,140,160,180`
2. `[0x01] [0xFE] heatingtime:5000`

**Slave response:**
- Updates internal heater settings vector: `[100, 120, 140, 160, 180]`
- Sets heating time to 5000ms
- Prints confirmation to serial: "I2C: Heater settings updated: 100 120 140 160 180"

### Example 2: Running an Experiment

**Master sends:**
1. `[0x02] gas_test_ethanol` (Start experiment)
2. Periodic `[0x04]` followed by `requestFrom(8)` (Read data)
3. `[0x03]` (Stop experiment)

**Slave behavior:**
- Begins cycling through heater settings
- Records data to circular buffers using double-buffering
- Responds to data requests with latest readings
- Continuously appends data to single experiment CSV file
- Creates new file on first write, appends on subsequent writes

### Example 3: Reading Live Data

**Master code:**
```cpp
// Request data
Wire.beginTransmission(0x08);
Wire.write(CMD_READ_DATA);
Wire.endTransmission();

// Read response
uint8_t bytesReceived = Wire.requestFrom(0x08, 8);
if (bytesReceived == 8) {
    uint8_t response[8];
    for (int i = 0; i < 8; i++) {
        response[i] = Wire.read();
    }
    
    // Parse data
    uint16_t setting = (response[0] << 8) | response[1];
    uint32_t timestamp = (response[2] << 24) | (response[3] << 16) | 
                        (response[4] << 8) | response[5];
    int16_t raw_value = (response[6] << 8) | response[7];
}
```

## Error Handling

### I2C Transmission Errors
- **Error Code 0**: Success
- **Error Code 1**: Data too long for buffer
- **Error Code 2**: Received NACK on address
- **Error Code 3**: Received NACK on data
- **Error Code 4**: Other error

### Data Validation
- Empty data responses (all zeros) indicate no new data available
- Invalid experiment names are rejected
- Out-of-range PWM values are clamped to 0-255

## Slave Internal Behavior

### Experiment States
- **EXP_IDLE**: No experiment running, preview mode active
- **EXP_RUNNING**: Experiment active, data recording to buffers

### Buffer Management
- Uses dual circular buffers (A and B) with 500 sample capacity each
- Automatic buffer switching when one becomes full
- Background file writing using FreeRTOS tasks

### File Output Format
CSV files are saved to LittleFS with the following format:
```csv
setting,timestamp,raw_value,resistance
100,1234567,12345,1234.56
120,1234572,12456,1245.67
...
```

## Usage with I2C Master Test

The `i2c_master_test.cpp` provides an interactive menu for testing all protocol features:

1. **Set heating parameters**: Choose between single PWM or comma-separated list
2. **Start experiment**: Enter experiment name
3. **Stop experiment**: End current experiment
4. **Read data**: Get single data sample
5. **List files**: Browse files stored on slave's LittleFS
6. **Read file**: Download and display file content with CRC verification
7. **Show menu**: Display available options
8. **Continuous reading**: Monitor data stream

## Best Practices

1. **Timing**: Allow small delays (100ms) between rapid I2C commands
2. **Data Size**: Keep experiment names under 20 characters, filenames under 26 characters
3. **Error Checking**: Always check I2C transmission return codes
4. **CRC Verification**: Always verify CRC16 checksums for file transfers
5. **Buffer Management**: Monitor slave serial output for buffer status
6. **File System**: Ensure LittleFS has sufficient space before long experiments
7. **File Transfer**: Use progressive reading with status updates for large files

## Troubleshooting

### Common Issues
- **No response**: Check wiring and I2C addresses
- **Garbled data**: Verify clock speeds match (100kHz)
- **Missing data**: Ensure experiment is running before reading
- **File errors**: Check LittleFS initialization and free space

### Debug Output
Both devices provide serial debug output:
- **Slave**: Reports I2C commands received and buffer status
- **Master**: Shows transmission results and parsed data

## Future Enhancements

Potential protocol extensions:
- Status inquiry commands
- Real-time streaming mode
- Configuration persistence
- Multi-sensor support
- Compressed data formats

