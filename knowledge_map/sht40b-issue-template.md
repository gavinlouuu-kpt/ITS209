# GitHub Issue: Implement SHT40B Temperature and Humidity Sensor Test

**Use this template to create the GitHub issue manually via the web interface, or run `.\scripts\create-sht40b-issue.ps1` after installing GitHub CLI.**

## Issue Title
```
Implement SHT40B Temperature and Humidity Sensor Test
```

## Issue Body
```markdown
## Overview
Implement test case for SHT40B temperature and humidity sensor readout on the master device (ITS209 PCB). The sensor is directly connected to the master device and communicates via I2C.

## Hardware Configuration
- **Sensor**: SHT40B (SHT40-AD1B-R2) from schematic SCH_Main Board_2025-11-25.pdf
- **I2C Address**: 0x44 (default)
- **Location**: Master device (ITS209 PCB)
- **Slave Device**: async_ads PCB at I2C address 0x08 (for communication testing)

## Test Requirements

### Test Cases to Implement
1. **Sensor Initialization Test**
   - Verify sensor is detected and initialized correctly
   - Check sensor precision settings

2. **Temperature Reading Test**
   - Validate temperature readings are within valid range (-40°C to 125°C)
   - Check for reasonable operating range (0-60°C for typical use)
   - Verify readings are not NaN

3. **Humidity Reading Test**
   - Validate humidity readings are within valid range (0-100% RH)
   - Verify readings are not NaN

4. **Sensor Precision Test**
   - Test high precision mode
   - Verify readings remain valid in different precision modes

5. **Slave Device Communication Test**
   - Verify I2C communication with async_ads slave device (address 0x08)
   - Ensure master device can communicate with slave while sensor is active

6. **Reading Consistency Test**
   - Multiple consecutive readings should be consistent
   - Temperature variance < 2°C
   - Humidity variance < 5% RH

## Implementation Details

### Files Created
- `test/test_sht40b/test_main.cpp` - Unity test case for SHT40B sensor
- `platformio.ini` - Updated with Adafruit SHT4x Library dependency

### Library Dependency
- `adafruit/Adafruit SHT4x Library@^2.0.0`

### I2C Configuration
- SDA Pin: 21
- SCL Pin: 22
- I2C Frequency: 100kHz (to match slave device)

## Acceptance Criteria
- [x] SHT4x library added to platformio.ini
- [x] Test file created following existing test pattern (test_temperature)
- [ ] All test cases pass successfully
- [ ] Sensor readings are within expected ranges
- [ ] I2C communication with slave device verified
- [ ] Test integrates with Unity test framework

## Related
- Test follows pattern from `test/test_temperature/test_main.cpp`
- Master device (ITS209) contains SHT40B sensor
- Slave device (async_ads) communication tested for integration
```

## Labels
- `testing`
- `enhancement`
- `hardware`

## Instructions to Create Issue

### Option 1: Using GitHub Web Interface
1. Go to https://github.com/gavinlouuu-kpt/ITS209/issues
2. Click "New issue"
3. Copy the Issue Title above
4. Copy the Issue Body above
5. Add the labels listed above
6. Click "Submit new issue"
7. Add the issue to your GitHub Project board

### Option 2: Using GitHub CLI (after installation)
1. Install GitHub CLI: `winget install --id GitHub.cli`
2. Authenticate: `gh auth login`
3. Run: `.\scripts\create-sht40b-issue.ps1`


