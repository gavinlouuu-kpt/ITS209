# Script to create GitHub issue for I2C Master-Slave test
# Requires GitHub CLI (gh) to be installed and authenticated
# Run: .\scripts\create-i2c-slave-test-issue.ps1

# Refresh PATH to include newly installed programs
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

$issueTitle = "Implement I2C Master-Slave Communication Test"
$issueBody = @"
## Overview
Implement Unity test case for I2C communication between master device (ITS209) and slave device (async_ads) using default I2C pins.

## Hardware Configuration
- **Master Device (ITS209)**:
  - Default I2C pins: SDA=21, SCL=22 (ESP32 default)
  - I2C frequency: 100kHz
  - Uses Wire.begin() with default pins

- **Slave Device (async_ads)**:
  - I2C address: 0x08
  - I2C pins: SDA=25, SCL=26
  - I2C frequency: 100kHz
  - Protocol: Command-based I2C communication

## Test Requirements

### Test Cases Implemented
1. **I2C Initialization Test**
   - Verify I2C bus initialization with default pins
   - Set I2C frequency to 100kHz

2. **Slave Device Detection Test**
   - Detect slave device at address 0x08
   - Verify I2C communication is established
   - Handle error codes appropriately

3. **I2C Bus Integrity Test**
   - Scan I2C bus for devices
   - Verify at least one device is detected

4. **Read Data Command Test**
   - Test CMD_READ_DATA (0x04) command
   - Parse 8-byte response: [Setting_High] [Setting_Low] [Timestamp_3] [Timestamp_2] [Timestamp_1] [Timestamp_0] [Raw_Value_High] [Raw_Value_Low]
   - Verify data validity (not all zeros or 0xFF)

5. **Stop Experiment Command Test**
   - Test CMD_STOP_EXP (0x03) command
   - Verify command transmission success

6. **WiFi Status Command Test**
   - Test CMD_WIFI_STATUS (0x06) command
   - Handle response appropriately

7. **Multiple Read Consistency Test**
   - Read data multiple times
   - Verify timestamp consistency between reads
   - Ensure data is updating correctly

## Implementation Details

### Files Created
- `test/test_i2c_slave/test_main.cpp` - Unity test case for I2C master-slave communication

### I2C Protocol Commands Tested
- `CMD_READ_DATA` (0x04) - Request sensor data
- `CMD_STOP_EXP` (0x03) - Stop experiment
- `CMD_WIFI_STATUS` (0x06) - Get WiFi status

### Data Response Format
When master requests data using CMD_READ_DATA, slave responds with 8 bytes:
- **Setting** (16-bit): Current heater PWM setting
- **Timestamp** (32-bit): Timestamp in milliseconds since boot
- **Raw_Value** (16-bit): Raw ADC reading from ADS1115

## Acceptance Criteria
- [x] Test file created following existing test pattern
- [x] I2C initialization with default pins (SDA=21, SCL=22)
- [x] Slave device detection at address 0x08
- [x] CMD_READ_DATA command implementation
- [x] Data parsing and validation
- [x] Multiple command tests
- [x] Data consistency verification
- [ ] All test cases pass successfully
- [ ] Test integrates with Unity test framework

## Related
- Test follows pattern from `test/test_temperature/test_main.cpp`
- Uses async_ads I2C protocol documented in `C:/Users/gavin/Developer/async_ads/I2C_Communication_Protocol.md`
- Master device (ITS209) communicates with slave device (async_ads)
"@

$labels = "testing,enhancement"

Write-Host "Creating GitHub issue: $issueTitle" -ForegroundColor Cyan

# Check if GitHub CLI is installed
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Host "Error: GitHub CLI (gh) is not installed." -ForegroundColor Red
    Write-Host "Install it with: winget install --id GitHub.cli" -ForegroundColor Yellow
    Write-Host "Or visit: https://cli.github.com/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Once installed, authenticate with: gh auth login" -ForegroundColor Yellow
    exit 1
}

# Check if authenticated
$authStatus = gh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: GitHub CLI is not authenticated." -ForegroundColor Red
    Write-Host "Run: gh auth login" -ForegroundColor Yellow
    exit 1
}

# Create the issue first (without labels to avoid failure if labels don't exist)
Write-Host "Creating issue..." -ForegroundColor Cyan
$issueOutput = gh issue create --repo gavinlouuu-kpt/ITS209 --title $issueTitle --body $issueBody 2>&1

if ($LASTEXITCODE -eq 0) {
    # Extract issue number from output (format: "https://github.com/.../issues/123")
    $issueNumber = $issueOutput | Select-String -Pattern 'issues/(\d+)' | ForEach-Object { $_.Matches.Groups[1].Value }
    
    Write-Host "Issue created successfully! Issue #$issueNumber" -ForegroundColor Green
    
    # Try to add labels (ignore errors if labels don't exist)
    if ($issueNumber) {
        Write-Host "Attempting to add labels..." -ForegroundColor Cyan
        $labelArray = $labels -split ','
        foreach ($label in $labelArray) {
            $label = $label.Trim()
            if ($label) {
                gh issue edit --repo gavinlouuu-kpt/ITS209 $issueNumber --add-label $label 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "  Added label: $label" -ForegroundColor Gray
                } else {
                    Write-Host "  Label '$label' not found in repository (skipping)" -ForegroundColor Yellow
                }
            }
        }
    }
    
    Write-Host ""
    Write-Host "Issue URL: $issueOutput" -ForegroundColor Cyan
    Write-Host "You can view all issues at: https://github.com/gavinlouuu-kpt/ITS209/issues" -ForegroundColor Cyan
} else {
    Write-Host "Failed to create issue. Error:" -ForegroundColor Red
    Write-Host $issueOutput -ForegroundColor Red
}


