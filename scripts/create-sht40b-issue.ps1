# Script to create GitHub issue for SHT40B sensor test
# Requires GitHub CLI (gh) to be installed and authenticated
# Run: .\scripts\create-sht40b-issue.ps1

# Refresh PATH to include newly installed programs
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

$issueTitle = "Implement SHT40B Temperature and Humidity Sensor Test"
$issueBody = @"
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
"@

$labels = "testing,enhancement,hardware"

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

