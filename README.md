# ITS209 Project

ESP32-based temperature monitoring system using Adafruit MAX31865 RTD sensor.

## Overview

This project implements temperature sensing and monitoring using the Adafruit MAX31865 RTD (Resistance Temperature Detector) sensor on an ESP-WROVER-KIT development board.

## Hardware

- **Board**: ESP-WROVER-KIT (ESP32)
- **Sensor**: Adafruit MAX31865 RTD Sensor
- **RTD Type**: PT100 (100Ω nominal resistance)

## Software

- **Platform**: PlatformIO
- **Framework**: Arduino
- **Libraries**:
  - Adafruit MAX31865 library (v1.6.2+)
- **Test Framework**: Unity

## Project Structure

```
ITS209/
├── src/              # Source code
│   └── main.cpp      # Main application
├── test/             # Unit tests
│   └── test_temperature/
│       └── test_main.cpp
├── include/          # Header files
├── lib/              # Project-specific libraries
├── scripts/          # Helper scripts
└── knowledge_map/    # Project documentation
```

## Getting Started

### Prerequisites

- PlatformIO IDE or PlatformIO CLI
- ESP-WROVER-KIT board
- Adafruit MAX31865 sensor connected via SPI

### Building

```bash
pio run
```

### Uploading

```bash
pio run --target upload
```

### Testing

```bash
pio test
```

## Task Management

This project uses **GitHub Projects** for task management and issue tracking.

### Quick Links

- **Issues**: [View all issues](https://github.com/gavinlouuu-kpt/ITS209/issues)
- **Projects**: [View project board](https://github.com/gavinlouuu-kpt/ITS209/projects)

### Creating Tasks

**Via GitHub Web UI:**
1. Go to [Issues](https://github.com/gavinlouuu-kpt/ITS209/issues)
2. Click "New issue"
3. Fill in details and add to your project board

**Via Command Line:**
```powershell
# Using the helper script
.\scripts\github-tasks.ps1 create "Task title" "Task description"

# Or using GitHub CLI directly
gh issue create --title "Task title" --body "Task description"
```

### Managing Tasks

- View tasks: `.\scripts\github-tasks.ps1 list`
- View specific task: `.\scripts\github-tasks.ps1 view <issue-number>`
- Add comment: `.\scripts\github-tasks.ps1 comment <issue-number> "Comment text"`
- Close task: `.\scripts\github-tasks.ps1 close <issue-number>`

For detailed workflows and best practices, see:
- [GitHub Projects Setup Guide](knowledge_map/github-projects-setup.md)
- [Task Management Workflow](knowledge_map/task-management.md)

## Documentation

Additional documentation is available in the `knowledge_map/` directory:

- `github-projects-setup.md` - Setting up GitHub Projects
- `task-management.md` - Task management workflows and best practices

## Development

### SPI Configuration

The MAX31865 sensor uses SPI communication. Default pin configuration:
- CS: Pin 5
- DI (MOSI): Pin 23
- DO (MISO): Pin 19
- CLK: Pin 18

### Sensor Configuration

- RTD Type: PT100
- Reference Resistor (Rref): 430.0Ω
- Nominal Resistance (Rnominal): 100.0Ω
- Wiring: 2-wire configuration

## License

[Add your license here]

## Contributing

1. Create an issue in GitHub Projects
2. Create a feature branch
3. Make your changes
4. Submit a pull request referencing the issue

