# FTP Server Setup Guide

## Overview
The ESP32 device now includes an integrated FTP server that allows easy access to data files stored in the LittleFS filesystem. This enables convenient data extraction and file management without needing to physically connect to the device.

## Configuration

### WiFi Setup
Before using the FTP server, you need to configure your WiFi credentials in `src/main.cpp`:

1. Open `src/main.cpp`
2. Find these lines near the top:
   ```cpp
   #define WIFI_SSID     "YourWiFiSSID"
   #define WIFI_PASSWORD "YourWiFiPassword"
   ```
3. Replace `"YourWiFiSSID"` with your actual WiFi network name
4. Replace `"YourWiFiPassword"` with your actual WiFi password

### FTP Credentials
The default FTP credentials are:
- **Username**: `ftp`
- **Password**: `ftp`

To change these, modify these lines in `src/main.cpp`:
```cpp
#define FTP_USER     "ftp"
#define FTP_PASSWORD "ftp"
```

## Usage

### Connecting to the FTP Server
1. Flash the updated firmware to your ESP32
2. Open the Serial Monitor to see the device's IP address
3. Use any FTP client to connect:
   - **Host**: The IP address shown in Serial Monitor
   - **Port**: 21 (default FTP port)
   - **Username**: `ftp` (or your custom username)
   - **Password**: `ftp` (or your custom password)

### Recommended FTP Clients
- **Windows**: WinSCP, FileZilla
- **macOS**: FileZilla, Cyberduck
- **Linux**: FileZilla, command-line `ftp`
- **Mobile**: AndFTP (Android), FTP Manager (iOS)

### Available Files
The FTP server provides access to the LittleFS filesystem where experimental data is stored. You'll find:
- Experiment data files (CSV format)
- Log files
- Configuration files

## Serial Commands
The device now supports additional serial commands:
- `network` - Show current WiFi and FTP status
- `ftp` - Same as network command
- `help` - Show all available commands

## Troubleshooting

### WiFi Connection Issues
- Verify WiFi credentials are correct
- Check that the ESP32 is within range of your WiFi network
- Monitor Serial output for connection status

### FTP Connection Issues
- Ensure the ESP32 is connected to WiFi (check Serial output)
- Verify you're using the correct IP address
- Check firewall settings on your computer
- Try different FTP clients

### Network Monitoring
The device automatically:
- Checks WiFi connection every 30 seconds
- Attempts to reconnect if connection is lost
- Logs network status to debug logs

## Security Notes
- Change default FTP credentials for production use
- Consider using this only on trusted networks
- The connection is not encrypted (standard FTP)

## Example Connection (Command Line)
```bash
ftp <ESP32_IP_ADDRESS>
# Enter username: ftp
# Enter password: ftp
# Use 'ls' to list files
# Use 'get filename' to download files
# Use 'quit' to exit
```

