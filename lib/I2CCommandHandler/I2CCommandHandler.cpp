#include "I2CCommandHandler.h"
#include "definitions.h"
#include "FileManager.h"
#include <WiFi.h>

I2CCommandHandler::I2CCommandHandler(
    volatile ExperimentState& expState,
    String& currentExpName,
    String& currentExpFilename,
    unsigned long& expStartTime,
    bool& isFirstWrite,
    std::vector<int>& heaterSettings,
    int& heatingtime,
    CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE>& bufferA,
    CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE>& bufferB,
    std::vector<LogEntry>& debugLogs,
    WiFiCredentials& wifiCredentials,
    bool& wifiCredentialsChanged,
    bool& networkAvailable,
    AddLogFn addLog,
    GetLogStatsFn getLogStats,
    FileManager& fileManager)
    : _expState(expState),
      _currentExpName(currentExpName),
      _currentExpFilename(currentExpFilename),
      _expStartTime(expStartTime),
      _isFirstWrite(isFirstWrite),
      _heaterSettings(heaterSettings),
      _heatingtime(heatingtime),
      _bufferA(bufferA),
      _bufferB(bufferB),
      _debugLogs(debugLogs),
      _wifiCredentials(wifiCredentials),
      _wifiCredentialsChanged(wifiCredentialsChanged),
      _networkAvailable(networkAvailable),
      _addDebugLog(addLog),
      _getLogStats(getLogStats),
      _fileManager(fileManager)
{
    // Constructor initialization
}

void I2CCommandHandler::setup() {
    // Initialization can go here if needed
}

void I2CCommandHandler::setWiFiStatus(uint8_t status) {
    _lastWifiStatus = status;
}

void I2CCommandHandler::receiveEvent(int byteCount)
{
    if (Wire.available() < 1)
    {
        _addDebugLog("I2C: receiveEvent called but no data available");
        return;
    }

    uint8_t command = Wire.read();
    _lastReceivedCommand = command;
    
    if (command != CMD_READ_DATA && command != CMD_READ_LOGS)
    {
        _addDebugLog("I2C cmd received: 0x" + String(command, HEX));
    }

    switch (command)
    {
    case CMD_SET_HEATING:
        if (Wire.available() >= 1)
        {
            uint8_t marker = Wire.read();
            
            if (marker == 0xFF) // Heater settings list
            {
                String heaterCommand = "";
                while (Wire.available())
                {
                    char c = Wire.read();
                    heaterCommand += c;
                }
                
                if (heaterCommand.startsWith("heaterSettings:"))
                {
                    _heaterSettings.clear();
                    String settingsStr = heaterCommand.substring(15);
                    int startIndex = 0;
                    int endIndex;
                    while ((endIndex = settingsStr.indexOf(',', startIndex)) != -1)
                    {
                        _heaterSettings.push_back(settingsStr.substring(startIndex, endIndex).toInt());
                        startIndex = endIndex + 1;
                    }
                    _heaterSettings.push_back(settingsStr.substring(startIndex).toInt());
                    
                    Serial.print("I2C: Heater settings updated: ");
                    for (int setting : _heaterSettings)
                    {
                        Serial.print(setting);
                        Serial.print(" ");
                    }
                    Serial.println();
                    _addDebugLog("Heater settings updated via I2C");
                }
            }
            else if (marker == 0xFE) // Heating time
            {
                String timeCommand = "";
                while (Wire.available())
                {
                    char c = Wire.read();
                    timeCommand += c;
                }
                
                if (timeCommand.startsWith("heatingtime:"))
                {
                    _heatingtime = timeCommand.substring(12).toInt();
                    Serial.print("I2C: Heating time set to: ");
                    Serial.println(_heatingtime);
                    _addDebugLog("Heating time set to: " + String(_heatingtime) + "ms");
                }
            }
            else if (Wire.available() >= 2) // Original single PWM format
            {
                uint8_t pwm = marker;
                uint16_t duration = (Wire.read() << 8) | Wire.read();
                _heaterSettings.clear();
                _heaterSettings.push_back(pwm);
                _heatingtime = duration;
                
                Serial.print("I2C: Single PWM set to: ");
                Serial.print(pwm);
                Serial.print(", Duration: ");
                Serial.println(duration);
                _addDebugLog("Single PWM set: " + String(pwm) + ", Duration: " + String(duration));
            }
        }
        break;

    case CMD_START_EXP:
        if (Wire.available() > 0)
        {
            _currentExpName = "";
            while (Wire.available())
            {
                char c = Wire.read();
                _currentExpName += c;
            }
            _currentExpFilename = "/" + _currentExpName + ".csv";
            _isFirstWrite = true;
            _expState = EXP_RUNNING;
            _expStartTime = millis();
            Serial.print("I2C: Starting experiment: ");
            Serial.print(_currentExpName);
            Serial.print(" -> ");
            Serial.println(_currentExpFilename);
            _addDebugLog("Experiment started: " + _currentExpName);
        }
        break;

    case CMD_STOP_EXP:
        _expState = EXP_IDLE;
        
        // Write any remaining data in buffers when experiment stops using FileManager
        if (!_bufferA.isEmpty())
        {
            _fileManager.createFileWriteTask(&_bufferA, _currentExpFilename, false);
        }
        if (!_bufferB.isEmpty())
        {
            _fileManager.createFileWriteTask(&_bufferB, _currentExpFilename, false);
        }
        
        Serial.print("I2C: Experiment stopped. Final data saved to: ");
        Serial.println(_currentExpFilename);
        _addDebugLog("Experiment stopped, data saved to: " + _currentExpFilename);
        break;

    case CMD_SET_WIFI:
        if (Wire.available() > 0)
        {
            String wifiData = "";
            while (Wire.available())
            {
                char c = Wire.read();
                wifiData += c;
            }
            
            // Parse WiFi credentials in format: "SSID|PASSWORD"
            int separatorIndex = wifiData.indexOf('|');
            if (separatorIndex > 0 && separatorIndex < wifiData.length() - 1)
            {
                String newSSID = wifiData.substring(0, separatorIndex);
                String newPassword = wifiData.substring(separatorIndex + 1);
                
                if (newSSID.length() > 0 && newPassword.length() > 0)
                {
                    _wifiCredentials = WiFiCredentials(newSSID, newPassword);
                    _wifiCredentialsChanged = true;
                    _lastWifiStatus = 1; // Set to "connecting" status
                    
                    Serial.printf("I2C: WiFi credentials updated - SSID: %s\n", newSSID.c_str());
                    _addDebugLog("WiFi credentials updated via I2C: " + newSSID);
                }
                else
                {
                    Serial.println("I2C: Invalid WiFi credentials format");
                    _addDebugLog("Invalid WiFi credentials received via I2C");
                    _lastWifiStatus = 3; // Set to "failed" status
                }
            }
            else
            {
                Serial.println("I2C: WiFi credentials must be in format: SSID|PASSWORD");
                _addDebugLog("Invalid WiFi format received via I2C");
                _lastWifiStatus = 3; // Set to "failed" status
            }
        }
        break;

    case CMD_WIFI_STATUS:
        // No need to store this command, will be handled immediately in requestEvent
        _addDebugLog("WiFi status requested via I2C");
        break;

    case CMD_GET_IP:
        // No need to store this command, will be handled immediately in requestEvent
        _addDebugLog("IP address requested via I2C");
        break;

    case CMD_READ_LOGS:
        _addDebugLog("Log read requested via I2C");
        break;

    case CMD_CLEAR_LOGS:
        _debugLogs.clear();
        _addDebugLog("Debug logs cleared via I2C");
        break;

    case CMD_LOG_STATS:
        Serial.println("I2C: Log statistics requested");
        break;
    }
}

void I2CCommandHandler::requestEvent()
{
    static bool logReadMode = false;
    static size_t logIndex = 0;
    
    unsigned long now = millis();
    _lastI2CActivity = now;
    _i2cRequestCount++;
    
    if (now - _i2cMonitorWindow > 1000)
    {
        if (_i2cRequestCount > 50)
        {
            _addDebugLog("I2C high freq: " + String(_i2cRequestCount) + " req/sec");
            delayMicroseconds(100);
        }
        _i2cRequestCount = 0;
        _i2cMonitorWindow = now;
    }
    
    if (_lastReceivedCommand == CMD_READ_LOGS)
    {
        logReadMode = true;
        logIndex = 0;
        _lastReceivedCommand = 0;
        requestEvent();
        return;
    }
    
    if (logReadMode)
    {
        uint8_t response[32] = {0};
        
        if (logIndex < _debugLogs.size())
        {
            LogEntry entry = _debugLogs[logIndex];
            
            uint8_t msgLen = min(entry.message.length(), (size_t)26);
            response[0] = msgLen;
            response[1] = (entry.timestamp >> 24) & 0xFF;
            response[2] = (entry.timestamp >> 16) & 0xFF;
            response[3] = (entry.timestamp >> 8) & 0xFF;
            response[4] = entry.timestamp & 0xFF;
            memcpy(&response[5], entry.message.c_str(), msgLen);
            
            logIndex++;
        }
        else
        {
            logReadMode = false;
            logIndex = 0;
            response[0] = 0;
        }
        
        Wire.write(response, 32);
        return;
    }

    if (_lastReceivedCommand == CMD_LOG_STATS)
    {
        uint8_t response[32] = {0};
        String stats = _getLogStats();
        
        uint8_t statsLen = min(stats.length(), (size_t)26);
        uint32_t totalEntries = _debugLogs.size();
        
        response[0] = statsLen;
        response[1] = (totalEntries >> 24) & 0xFF;
        response[2] = (totalEntries >> 16) & 0xFF;
        response[3] = (totalEntries >> 8) & 0xFF;
        response[4] = totalEntries & 0xFF;
        memcpy(&response[5], stats.c_str(), statsLen);
        
        _lastReceivedCommand = 0;
        Wire.write(response, 32);
        return;
    }
    
    if (_lastReceivedCommand == CMD_WIFI_STATUS)
    {
        uint8_t response[8] = {0};
        
        // Check if network is available (AP mode active)
        if (_networkAvailable)
        {
            // If we have a recent status from credential change, use it
            // Otherwise, check if AP is actually running
            if (_lastWifiStatus == 0)
            {
                // No recent credential change, check current AP status
                String apSSID = WiFi.softAPSSID();
                if (apSSID.length() > 0)
                {
                    response[0] = 2; // Success - AP is running
                }
                else
                {
                    response[0] = 0; // Unknown - AP should be running but isn't
                }
            }
            else
            {
                response[0] = _lastWifiStatus; // Use status from credential change
            }
        }
        else
        {
            response[0] = 0; // Unknown - network not available
        }
        
        // Current WiFi connection status
        // For AP mode, WiFi.status() might not be accurate, so also check AP status
        if (_networkAvailable && WiFi.softAPSSID().length() > 0)
        {
            response[1] = 3; // WL_CONNECTED equivalent for AP mode
        }
        else
        {
            response[1] = WiFi.status(); // Current WiFi connection status
        }
        
        _lastReceivedCommand = 0; // Reset command
        Wire.write(response, 8);
        return;
    }
    
    if (_lastReceivedCommand == CMD_GET_IP)
    {
        uint8_t response[16] = {0}; // Larger response for IP address

        // Check if network is available first
        if (_networkAvailable)
        {
            // Debug logging
            String apSSID = WiFi.softAPSSID();
            IPAddress softAPIP = WiFi.softAPIP();
            _addDebugLog("IP request: AP SSID='" + apSSID + "', AP IP=" + softAPIP.toString());

            // Check if softAP is active (AP mode) - this is the primary mode for this device
            if (apSSID.length() > 0 && softAPIP[0] != 0)
            {
                response[0] = 1; // Connected status (AP mode)
                response[1] = softAPIP[0]; // IP address bytes
                response[2] = softAPIP[1];
                response[3] = softAPIP[2];
                response[4] = softAPIP[3];

                // Include softAP SSID
                response[5] = min(apSSID.length(), (size_t)10); // SSID length (max 10 chars in response)
                for (int i = 0; i < min(apSSID.length(), (size_t)10); i++)
                {
                    response[6 + i] = apSSID[i];
                }
            }
            // Fallback: check for STA mode connection (though this device primarily operates in AP mode)
            else if (WiFi.status() == WL_CONNECTED)
            {
                IPAddress ip = WiFi.localIP();
                response[0] = 1; // Connected status
                response[1] = ip[0]; // IP address bytes
                response[2] = ip[1];
                response[3] = ip[2];
                response[4] = ip[3];

                // Also include SSID length and first part of SSID
                String ssid = WiFi.SSID();
                response[5] = min(ssid.length(), (size_t)10); // SSID length (max 10 chars in response)
                for (int i = 0; i < min(ssid.length(), (size_t)10); i++)
                {
                    response[6 + i] = ssid[i];
                }
            }
            else
            {
                response[0] = 0; // Not connected status
            }
        }
        else
        {
            // Network not available
            response[0] = 0; // Not connected status
            _addDebugLog("IP request: Network not available");
        }

        _lastReceivedCommand = 0; // Reset command
        Wire.write(response, 16);
        return;
    }
    
    // Explicit handling for CMD_READ_DATA
    if (_lastReceivedCommand == CMD_READ_DATA)
    {
        uint8_t response[8] = {0};
        
        // Only return data if experiment is running and buffer has data
        if (_expState == EXP_RUNNING && !_bufferA.isEmpty())
        {
            SingleChannel data = _bufferA.first();
            response[0] = (data.setting >> 8) & 0xFF;
            response[1] = data.setting & 0xFF;
            response[2] = (data.timestamp >> 24) & 0xFF;
            response[3] = (data.timestamp >> 16) & 0xFF;
            response[4] = (data.timestamp >> 8) & 0xFF;
            response[5] = data.timestamp & 0xFF;
            response[6] = (data.channel_0 >> 8) & 0xFF;
            response[7] = data.channel_0 & 0xFF;
        }
        // Otherwise return all zeros to indicate no data available
        
        _lastReceivedCommand = 0; // Reset command
        Wire.write(response, 8);
        return;
    }
    
    // Default fallback for any other request (shouldn't normally reach here)
    uint8_t response[8] = {0};
    Wire.write(response, 8);
} 