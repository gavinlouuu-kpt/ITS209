#ifndef I2C_COMMAND_HANDLER_H
#define I2C_COMMAND_HANDLER_H

#include <Arduino.h>
#include <Wire.h>
#include <vector>
#include "CircularBuffer.hpp"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "definitions.h"

// Forward declaration
class FileManager;

// I2C command definitions
#define CMD_SET_HEATING 0x01
#define CMD_START_EXP 0x02
#define CMD_STOP_EXP 0x03
#define CMD_READ_DATA 0x04
#define CMD_SET_WIFI 0x05
#define CMD_WIFI_STATUS 0x06
#define CMD_GET_IP 0x09
#define CMD_READ_LOGS 0x07
#define CMD_CLEAR_LOGS 0x08
#define CMD_LOG_STATS 0x0A

// Callback function types for dependency injection
using AddLogFn = void (*)(const String&);
using GetLogStatsFn = String (*)();
using GetCurrentSensorReadingFn = int16_t (*)(); // Returns current sensor reading from ADS1115

class I2CCommandHandler {
public:
    I2CCommandHandler(
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
        GetCurrentSensorReadingFn getCurrentSensorReading,
        FileManager& fileManager
    );

    void setup();
    void receiveEvent(int byteCount);
    void requestEvent();
    void setWiFiStatus(uint8_t status); // Method to update WiFi status from main

private:
    // References to main sketch's variables
    volatile ExperimentState& _expState;
    String& _currentExpName;
    String& _currentExpFilename;
    unsigned long& _expStartTime;
    bool& _isFirstWrite;
    std::vector<int>& _heaterSettings;
    int& _heatingtime;
    CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE>& _bufferA;
    CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE>& _bufferB;
    std::vector<LogEntry>& _debugLogs;
    WiFiCredentials& _wifiCredentials;
    bool& _wifiCredentialsChanged;
    bool& _networkAvailable;

    // Callbacks
    AddLogFn _addDebugLog;
    GetLogStatsFn _getLogStats;
    GetCurrentSensorReadingFn _getCurrentSensorReading;
    
    // File manager reference
    FileManager& _fileManager;

    // Internal state
    volatile uint8_t _lastReceivedCommand = 0;
    
    // WiFi status tracking
    volatile uint8_t _lastWifiStatus = 0; // 0=unknown, 1=connecting, 2=success, 3=failed
    
    // I2C performance monitoring
    unsigned long _lastI2CActivity = 0;
    unsigned int _i2cRequestCount = 0;
    unsigned long _i2cMonitorWindow = 0;
};

#endif // I2C_COMMAND_HANDLER_H 