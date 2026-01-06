#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <Arduino.h>

// Global buffer size configuration - change this to adjust buffer size throughout the system
constexpr size_t CIRCULAR_BUFFER_SIZE = 2000;

// Experiment states
enum ExperimentState
{
    EXP_IDLE,
    EXP_RUNNING
};

// Struct definitions
struct SingleChannel
{
    int setting;
    unsigned long timestamp;
    int16_t channel_0;
};

// Debug logging system
struct LogEntry
{
    unsigned long timestamp;
    String message;
};

// WiFi configuration structure
struct WiFiCredentials
{
    String ssid;
    String password;
    bool isValid;
    
    WiFiCredentials() : ssid(""), password(""), isValid(false) {}
    WiFiCredentials(const String& s, const String& p) : ssid(s), password(p), isValid(true) {}
};

#endif // DEFINITIONS_H

