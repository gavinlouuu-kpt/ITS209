#include <Arduino.h>
#include <Adafruit_ADS1X15.h>
#include <CircularBuffer.hpp>
#include <vector>
#include <LittleFS.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <Wire.h> // Include I2C library
#include <WiFi.h>
#include "ESP-FTP-Server-Lib.h"
#include "definitions.h"
#include "I2CCommandHandler.h"
#include "FileManager.h"
#include "storage.h"

// Default AP credentials - can be changed via I2C (persisted in LittleFS)
#define DEFAULT_WIFI_SSID "its209"
#define DEFAULT_WIFI_PASSWORD "its209_23"

// Function declarations
void saveWiFiCredentials();
void loadWiFiCredentials();
bool connectToWiFi(const String &ssid, const String &password);
void checkWiFiConnection();
void processWiFiCredentialChange();
String getNetworkStatus();
bool initializeNetworking();

// FTP credentials
#define FTP_USER "ftp"
#define FTP_PASSWORD "ftp"

#define I2C_SLAVE_SDA 18      // SDA pin for I2C slave (Wire)
#define I2C_SLAVE_SCL 23      // SCL pin for I2C slave (Wire)
#define I2C_SLAVE_FREQ 100000 // I2C frequency for slave (100kHz)

// I2C pin definitions for ADS1115 (using I2C1)
#define SDA_PIN_ADS 21 // SDA pin for ADS1115
#define SCL_PIN_ADS 22 // SCL pin for ADS1115

Adafruit_ADS1115 ads;            /* Use this for the 16-bit version */
const int PWM_Heater = 19;       // PWM pin for heater control
const int I2C_SLAVE_ADDR = 0x08; // I2C slave address

// FTP Server instance
FTPServer ftp;

// Experiment control
volatile ExperimentState expState = EXP_IDLE;
String currentExpName = "";
String currentExpFilename = "";
unsigned long expStartTime = 0;
bool isFirstWrite = true;

// WiFi configuration
WiFiCredentials wifiCredentials(DEFAULT_WIFI_SSID, DEFAULT_WIFI_PASSWORD);
bool wifiCredentialsChanged = false;
bool networkAvailable = false; // Track if networking is available

const int PWM_CHANNEL = 0;    // PWM channel
const int PWM_FREQ = 10000;   // PWM frequency
const int PWM_RESOLUTION = 8; // 8-bit resolution

// Pin connected to the ALERT/RDY signal for new sample notification.
constexpr int READY_PIN = 3;

// Resistance calculation constants
const float load_resistance = 10000.0; // Load resistance in ohms
const float input_voltage = 3.3;       // Input voltage in volts

// This is required on ESP32 to put the ISR in IRAM. Define as
// empty for other platforms. Be careful - other platforms may have
// other requirements.
#ifndef IRAM_ATTR
#define IRAM_ATTR
#endif

volatile bool new_data = false;
void IRAM_ATTR NewDataReadyISR()
{
    new_data = true;
}

// Function to convert ADS reading to voltage
float ads_to_voltage(int16_t ads0)
{
    return ads0 * 0.000125; // Convert to voltage in volts
}

// Global variables that need to be declared before functions
std::vector<int> heaterSettings = {
    100,
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    225,
    205,
    185,
    165,
    145,
    125}; // Example heater settings

int heatingtime = 5; // Heating time in milliseconds

// Buffer and task management variables
CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE> bufferA;
CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE> bufferB;
volatile bool useBufferA = true;    // Flag to determine which buffer is actively recording
unsigned long fileCounter = 0;      // Counter for unique file names
unsigned int bufferChangeCount = 0; // Counter for buffer changes (log every 10th)

// Debug logging system
std::vector<LogEntry> debugLogs;         // Store all log entries (expandable)
constexpr size_t MAX_LOG_ENTRIES = 1000; // Optional limit to prevent memory overflow

// Function to add debug log entry
void addDebugLog(const String &message)
{
    LogEntry entry;
    entry.timestamp = millis();
    entry.message = message;

    // Optional: Remove oldest entries if we exceed maximum
    if (debugLogs.size() >= MAX_LOG_ENTRIES)
    {
        // Remove the first 100 entries to make room
        debugLogs.erase(debugLogs.begin(), debugLogs.begin() + 100);
        Serial.println("LOG: Removed oldest 100 log entries to prevent memory overflow");
    }

    debugLogs.push_back(entry);

    // Also print to serial for immediate debugging
    Serial.print("[");
    Serial.print(entry.timestamp);
    Serial.print("] LOG: ");
    Serial.println(entry.message);
}

// Function to get log statistics
String getLogStats()
{
    size_t totalEntries = debugLogs.size();
    size_t memoryUsed = totalEntries * (sizeof(LogEntry) + 20); // Approximate memory usage

    return "Logs: " + String(totalEntries) + " entries, ~" + String(memoryUsed) + " bytes";
}

// Function to add debug log entry with priority (to reduce spam during experiments)
void addDebugLogPriority(const String &message, bool highPriority = false)
{
    // During experiments, only log high priority messages
    if (expState == EXP_RUNNING && !highPriority)
    {
        return; // Skip low priority logs during experiments
    }
    addDebugLog(message);
}

// File Manager Instance
FileManager fileManager(addDebugLog);

// I2C Command Handler Instance
I2CCommandHandler i2cHandler(
    expState,
    currentExpName,
    currentExpFilename,
    expStartTime,
    isFirstWrite,
    heaterSettings,
    heatingtime,
    bufferA,
    bufferB,
    debugLogs,
    wifiCredentials,
    wifiCredentialsChanged,
    networkAvailable,
    addDebugLog,
    getLogStats,
    fileManager);

// Wrapper for receive event
void receiveEventWrapper(int byteCount)
{
    i2cHandler.receiveEvent(byteCount);
}

// Wrapper for request event
void requestEventWrapper()
{
    i2cHandler.requestEvent();
}

// Function to connect to WiFi with given credentials
// (STA connect helper left intentionally in place in case it is needed later)
bool connectToWiFi(const String &ssid, const String &password)
{
    Serial.printf("Connecting to WiFi (STA): %s\n", ssid.c_str());
    WiFi.begin(ssid.c_str(), password.c_str());

    unsigned long startTime = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - startTime < 15000)
    {
        delay(500);
        Serial.print(".");
    }

    if (WiFi.status() == WL_CONNECTED)
    {
        Serial.println("\nWiFi connected successfully (STA)!");
        Serial.print("IP address: ");
        Serial.println(WiFi.localIP());
        return true;
    }
    else
    {
        Serial.printf("\nFailed to connect to WiFi (STA): %s\n", ssid.c_str());
        return false;
    }
}

// Function to immediately process WiFi credential changes (called from I2C handler)
void processWiFiCredentialChange()
{
    if (!wifiCredentialsChanged)
        return;

    Serial.println("Processing AP credential change immediately...");
    addDebugLog("Processing immediate AP credential change");

    // Indicate connecting
    i2cHandler.setWiFiStatus(1); // 1 = connecting

    // Stop advertising previous AP (if any)
    networkAvailable = false;
    // Best-effort disconnect of softAP (true = erase config)
    WiFi.softAPdisconnect(true);
    delay(500);

    // Start AP with new credentials
    // Configure a fixed SoftAP IP before starting
    if (!WiFi.softAPConfig(IPAddress(192, 168, 4, 1), IPAddress(192, 168, 4, 1), IPAddress(255, 255, 255, 0)))
    {
        Serial.println("Failed to set SoftAP IP configuration");
    }
    bool apStarted = WiFi.softAP(wifiCredentials.ssid.c_str(), wifiCredentials.password.c_str());
    if (apStarted)
    {
        // Start or reinitialize FTP server on LittleFS
        ftp.addUser(FTP_USER, FTP_PASSWORD);
        ftp.addFilesystem("LittleFS", &LittleFS);
        ftp.begin();

        addDebugLog("SoftAP started with SSID: " + wifiCredentials.ssid);
        // Persistence is disabled; do not save credentials
        Serial.println("✓ AP configuration successful - credentials saved");
        i2cHandler.setWiFiStatus(2); // 2 = success
        networkAvailable = true;
    }
    else
    {
        addDebugLog("Failed to start SoftAP with new credentials");
        Serial.println("✗ Failed to start AP with provided credentials");
        i2cHandler.setWiFiStatus(3); // 3 = failed
    }

    wifiCredentialsChanged = false;
}

// Function to check WiFi connection and reconnect if needed
void checkWiFiConnection()
{
    // For AP-only mode we don't attempt STA reconnects.
    // Only handle immediate credential changes (I2C).
    if (wifiCredentialsChanged)
    {
        processWiFiCredentialChange();
    }
}

// Function to save WiFi credentials to persistent storage
void saveWiFiCredentials()
{
    // Persistence disabled: AP credentials are now fixed defaults and not stored.
    addDebugLog("AP credential persistence disabled; not saving to LittleFS");
}

// Function to load WiFi credentials from persistent storage
void loadWiFiCredentials()
{
    // We no longer persist AP credentials. Remove any legacy credential file to clean LittleFS,
    // and use compile-time defaults as the authoritative AP credentials.
    if (LittleFS.exists("/wifi_config.json"))
    {
        if (LittleFS.remove("/wifi_config.json"))
        {
            addDebugLog("Removed legacy AP credential file from LittleFS");
            Serial.println("Removed legacy AP credentials from LittleFS");
        }
        else
        {
            addDebugLog("Failed to remove legacy AP credential file from LittleFS");
        }
    }

    // Use built-in defaults
    wifiCredentials = WiFiCredentials(DEFAULT_WIFI_SSID, DEFAULT_WIFI_PASSWORD);
    addDebugLog("Using built-in AP credentials: " + wifiCredentials.ssid);
}

// Function to initialize networking (WiFi and FTP)
bool initializeNetworking()
{
    Serial.println("Initializing networking (SoftAP mode)...");

    // Load AP credentials from persistent storage (or defaults)
    loadWiFiCredentials();

    // Configure a fixed SoftAP IP before starting
    if (!WiFi.softAPConfig(IPAddress(192, 168, 4, 1), IPAddress(192, 168, 4, 1), IPAddress(255, 255, 255, 0)))
    {
        Serial.println("Failed to set SoftAP IP configuration");
    }

    // Start SoftAP with loaded credentials
    bool apStarted = WiFi.softAP(wifiCredentials.ssid.c_str(), wifiCredentials.password.c_str());
    if (!apStarted)
    {
        // Try defaults if starting with stored creds failed
        Serial.println("Failed to start AP with stored credentials, trying defaults...");
        wifiCredentials = WiFiCredentials(DEFAULT_WIFI_SSID, DEFAULT_WIFI_PASSWORD);
        // Re-apply fixed SoftAP IP before retry
        WiFi.softAPConfig(IPAddress(192, 168, 4, 1), IPAddress(192, 168, 4, 1), IPAddress(255, 255, 255, 0));
        apStarted = WiFi.softAP(wifiCredentials.ssid.c_str(), wifiCredentials.password.c_str());
        if (!apStarted)
        {
            Serial.println("Failed to start AP with default credentials");
            Serial.println("AP can be configured via I2C command");
            Serial.println("System will continue in standalone mode");
            networkAvailable = false;
            return false;
        }
    }

    // Start FTP server serving LittleFS
    ftp.addUser(FTP_USER, FTP_PASSWORD);
    ftp.addFilesystem("LittleFS", &LittleFS);
    ftp.begin();
    Serial.println("SoftAP started");
    Serial.print("AP SSID: ");
    Serial.println(wifiCredentials.ssid);
    Serial.print("AP IP: ");
    Serial.println(WiFi.softAPIP());
    Serial.println("FTP Server started");
    Serial.println("FTP credentials - User: " + String(FTP_USER) + ", Password: " + String(FTP_PASSWORD));

    networkAvailable = true;
    return true;
}

// Function to get WiFi and FTP status
String getNetworkStatus()
{
    String status = "AP: ";
    if (networkAvailable)
    {
        status += "Hosting " + wifiCredentials.ssid + " (" + WiFi.softAPIP().toString() + ")";
        status += ", FTP: Active";
    }
    else
    {
        status += "Standalone Mode (No Network)";
    }
    return status;
}

// Function to convert ADS reading to resistance
float ads_to_resistance(int16_t ads0)
{
    float voltage = ads_to_voltage(ads0);

    // Protect against division by zero and invalid voltage values
    if (voltage <= 0 || voltage >= input_voltage)
    {
        return 0; // Return 0 for invalid readings
    }

    // Calculate resistance using voltage divider formula
    // R1 = R_load * (V_in - V_out) / V_out
    float R1 = load_resistance * (input_voltage - voltage) / voltage;

    // Sanity check - return 0 for unrealistic resistance values
    if (R1 < 0 || R1 > 1e9) // Cap at 1GOhm
    {
        return 0;
    }

    return R1; // Return resistance in ohms
}

void muxRecord(const std::vector<int> &heaterSettings, int heatingTime)
{
    // Get pointer to currently active buffer
    CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE> *activeBuffer = useBufferA ? &bufferA : &bufferB;
    CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE> *inactiveBuffer = useBufferA ? &bufferB : &bufferA;

    // If the inactive buffer has data and no write is in progress, start writing using FileManager
    if (!inactiveBuffer->isEmpty() && !fileManager.isWriteTaskActive())
    {
        fileManager.createFileWriteTask(inactiveBuffer, currentExpFilename, false);
    }

    // Record data to active buffer
    for (int setting : heaterSettings)
    {
        ledcWrite(PWM_CHANNEL, setting);
        // non-blocking delay for heating time
        uint32_t heatDuration = millis();
        while (millis() - heatDuration < heatingTime)
        {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
        unsigned long timestamp = millis();              // Get current timestamp
        int16_t result = ads.getLastConversionResults(); // Get sensor reading

        SingleChannel data;         // Create a struct instance
        data.setting = setting;     // Assign the setting
        data.timestamp = timestamp; // Assign the timestamp
        data.channel_0 = result;    // Assign the sensor result

        activeBuffer->push(data); // Push structured data to active buffer

        // Check if active buffer is full, if so, switch buffers
        if (activeBuffer->isFull())
        {
            // Switch to the other buffer
            useBufferA = !useBufferA;
            Serial.print("Switched to buffer ");
            Serial.println(useBufferA ? "A" : "B");
            addDebugLogPriority("Switched to buffer " + String(useBufferA ? "A" : "B"), false);
            break; // Exit the recording loop to allow buffer switch
        }
    }
}

void muxPreview(const std::vector<int> &heaterSettings, int heatingTime)
{
    for (int setting : heaterSettings)
    {
        ledcWrite(PWM_CHANNEL, setting);
        // non-blocking delay for heating time
        uint32_t heatDuration = millis();
        while (millis() - heatDuration < heatingTime)
        {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
        unsigned long timestamp = millis();              // Get current timestamp
        int16_t result = ads.getLastConversionResults(); // Get sensor reading

        // Teleplot format: >variable_name:value|timestamp
        // Each heater setting gets its own variable with resistance value
        float resistance = ads_to_resistance(result);
        Serial.print(">setting_");
        Serial.print(setting);
        Serial.print("_ohms:");
        Serial.print(resistance);
        Serial.print("|");
        Serial.println(timestamp);
    }
}

void setup(void)
{
    Serial.begin(115200);

    // Initialize LittleFS with safe auto-recover
    if (!initLittleFS())
    {
        Serial.println("Failed to initialize LittleFS storage");
        while (1)
            ;
    }

    // Initialize File Manager
    if (!fileManager.initializeStorage(STORAGE_LITTLEFS))
    {
        Serial.println("Failed to initialize storage");
        while (1)
            ;
    }

    // Initialize networking (optional - system can work without it)
    networkAvailable = initializeNetworking();

    // Initialize I2C0 (Wire) as slave for receiving commands
    addDebugLog("Initializing I2C slave (Wire0): addr=0x" + String(I2C_SLAVE_ADDR, HEX) +
                ", SDA=" + String(I2C_SLAVE_SDA) +
                ", SCL=" + String(I2C_SLAVE_SCL) +
                ", FREQ=" + String(I2C_SLAVE_FREQ) + "Hz");
    Wire.begin(I2C_SLAVE_ADDR, I2C_SLAVE_SDA, I2C_SLAVE_SCL, I2C_SLAVE_FREQ); // SDA=25, SCL=26, 100kHz
    Wire.onReceive(receiveEventWrapper);
    Wire.onRequest(requestEventWrapper);
    Serial.println("I2C slave initialized on Wire (I2C0)");
    addDebugLog("I2C slave initialized on Wire (I2C0)");

    // Initialize I2C1 (Wire1) as master for ADS1115 communication
    Wire1.begin(SDA_PIN_ADS, SCL_PIN_ADS);
    Serial.println("I2C master initialized on Wire1 (I2C1) for ADS1115");

    // Setup PWM for heater control
    ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION);
    ledcAttachPin(PWM_Heater, PWM_CHANNEL);
    ledcWrite(PWM_CHANNEL, 200);

    // Initialize ADS1115 on Wire1 (I2C1)
    ads.setGain(GAIN_ONE);        // 1x gain   +/- 4.096V  1 bit = 2mV      0.125mV
    if (!ads.begin(0x48, &Wire1)) // Specify Wire1 for ADS1115
    {
        Serial.println("Failed to initialize ADS.");
        while (1)
            ;
    }
    Serial.println("ADS1115 initialized successfully on I2C1");

    pinMode(READY_PIN, INPUT);

    // Start continuous conversions.
    ads.startADCReading(ADS1X15_REG_CONFIG_MUX_SINGLE_0, /*continuous=*/true);

    // Initialize I2C Command handler
    i2cHandler.setup();

    Serial.println("System initialized with double buffering and FileManager");
    Serial.println("Buffer A and B ready for data recording");
    Serial.println("Storage: " + fileManager.getStorageInfo());
    Serial.println("Network: " + getNetworkStatus());
    Serial.println("Use 'help' command for available serial commands");

    if (networkAvailable)
    {
        Serial.println("✓ Full functionality available (Network + Storage)");
    }
    else
    {
        Serial.println("⚠ Running in standalone mode (Storage only)");
        Serial.println("  - Experiments will work normally");
        Serial.println("  - Data saved to local storage");
        Serial.println("  - Configure WiFi via I2C for network features");
    }

    addDebugLog("System initialized successfully with FileManager");
    addDebugLog("Double buffering system ready");
    addDebugLog("Continuous logging enabled (max " + String(MAX_LOG_ENTRIES) + " entries)");

    if (networkAvailable)
    {
        addDebugLog("WiFi connected: " + WiFi.localIP().toString());
        addDebugLog("FTP server started with LittleFS access");
    }
    else
    {
        addDebugLog("Running in standalone mode - no network");
    }
}

void loop(void)
{
    // Handle FTP server only if networking is available
    if (networkAvailable)
    {
        ftp.handle();
    }

    // Check for immediate WiFi credential changes (interrupt-like behavior)
    if (wifiCredentialsChanged)
    {
        processWiFiCredentialChange();
        // Update network availability after credential change
        if (WiFi.status() == WL_CONNECTED && !networkAvailable)
        {
            // WiFi just connected, initialize FTP
            ftp.addUser(FTP_USER, FTP_PASSWORD);
            ftp.addFilesystem("LittleFS", &LittleFS);
            ftp.begin();
            networkAvailable = true;
            addDebugLog("FTP server started after WiFi reconnection");
        }
    }

    // Periodically check WiFi connection (every 30 seconds) only if we had network before
    if (networkAvailable)
    {
        static unsigned long lastWiFiCheck = 0;
        if (millis() - lastWiFiCheck > 30000)
        {
            checkWiFiConnection();
            lastWiFiCheck = millis();
        }
    }

    // Handle experiment recording
    if (expState == EXP_RUNNING)
    {
        muxRecord(heaterSettings, heatingtime);
    }

    // Process serial commands only if no experiment is running
    if (Serial.available() && expState == EXP_IDLE)
    {
        String input = Serial.readStringUntil('\n');
        input.trim(); // Remove any whitespace/newlines

        if (input.startsWith("heatingtime:"))
        {
            heatingtime = input.substring(12).toInt();
            Serial.print("Heating time set to: ");
            Serial.println(heatingtime);
        }
        else if (input.startsWith("heaterSettings:"))
        {
            heaterSettings.clear();
            input = input.substring(15);
            int startIndex = 0;
            int endIndex;
            while ((endIndex = input.indexOf(',', startIndex)) != -1)
            {
                heaterSettings.push_back(input.substring(startIndex, endIndex).toInt());
                startIndex = endIndex + 1;
            }
            heaterSettings.push_back(input.substring(startIndex).toInt()); // Add last value
            Serial.print("Heater settings updated: ");
            for (int setting : heaterSettings)
            {
                Serial.print(setting);
                Serial.print(" ");
            }
            Serial.println();
        }
        else if (input == "network" || input == "ftp")
        {
            Serial.println(getNetworkStatus());
        }
        else if (input == "help")
        {
            Serial.println("Available commands:");
            Serial.println("  heatingtime:<value> - Set heating time");
            Serial.println("  heaterSettings:<csv> - Set heater settings");
            Serial.println("  network - Show network status");
            Serial.println("  ftp - Show FTP status");
            Serial.println("  help - Show this help");
        }
    }
    // Only run preview when not in an experiment
    if (expState == EXP_IDLE)
    {
        muxPreview(heaterSettings, heatingtime);
    }
}

