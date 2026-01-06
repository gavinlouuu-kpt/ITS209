#include <Arduino.h>
#include <Wire.h>
#include <map>

// I2C configuration - matches the slave device
#define I2C_SLAVE_ADDR 0x08 // Address of the slave ESP32
#define SDA_PIN 21          // SDA pin for I2C master (matches slave)
#define SCL_PIN 22          // SCL pin for I2C master (matches slave)

// Command definitions (must match slave)
#define CMD_SET_HEATING 0x01 // Set heater parameters
#define CMD_START_EXP 0x02   // Start experiment
#define CMD_STOP_EXP 0x03    // Stop experiment
#define CMD_READ_DATA 0x04   // Read data
#define CMD_SET_WIFI 0x05    // Set WiFi credentials
#define CMD_WIFI_STATUS 0x06 // Get WiFi status
#define CMD_GET_IP 0x09      // Get IP address
#define CMD_READ_LOGS 0x07   // Read debug logs
#define CMD_CLEAR_LOGS 0x08  // Clear debug logs



// Menu item structure
struct MenuItem {
    String description;
    String helpText;
    void (*function)();
};

// Function declarations
void printMenu();
void showHelp();
void setHeatingParameters();
void startExperiment();
void stopExperiment();
void readData();
void setWiFiCredentials();
void checkWiFiStatus();
void getIPAddress();
void readLogs();
void clearLogs();
void continuousDataReading();
void sendCommand(uint8_t command);
void sendCommandWithData(uint8_t command, uint8_t *data, size_t length);
void printDataResponse(uint8_t *response, size_t length);

// Menu hash table
std::map<String, MenuItem> menuItems = {
    {"1", {
        "Set heating parameters (single PWM or list)", 
        "Configure heater settings. Choose between single PWM value with duration or list of PWM values for sequential heating.",
        setHeatingParameters
    }},
    {"2", {
        "Start experiment", 
        "Begin data collection experiment with a custom name. The slave will start recording sensor data.",
        startExperiment
    }},
    {"3", {
        "Stop experiment", 
        "End the current experiment and save any remaining data to the slave's filesystem.",
        stopExperiment
    }},
    {"4", {
        "Read data (single)", 
        "Request one data point from the slave including setting, timestamp, and sensor reading with resistance calculation.",
        readData
    }},
    {"5", {
        "Set WiFi credentials", 
        "Configure WiFi SSID and password on the slave device. Credentials are saved to persistent storage.",
        setWiFiCredentials
    }},
    {"6", {
        "Check WiFi status", 
        "Check the current WiFi connection status on the slave device.",
        checkWiFiStatus
    }},
    {"7", {
        "Get IP address", 
        "Request the current IP address and network information from the slave device.",
        getIPAddress
    }},
    {"8", {
        "Read debug logs from slave", 
        "Retrieve all debug log entries from the slave with timestamps (complete history).",
        readLogs
    }},
    {"9", {
        "Clear debug logs on slave", 
        "Delete all debug log entries from the slave's memory.",
        clearLogs
    }},
    {"10", {
        "Continuous data reading", 
        "Start continuous polling of sensor data every second. Press any key to stop.",
        continuousDataReading
    }},
    {"menu", {
        "Show command menu", 
        "Display the main menu with all available commands.",
        printMenu
    }},
    {"help", {
        "Show detailed help", 
        "Display detailed descriptions of all available commands and their functions.",
        showHelp
    }}
};

void setup()
{
    Serial.begin(115200);
    while (!Serial)
    {
        delay(10); // Wait for Serial to be ready
    }

    // Initialize I2C as master
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(100000); // 100kHz I2C clock

    Serial.println("=== ESP32 I2C Master Test ===");
    Serial.println("Connected to slave at address 0x08");
    Serial.println();

    printMenu();
}

void loop()
{
    if (Serial.available())
    {
        String input = Serial.readStringUntil('\n');
        input.trim();

        // Find the function to call based on the input
        auto it = menuItems.find(input);
        if (it != menuItems.end())
        {
            it->second.function();
        }
        else if (input.length() > 0)
        {
            Serial.println("Invalid option. Type 'menu' for menu or 'help' for detailed descriptions.");
        }
    }

    delay(100);
}

void printMenu()
{
    Serial.println("=== I2C Master Commands ===");
    
    // Show numbered menu items first
    for (const auto& item : menuItems)
    {
        if (item.first != "help" && item.first != "menu") // Skip special commands in the main menu
        {
            Serial.println(item.first + ". " + item.second.description);
        }
    }
    
    Serial.println("\nSpecial commands:");
    Serial.println("  'menu' - Show this menu");
    Serial.println("  'help' - Show detailed descriptions of all commands");
    Serial.println("\nEnter option (1-10), 'menu', or 'help':");
}

void showHelp()
{
    Serial.println("\n=== Detailed Help ===");
    for (const auto& item : menuItems)
    {
        Serial.println("Command: " + item.first);
        Serial.println("Description: " + item.second.description);
        Serial.println("Help Text: " + item.second.helpText);
        Serial.println("----------------------------------------");
    }
    Serial.println();
}

void setHeatingParameters()
{
    Serial.println("\n=== Set Heating Parameters ===");
    Serial.println("Choose option:");
    Serial.println("1. Single PWM value");
    Serial.println("2. List of PWM values (comma-separated)");
    Serial.print("Enter option (1-2): ");

    while (!Serial.available())
    {
        delay(10);
    }
    String option = Serial.readStringUntil('\n');
    option.trim();

    if (option == "1")
    {
        // Original single PWM functionality
        Serial.print("Enter PWM value (0-255): ");
        while (!Serial.available())
        {
            delay(10);
        }
        uint8_t pwm = Serial.readStringUntil('\n').toInt();

        Serial.print("Enter heating duration (ms, 0-65535): ");
        while (!Serial.available())
        {
            delay(10);
        }
        uint16_t duration = Serial.readStringUntil('\n').toInt();

        // Prepare data packet: [PWM (1 byte), Duration (2 bytes, big-endian)]
        uint8_t data[3];
        data[0] = pwm;
        data[1] = (duration >> 8) & 0xFF; // High byte
        data[2] = duration & 0xFF;        // Low byte

        Serial.printf("Sending: PWM=%d, Duration=%d ms\n", pwm, duration);

        Wire.beginTransmission(I2C_SLAVE_ADDR);
        Wire.write(CMD_SET_HEATING);
        Wire.write(data, 3);
        uint8_t result = Wire.endTransmission();

        if (result == 0)
        {
            Serial.println("✓ Single heating parameter set successfully");
        }
        else
        {
            Serial.printf("✗ Failed to send heating parameters (error: %d)\n", result);
        }
    }
    else if (option == "2")
    {
        // New functionality: send heater settings list via serial protocol
        Serial.print("Enter PWM values (comma-separated, e.g., 100,120,140,160): ");
        while (!Serial.available())
        {
            delay(10);
        }
        String heaterList = Serial.readStringUntil('\n');
        heaterList.trim();

        Serial.print("Enter heating duration (ms): ");
        while (!Serial.available())
        {
            delay(10);
        }
        String duration = Serial.readStringUntil('\n');
        duration.trim();

        Serial.printf("Sending heater settings: %s with duration: %s ms\n", heaterList.c_str(), duration.c_str());

        // Send heaterSettings command
        String heaterCommand = "heaterSettings:" + heaterList;
        Wire.beginTransmission(I2C_SLAVE_ADDR);
        Wire.write(CMD_SET_HEATING);
        Wire.write(0xFF); // Special marker to indicate this is a heater settings list
        for (int i = 0; i < heaterCommand.length(); i++)
        {
            Wire.write(heaterCommand[i]);
        }
        uint8_t result1 = Wire.endTransmission();

        delay(100); // Small delay between commands

        // Send heatingtime command
        String timeCommand = "heatingtime:" + duration;
        Wire.beginTransmission(I2C_SLAVE_ADDR);
        Wire.write(CMD_SET_HEATING);
        Wire.write(0xFE); // Special marker to indicate this is heating time
        for (int i = 0; i < timeCommand.length(); i++)
        {
            Wire.write(timeCommand[i]);
        }
        uint8_t result2 = Wire.endTransmission();

        if (result1 == 0 && result2 == 0)
        {
            Serial.println("✓ Heater settings list and duration set successfully");
        }
        else
        {
            Serial.printf("✗ Failed to send commands (errors: %d, %d)\n", result1, result2);
        }
    }
    else
    {
        Serial.println("✗ Invalid option");
    }
    Serial.println();
}

void startExperiment()
{
    Serial.println("\n=== Start Experiment ===");
    Serial.print("Enter experiment name: ");

    while (!Serial.available())
    {
        delay(10);
    }
    String expName = Serial.readStringUntil('\n');
    expName.trim();

    if (expName.length() == 0)
    {
        Serial.println("✗ Experiment name cannot be empty");
        return;
    }

    Serial.printf("Starting experiment: '%s'\n", expName.c_str());

    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_START_EXP);
    // Send experiment name as string
    for (int i = 0; i < expName.length(); i++)
    {
        Wire.write(expName[i]);
    }
    uint8_t result = Wire.endTransmission();

    if (result == 0)
    {
        Serial.println("✓ Experiment started successfully");
        Serial.println("The slave should now be recording data...");
    }
    else
    {
        Serial.printf("✗ Failed to start experiment (error: %d)\n", result);
    }
    Serial.println();
}

void stopExperiment()
{
    Serial.println("\n=== Stop Experiment ===");

    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_STOP_EXP);
    uint8_t result = Wire.endTransmission();

    if (result == 0)
    {
        Serial.println("✓ Experiment stopped successfully");
        Serial.println("The slave should save any remaining data...");
    }
    else
    {
        Serial.printf("✗ Failed to stop experiment (error: %d)\n", result);
    }
    Serial.println();
}

void readData()
{
    // Request data from slave
    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_READ_DATA);
    uint8_t result = Wire.endTransmission();

    if (result != 0)
    {
        Serial.printf("✗ Failed to request data (error: %d)\n", result);
        return;
    }

    // Request 8 bytes of response data
    uint8_t bytesReceived = Wire.requestFrom(I2C_SLAVE_ADDR, 8);

    if (bytesReceived == 8)
    {
        uint8_t response[8];
        bool allZeros = true;
        bool allOnes = true;
        
        for (int i = 0; i < 8; i++)
        {
            response[i] = Wire.read();
            if (response[i] != 0) allZeros = false;
            if (response[i] != 0xFF) allOnes = false;
        }

        // Check if response indicates no data (all zeros or all 0xFF)
        if (allZeros || allOnes)
        {
            Serial.println("No new data available (experiment might be idle)");
            return;
        }

        // Parse the response according to slave's format
        uint16_t setting = (response[0] << 8) | response[1];
        uint32_t timestamp = (response[2] << 24) | (response[3] << 16) | (response[4] << 8) | response[5];
        int16_t raw_value = (response[6] << 8) | response[7];

        // Additional validation: check for reasonable values
        // Setting should be 0-255 (PWM range), timestamp should be reasonable (not max uint32)
        // Raw value should be within ADC range (-32768 to 32767, but typically much smaller)
        if (setting > 255 || timestamp == 0xFFFFFFFF || raw_value == -1 || raw_value == 0)
        {
            Serial.println("No new data available (experiment might be idle)");
            return;
        }

        // Data appears valid, display it
        Serial.printf("Data: Setting=%d, Timestamp=%lu, Raw=%d\n", setting, timestamp, raw_value);

        // Convert to resistance (same calculation as slave) - with proper error handling
        float voltage = raw_value * 0.000125; // Convert to voltage
        
        if (voltage > 0 && voltage < 3.3) // Protect against division by zero and invalid voltages
        {
            float load_resistance = 10000.0;
            float input_voltage = 3.3;
            
            // Use proper voltage divider formula: R1 = R_load * (V_in - V_out) / V_out
            float resistance = load_resistance * (input_voltage - voltage) / voltage;
            
            // Sanity check for realistic resistance values
            if (resistance > 0 && resistance < 1e9) // Cap at 1GOhm
            {
                Serial.printf("      Voltage=%.3fV, Resistance=%.2f ohms\n", voltage, resistance);
            }
            else
            {
                Serial.printf("      Voltage=%.3fV, Resistance=INVALID (%.2f ohms)\n", voltage, resistance);
            }
        }
        else
        {
            Serial.printf("      Voltage=%.3fV (INVALID - cannot calculate resistance)\n", voltage);
        }
    }
    else
    {
        Serial.printf("✗ Expected 8 bytes, received %d bytes\n", bytesReceived);
    }
}

void setWiFiCredentials()
{
    Serial.println("\n=== Set WiFi Credentials ===");
    Serial.print("Enter WiFi SSID: ");

    while (!Serial.available())
    {
        delay(10);
    }
    String ssid = Serial.readStringUntil('\n');
    ssid.trim();

    if (ssid.length() == 0)
    {
        Serial.println("✗ SSID cannot be empty");
        return;
    }

    Serial.print("Enter WiFi Password: ");
    while (!Serial.available())
    {
        delay(10);
    }
    String password = Serial.readStringUntil('\n');
    password.trim();

    if (password.length() == 0)
    {
        Serial.println("✗ Password cannot be empty");
        return;
    }

    // Prepare WiFi credentials in format: "SSID|PASSWORD"
    String wifiData = ssid + "|" + password;
    
    Serial.printf("Setting WiFi credentials: SSID='%s', Password='%s'\n", ssid.c_str(), password.c_str());
    Serial.println("Note: The slave will attempt to connect with new credentials and save them if successful");

    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_SET_WIFI);
    // Send WiFi credentials as string
    for (int i = 0; i < wifiData.length(); i++)
    {
        Wire.write(wifiData[i]);
    }
    uint8_t result = Wire.endTransmission();

    if (result == 0)
    {
        Serial.println("✓ WiFi credentials sent successfully");
        Serial.println("Slave is attempting to connect...");
        
        // Wait a moment for the connection attempt and then check status
        delay(3000);
        checkWiFiStatus();
    }
    else
    {
        Serial.printf("✗ Failed to send WiFi credentials (error: %d)\n", result);
    }
    Serial.println();
}

void checkWiFiStatus()
{
    Serial.println("\n=== Check WiFi Status ===");

    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_WIFI_STATUS);
    uint8_t result = Wire.endTransmission();

    if (result != 0)
    {
        Serial.printf("✗ Failed to request WiFi status (error: %d)\n", result);
        return;
    }

    // Request 8 bytes of response data
    uint8_t bytesReceived = Wire.requestFrom(I2C_SLAVE_ADDR, 8);

    if (bytesReceived >= 2)
    {
        uint8_t response[8];
        for (int i = 0; i < bytesReceived; i++)
        {
            response[i] = Wire.read();
        }

        uint8_t wifiCommandStatus = response[0]; // 0=unknown, 1=connecting, 2=success, 3=failed
        uint8_t wifiConnectionStatus = response[1]; // Current WiFi.status()

        Serial.print("WiFi Command Status: ");
        switch (wifiCommandStatus)
        {
            case 0: Serial.println("Unknown"); break;
            case 1: Serial.println("Connecting..."); break;
            case 2: Serial.println("✓ Success - Connected with new credentials"); break;
            case 3: Serial.println("✗ Failed - Could not connect with provided credentials"); break;
            default: Serial.printf("Invalid status (%d)\n", wifiCommandStatus); break;
        }

        Serial.print("WiFi Connection Status: ");
        switch (wifiConnectionStatus)
        {
            case 3: Serial.println("✓ Connected (WL_CONNECTED or AP mode active)"); break;
            case 1: Serial.println("No SSID available (WL_NO_SSID_AVAIL)"); break;
            case 4: Serial.println("Connection failed (WL_CONNECT_FAILED)"); break;
            case 6: Serial.println("Disconnected (WL_DISCONNECTED)"); break;
            case 0: Serial.println("Idle status (WL_IDLE_STATUS)"); break;
            default: Serial.printf("Status code: %d\n", wifiConnectionStatus); break;
        }
    }
    else
    {
        Serial.printf("✗ Expected at least 2 bytes, received %d bytes\n", bytesReceived);
    }
    Serial.println();
}

void getIPAddress()
{
    Serial.println("\n=== Get IP Address ===");

    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_GET_IP);
    uint8_t result = Wire.endTransmission();

    if (result != 0)
    {
        Serial.printf("✗ Failed to request IP address (error: %d)\n", result);
        return;
    }

    // Request 16 bytes of response data
    uint8_t bytesReceived = Wire.requestFrom(I2C_SLAVE_ADDR, 16);

    if (bytesReceived >= 1)
    {
        uint8_t response[16];
        for (int i = 0; i < bytesReceived; i++)
        {
            response[i] = Wire.read();
        }

        uint8_t connectionStatus = response[0];
        
        if (connectionStatus == 1 && bytesReceived >= 5)
        {
            // Connected - extract IP address
            IPAddress ip(response[1], response[2], response[3], response[4]);
            Serial.printf("✓ Connected - IP Address: %s\n", ip.toString().c_str());
            
            // Extract SSID if available
            if (bytesReceived >= 6)
            {
                uint8_t ssidLength = response[5];
                if (ssidLength > 0 && ssidLength <= 10 && bytesReceived >= (6 + ssidLength))
                {
                    String ssid = "";
                    for (int i = 0; i < ssidLength; i++)
                    {
                        ssid += (char)response[6 + i];
                    }
                    Serial.printf("  Network: %s\n", ssid.c_str());
                }
            }
        }
        else
        {
            Serial.println("✗ Not connected to WiFi");
        }
    }
    else
    {
        Serial.printf("✗ Expected at least 1 byte, received %d bytes\n", bytesReceived);
    }
    Serial.println();
}

void readLogs()
{
    Serial.println("\n=== Read Debug Logs from Slave ===");
    Serial.println("Warning: Reading logs during experiments may cause I2C congestion");
    Serial.println("Note: System now stores complete log history (not circular buffer)");

    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_READ_LOGS);
    uint8_t result = Wire.endTransmission();

    if (result != 0)
    {
        Serial.printf("✗ Failed to send read logs command (error: %d)\n", result);
        return;
    }

    Serial.println("Debug logs from slave (complete history):");
    Serial.println("Timestamp\t\tMessage");
    Serial.println("----------------------------------------");

    // Read log entries
    int logCount = 0;
    while (true)
    {
        uint8_t bytesReceived = Wire.requestFrom(I2C_SLAVE_ADDR, 32);
        
        if (bytesReceived == 32)
        {
            uint8_t response[32];
            for (int i = 0; i < 32; i++)
            {
                response[i] = Wire.read();
            }

            uint8_t msgLen = response[0];
            if (msgLen == 0)
            {
                // End of logs
                break;
            }

            uint32_t timestamp = (response[1] << 24) | (response[2] << 16) | 
                               (response[3] << 8) | response[4];
            
            char message[27] = {0}; // Max 26 chars + null terminator
            memcpy(message, &response[5], min(msgLen, (uint8_t)26));
            
            Serial.printf("[%lu]\t%s\n", timestamp, message);
            logCount++;
        }
        else
        {
            Serial.printf("✗ Expected 32 bytes, received %d bytes\n", bytesReceived);
            break;
        }
        
        delay(50); // Longer delay between log requests to reduce I2C load
    }
    
    Serial.println("----------------------------------------");
    Serial.printf("✓ Read %d log entries (complete history)\n", logCount);
    Serial.println();
}

void clearLogs()
{
    Serial.println("\n=== Clear Debug Logs on Slave ===");

    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(CMD_CLEAR_LOGS);
    uint8_t result = Wire.endTransmission();

    if (result == 0)
    {
        Serial.println("✓ Debug logs cleared successfully");
    }
    else
    {
        Serial.printf("✗ Failed to clear logs (error: %d)\n", result);
    }
    Serial.println();
}



// Utility function to send simple commands
void sendCommand(uint8_t command)
{
    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(command);
    uint8_t result = Wire.endTransmission();

    if (result == 0)
    {
        Serial.println("✓ Command sent successfully");
    }
    else
    {
        Serial.printf("✗ Failed to send command (error: %d)\n", result);
    }
}

// Utility function to send commands with data
void sendCommandWithData(uint8_t command, uint8_t *data, size_t length)
{
    Wire.beginTransmission(I2C_SLAVE_ADDR);
    Wire.write(command);
    Wire.write(data, length);
    uint8_t result = Wire.endTransmission();

    if (result == 0)
    {
        Serial.println("✓ Command with data sent successfully");
    }
    else
    {
        Serial.printf("✗ Failed to send command with data (error: %d)\n", result);
    }
}

void continuousDataReading()
{
    Serial.println("Starting continuous data reading (press any key to stop)...");
    Serial.println("Note: Data will only be available when an experiment is running.");
    
    unsigned long startTime = millis();
    int readCount = 0;
    int dataCount = 0;
    
    while (!Serial.available())
    {
        readCount++;
        Serial.printf("[%lu] Reading #%d: ", millis() - startTime, readCount);
        
        // Temporarily capture serial output to detect if data was found
        // We'll rely on readData() to print appropriate messages
        readData();
        
        // Small delay to allow serial output to complete
        delay(50);
        
        // Check if we got valid data (this is approximate - readData prints the status)
        // The actual validation happens inside readData()
        
        delay(950); // Total 1 second between reads
    }
    
    Serial.readString(); // Clear the input buffer
    Serial.printf("Stopped continuous reading. Total reads: %d\n", readCount);
}

