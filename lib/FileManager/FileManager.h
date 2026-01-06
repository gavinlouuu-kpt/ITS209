#ifndef FILE_MANAGER_H
#define FILE_MANAGER_H

#include <Arduino.h>
#include <LittleFS.h>
#include <FS.h>
#include <vector>
#include "definitions.h"
#include "CircularBuffer.hpp"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

// Storage backend types for future extensibility
enum StorageBackend {
    STORAGE_LITTLEFS,
    STORAGE_SD_CARD,
    STORAGE_SPIFFS
};

// File operation result codes
enum FileOperationResult {
    FILE_OP_SUCCESS,
    FILE_OP_ERROR_OPEN,
    FILE_OP_ERROR_WRITE,
    FILE_OP_ERROR_READ,
    FILE_OP_ERROR_NOT_FOUND,
    FILE_OP_ERROR_STORAGE_FULL,
    FILE_OP_ERROR_INIT
};

// Callback function types
using AddLogFn = void (*)(const String&);

class FileManager {
public:
    FileManager(AddLogFn addLog);
    
    // Storage initialization and management
    bool initializeStorage(StorageBackend backend = STORAGE_LITTLEFS);
    bool formatStorage();
    bool isStorageReady();
    String getStorageInfo();
    
    // File operations
    FileOperationResult createExperimentFile(const String& experimentName, String& filename);
    FileOperationResult appendDataToFile(const String& filename, CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE>* buffer, bool isFirstWrite = false);
    FileOperationResult listFiles(std::vector<String>& filenames, std::vector<size_t>& filesizes);
    FileOperationResult readFile(const String& filename, File& file);
    FileOperationResult deleteFile(const String& filename);
    FileOperationResult getFileSize(const String& filename, size_t& size);
    
    // Task management for async file operations
    void createFileWriteTask(CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE>* buffer, const String& filename, bool isFirstWrite);
    bool isWriteTaskActive();
    
    // Storage backend management
    StorageBackend getCurrentBackend() const { return _currentBackend; }
    void setStorageBackend(StorageBackend backend) { _currentBackend = backend; }
    
    // Static task wrapper for FreeRTOS
    static void fileWriteTaskWrapper(void* parameter);

private:
    // Internal file operations
    FileOperationResult writeCSVHeader(File& file);
    FileOperationResult writeDataToCSV(File& file, CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE>* buffer);
    float calculateResistance(int16_t ads_reading);
    
    // Storage abstraction
    FS* getFileSystem();
    String getStoragePath(const String& filename);
    
    // Member variables
    StorageBackend _currentBackend;
    AddLogFn _addDebugLog;
    volatile bool _writeTaskActive;
    TaskHandle_t _fileWriteTaskHandle;
    
    // Storage configuration
    const float _loadResistance = 10000.0; // Load resistance in ohms
    const float _inputVoltage = 3.3;       // Input voltage in volts
    
    // Task parameters structure
    struct FileWriteTaskParams {
        FileManager* manager;
        CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE>* buffer;
        String filename;
        bool isFirstWrite;
    };
};

#endif // FILE_MANAGER_H 