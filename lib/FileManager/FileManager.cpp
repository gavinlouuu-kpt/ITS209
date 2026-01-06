#include "FileManager.h"

FileManager::FileManager(AddLogFn addLog) 
    : _currentBackend(STORAGE_LITTLEFS), 
      _addDebugLog(addLog), 
      _writeTaskActive(false),
      _fileWriteTaskHandle(NULL) {
}

bool FileManager::initializeStorage(StorageBackend backend) {
    _currentBackend = backend;
    
    switch (backend) {
        case STORAGE_LITTLEFS:
            if (!LittleFS.begin()) {
                _addDebugLog("ERROR: Failed to initialize LittleFS");
                return false;
            }
            _addDebugLog("LittleFS initialized successfully");
            return true;
            
        case STORAGE_SD_CARD:
            // Future implementation for SD card
            _addDebugLog("SD Card support not yet implemented");
            return false;
            
        case STORAGE_SPIFFS:
            // Future implementation for SPIFFS
            _addDebugLog("SPIFFS support not yet implemented");
            return false;
            
        default:
            _addDebugLog("ERROR: Unknown storage backend");
            return false;
    }
}

bool FileManager::formatStorage() {
    switch (_currentBackend) {
        case STORAGE_LITTLEFS:
            if (LittleFS.format()) {
                _addDebugLog("LittleFS formatted successfully");
                return true;
            } else {
                _addDebugLog("ERROR: Failed to format LittleFS");
                return false;
            }
            
        case STORAGE_SD_CARD:
            _addDebugLog("SD Card format not yet implemented");
            return false;
            
        case STORAGE_SPIFFS:
            _addDebugLog("SPIFFS format not yet implemented");
            return false;
            
        default:
            _addDebugLog("ERROR: Unknown storage backend for format");
            return false;
    }
}

bool FileManager::isStorageReady() {
    switch (_currentBackend) {
        case STORAGE_LITTLEFS:
            return LittleFS.begin(false); // Don't format if fails
            
        case STORAGE_SD_CARD:
            // Future implementation
            return false;
            
        case STORAGE_SPIFFS:
            // Future implementation
            return false;
            
        default:
            return false;
    }
}

String FileManager::getStorageInfo() {
    switch (_currentBackend) {
        case STORAGE_LITTLEFS: {
            size_t totalBytes = LittleFS.totalBytes();
            size_t usedBytes = LittleFS.usedBytes();
            return "LittleFS: " + String(usedBytes) + "/" + String(totalBytes) + " bytes used";
        }
        
        case STORAGE_SD_CARD:
            return "SD Card: Not implemented";
            
        case STORAGE_SPIFFS:
            return "SPIFFS: Not implemented";
            
        default:
            return "Unknown storage backend";
    }
}

FS* FileManager::getFileSystem() {
    switch (_currentBackend) {
        case STORAGE_LITTLEFS:
            return &LittleFS;
            
        case STORAGE_SD_CARD:
            // Future: return SD card FS
            return nullptr;
            
        case STORAGE_SPIFFS:
            // Future: return SPIFFS FS
            return nullptr;
            
        default:
            return nullptr;
    }
}

String FileManager::getStoragePath(const String& filename) {
    // Ensure filename starts with /
    if (filename.startsWith("/")) {
        return filename;
    } else {
        return "/" + filename;
    }
}

FileOperationResult FileManager::createExperimentFile(const String& experimentName, String& filename) {
    filename = "/" + experimentName + ".csv";
    
    FS* fs = getFileSystem();
    if (!fs) {
        _addDebugLog("ERROR: No valid filesystem available");
        return FILE_OP_ERROR_INIT;
    }
    
    File file = fs->open(filename, "w");
    if (!file) {
        _addDebugLog("ERROR: Failed to create experiment file: " + filename);
        return FILE_OP_ERROR_OPEN;
    }
    
    FileOperationResult result = writeCSVHeader(file);
    file.close();
    
    if (result == FILE_OP_SUCCESS) {
        _addDebugLog("Created experiment file: " + filename);
    }
    
    return result;
}

FileOperationResult FileManager::writeCSVHeader(File& file) {
    if (file.println("setting,timestamp,raw_value,resistance")) {
        return FILE_OP_SUCCESS;
    } else {
        _addDebugLog("ERROR: Failed to write CSV header");
        return FILE_OP_ERROR_WRITE;
    }
}

float FileManager::calculateResistance(int16_t ads_reading) {
    float voltage = ads_reading * 0.000125; // Convert to voltage in volts
    
    // Protect against division by zero and invalid voltage values
    if (voltage <= 0 || voltage >= _inputVoltage) {
        return 0; // Return 0 for invalid readings
    }
    
    // Calculate resistance using voltage divider formula
    // R1 = R_load * (V_in - V_out) / V_out
    float R1 = _loadResistance * (_inputVoltage - voltage) / voltage;
    
    // Sanity check - return 0 for unrealistic resistance values
    if (R1 < 0 || R1 > 1e9) { // Cap at 1GOhm
        return 0;
    }
    
    return R1; // Return resistance in ohms
}

FileOperationResult FileManager::writeDataToCSV(File& file, CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE>* buffer) {
    size_t dataCount = 0;
    
    while (!buffer->isEmpty()) {
        SingleChannel data = buffer->shift();
        float resistance = calculateResistance(data.channel_0);
        
        file.print(data.setting);
        file.print(",");
        file.print(data.timestamp);
        file.print(",");
        file.print(data.channel_0);
        file.print(",");
        file.println(resistance);
        
        dataCount++;
    }
    
    _addDebugLog("Wrote " + String(dataCount) + " records to CSV");
    return FILE_OP_SUCCESS;
}

FileOperationResult FileManager::appendDataToFile(const String& filename, CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE>* buffer, bool isFirstWrite) {
    FS* fs = getFileSystem();
    if (!fs) {
        return FILE_OP_ERROR_INIT;
    }
    
    String fullPath = getStoragePath(filename);
    const char* mode = isFirstWrite ? "w" : "a";
    
    File file = fs->open(fullPath, mode);
    if (!file) {
        _addDebugLog("ERROR: Failed to open file for " + String(isFirstWrite ? "writing" : "appending") + ": " + fullPath);
        return FILE_OP_ERROR_OPEN;
    }
    
    FileOperationResult result = FILE_OP_SUCCESS;
    
    // Write header if it's the first write
    if (isFirstWrite) {
        result = writeCSVHeader(file);
        if (result != FILE_OP_SUCCESS) {
            file.close();
            return result;
        }
    }
    
    // Write data
    result = writeDataToCSV(file, buffer);
    file.close();
    
    return result;
}

FileOperationResult FileManager::listFiles(std::vector<String>& filenames, std::vector<size_t>& filesizes) {
    FS* fs = getFileSystem();
    if (!fs) {
        return FILE_OP_ERROR_INIT;
    }
    
    filenames.clear();
    filesizes.clear();
    
    File dir = fs->open("/");
    if (!dir) {
        _addDebugLog("ERROR: Failed to open root directory");
        return FILE_OP_ERROR_OPEN;
    }
    
    File file = dir.openNextFile();
    while (file) {
        String filename = file.name();
        size_t filesize = file.size();
        
        // Remove leading slash for consistency
        if (filename.startsWith("/")) {
            filename = filename.substring(1);
        }
        
        filenames.push_back(filename);
        filesizes.push_back(filesize);
        
        file.close();
        file = dir.openNextFile();
    }
    
    dir.close();
    _addDebugLog("Listed " + String(filenames.size()) + " files");
    return FILE_OP_SUCCESS;
}

FileOperationResult FileManager::readFile(const String& filename, File& file) {
    FS* fs = getFileSystem();
    if (!fs) {
        return FILE_OP_ERROR_INIT;
    }
    
    String fullPath = getStoragePath(filename);
    file = fs->open(fullPath, "r");
    
    if (!file) {
        _addDebugLog("ERROR: Failed to open file for reading: " + fullPath);
        return FILE_OP_ERROR_NOT_FOUND;
    }
    
    _addDebugLog("Opened file for reading: " + fullPath + " (" + String(file.size()) + " bytes)");
    return FILE_OP_SUCCESS;
}

FileOperationResult FileManager::deleteFile(const String& filename) {
    FS* fs = getFileSystem();
    if (!fs) {
        return FILE_OP_ERROR_INIT;
    }
    
    String fullPath = getStoragePath(filename);
    
    if (fs->remove(fullPath)) {
        _addDebugLog("Deleted file: " + fullPath);
        return FILE_OP_SUCCESS;
    } else {
        _addDebugLog("ERROR: Failed to delete file: " + fullPath);
        return FILE_OP_ERROR_NOT_FOUND;
    }
}

FileOperationResult FileManager::getFileSize(const String& filename, size_t& size) {
    FS* fs = getFileSystem();
    if (!fs) {
        return FILE_OP_ERROR_INIT;
    }
    
    String fullPath = getStoragePath(filename);
    File file = fs->open(fullPath, "r");
    
    if (!file) {
        return FILE_OP_ERROR_NOT_FOUND;
    }
    
    size = file.size();
    file.close();
    return FILE_OP_SUCCESS;
}

void FileManager::createFileWriteTask(CircularBuffer<SingleChannel, CIRCULAR_BUFFER_SIZE>* buffer, const String& filename, bool isFirstWrite) {
    if (_writeTaskActive) {
        _addDebugLog("WARNING: File write task already active, skipping new task");
        return;
    }
    
    // Create task parameters
    FileWriteTaskParams* params = new FileWriteTaskParams();
    params->manager = this;
    params->buffer = buffer;
    params->filename = filename;
    params->isFirstWrite = isFirstWrite;
    
    _writeTaskActive = true;
    
    BaseType_t result = xTaskCreate(
        fileWriteTaskWrapper,    // Task function
        "FileWrite",             // Task name
        4096,                    // Stack size
        (void*)params,           // Parameter
        1,                       // Priority
        &_fileWriteTaskHandle    // Task handle
    );
    
    if (result != pdPASS) {
        _addDebugLog("ERROR: Failed to create file write task");
        _writeTaskActive = false;
        delete params;
    } else {
        _addDebugLog("Created file write task for: " + filename);
    }
}

bool FileManager::isWriteTaskActive() {
    return _writeTaskActive;
}

void FileManager::fileWriteTaskWrapper(void* parameter) {
    FileWriteTaskParams* params = static_cast<FileWriteTaskParams*>(parameter);
    
    if (params && params->manager) {
        FileOperationResult result = params->manager->appendDataToFile(
            params->filename, 
            params->buffer, 
            params->isFirstWrite
        );
        
        if (result == FILE_OP_SUCCESS) {
            params->manager->_addDebugLog("File write task completed successfully");
        } else {
            params->manager->_addDebugLog("ERROR: File write task failed with code " + String(result));
        }
        
        params->manager->_writeTaskActive = false;
    }
    
    delete params;
    vTaskDelete(NULL);
} 