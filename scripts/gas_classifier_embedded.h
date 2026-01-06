
/*
 * Gas Sensor Classifier for Embedded Systems
 * Generated from comprehensive analysis
 * Optimized for microcontrollers
 */

#ifndef GAS_CLASSIFIER_H
#define GAS_CLASSIFIER_H

#include <stdint.h>
#include <math.h>

// Configuration
#define NUM_PWM_SETTINGS 11
#define PRIMARY_THRESHOLD 14261
#define SECONDARY_THRESHOLD 23376

// Gas types
typedef enum {
    GAS_ETHANOL = 0,
    GAS_2_PROPANOL = 1,
    GAS_UNKNOWN = 2
} GasType;

// Classification result
typedef struct {
    GasType gas_type;
    float confidence;
    uint32_t timestamp;
} ClassificationResult;

// Fast classification function (minimal CPU)
ClassificationResult classify_gas_fast(uint32_t* resistance_readings) {
    ClassificationResult result;
    result.timestamp = 0; // Set your timestamp here
    
    uint32_t primary_reading = resistance_readings[0];  // PWM 240
    uint32_t secondary_reading = resistance_readings[9]; // PWM 220
    
    // Simple threshold-based classification
    if (primary_reading > PRIMARY_THRESHOLD) {
        result.gas_type = GAS_2_PROPANOL;
        result.confidence = (float)(primary_reading - PRIMARY_THRESHOLD) / (PRIMARY_THRESHOLD * 0.2f);
    } else {
        result.gas_type = GAS_ETHANOL;
        result.confidence = (float)(PRIMARY_THRESHOLD - primary_reading) / (PRIMARY_THRESHOLD * 0.2f);
    }
    
    // Clamp confidence to [0, 1]
    if (result.confidence > 1.0f) result.confidence = 1.0f;
    if (result.confidence < 0.1f) result.confidence = 0.1f;
    
    return result;
}

// Advanced classification with multiple criteria
ClassificationResult classify_gas_advanced(uint32_t* resistance_readings) {
    ClassificationResult result;
    result.timestamp = 0; // Set your timestamp here
    
    uint32_t pwm_240 = resistance_readings[0];
    uint32_t pwm_40 = resistance_readings[1];
    uint32_t pwm_220 = resistance_readings[9];
    
    int score = 0;
    
    // Criterion 1: Primary PWM threshold
    if (pwm_240 > PRIMARY_THRESHOLD) score++;
    
    // Criterion 2: Secondary PWM threshold
    if (pwm_220 > SECONDARY_THRESHOLD) score++;
    
    // Criterion 3: Ratio analysis (avoid division by zero)
    if (pwm_40 > 1000) {
        uint32_t ratio = pwm_240 / pwm_40;
        if (ratio > 800) score++;
    }
    
    // Final classification
    if (score >= 2) {
        result.gas_type = GAS_2_PROPANOL;
        result.confidence = (float)score / 3.0f;
    } else {
        result.gas_type = GAS_ETHANOL;
        result.confidence = (float)(3 - score) / 3.0f;
    }
    
    return result;
}

// Utility function to convert result to string
const char* gas_type_to_string(GasType gas) {
    switch (gas) {
        case GAS_ETHANOL: return "Ethanol";
        case GAS_2_PROPANOL: return "2-Propanol";
        default: return "Unknown";
    }
}

#endif // GAS_CLASSIFIER_H

