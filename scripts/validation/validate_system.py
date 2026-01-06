#!/usr/bin/env python3
"""
System Validation: Test the gas classifier with real data
Demonstrates practical usage and validation approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'analysis'))
from gas_classifier_realtime import GasSensorClassifier
import warnings
warnings.filterwarnings('ignore')

def load_static_data():
    """Load the static gas sensor data"""
    etoh_1 = pd.read_csv('results/static/static_etoh_1.csv')
    etoh_2 = pd.read_csv('results/static/static_etoh_2.csv')
    prop_1 = pd.read_csv('results/static/static_2pol_1.csv')
    prop_2 = pd.read_csv('results/static/static_2pol_2.csv')
    
    return etoh_1, etoh_2, prop_1, prop_2

def align_data_simple(df):
    """Simple data alignment for validation"""
    # Get PWM setting columns
    pwm_cols = [col for col in df.columns if col.startswith('setting_')]
    
    # Convert to numeric
    for col in pwm_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Group by cycles and take median of each setting
    num_settings = len(pwm_cols)
    df['cycle'] = df.index // num_settings
    
    aligned_data = []
    for cycle in df['cycle'].unique():
        cycle_data = df[df['cycle'] == cycle]
        if len(cycle_data) < num_settings:
            continue
            
        readings = []
        for col in pwm_cols:
            values = cycle_data[col].dropna()
            if len(values) > 0:
                readings.append(np.median(values))
            else:
                readings.append(np.nan)
        
        if not any(np.isnan(readings)):
            aligned_data.append(readings)
    
    return np.array(aligned_data)

def validate_classifier():
    """Comprehensive validation of the gas classifier"""
    print("="*80)
    print("GAS SENSOR CLASSIFIER VALIDATION")
    print("="*80)
    
    # Load data
    etoh_1, etoh_2, prop_1, prop_2 = load_static_data()
    print(f"Loaded data: ETH1={etoh_1.shape}, ETH2={etoh_2.shape}")
    print(f"             PROP1={prop_1.shape}, PROP2={prop_2.shape}")
    
    # Align data
    etoh_1_aligned = align_data_simple(etoh_1)
    etoh_2_aligned = align_data_simple(etoh_2)
    prop_1_aligned = align_data_simple(prop_1)
    prop_2_aligned = align_data_simple(prop_2)
    
    print(f"\nAligned: ETH1={len(etoh_1_aligned)}, ETH2={len(etoh_2_aligned)}")
    print(f"         PROP1={len(prop_1_aligned)}, PROP2={len(prop_2_aligned)}")
    
    # Initialize classifier
    classifier = GasSensorClassifier()
    
    # Test on real data
    print("\n" + "="*60)
    print("REAL DATA VALIDATION")
    print("="*60)
    
    datasets = [
        (etoh_1_aligned, "Ethanol-1", "ethanol"),
        (etoh_2_aligned, "Ethanol-2", "ethanol"),
        (prop_1_aligned, "2-Propanol-1", "2-propanol"),
        (prop_2_aligned, "2-Propanol-2", "2-propanol")
    ]
    
    all_results = []
    
    for data, name, true_label in datasets:
        print(f"\nTesting {name} ({len(data)} samples):")
        print("-" * 50)
        
        simple_correct = 0
        advanced_correct = 0
        confidences_simple = []
        confidences_advanced = []
        
        # Test random sample of readings
        sample_size = min(20, len(data))
        sample_indices = np.random.choice(len(data), sample_size, replace=False)
        
        for i in sample_indices:
            readings = data[i]
            
            # Simple classification
            pred_simple, conf_simple = classifier.classify_simple(readings)
            confidences_simple.append(conf_simple)
            if pred_simple == true_label:
                simple_correct += 1
            
            # Advanced classification
            pred_advanced, conf_advanced = classifier.classify_advanced(readings)
            confidences_advanced.append(conf_advanced)
            if pred_advanced == true_label:
                advanced_correct += 1
        
        simple_accuracy = simple_correct / sample_size
        advanced_accuracy = advanced_correct / sample_size
        
        print(f"Simple Method:")
        print(f"  Accuracy: {simple_accuracy:.1%} ({simple_correct}/{sample_size})")
        print(f"  Avg Confidence: {np.mean(confidences_simple):.3f}")
        
        print(f"Advanced Method:")
        print(f"  Accuracy: {advanced_accuracy:.1%} ({advanced_correct}/{sample_size})")
        print(f"  Avg Confidence: {np.mean(confidences_advanced):.3f}")
        
        all_results.append({
            'dataset': name,
            'true_label': true_label,
            'simple_accuracy': simple_accuracy,
            'advanced_accuracy': advanced_accuracy,
            'simple_confidence': np.mean(confidences_simple),
            'advanced_confidence': np.mean(confidences_advanced)
        })
    
    # Overall results
    print("\n" + "="*60)
    print("OVERALL VALIDATION RESULTS")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)
    
    print("\nAccuracy by Dataset:")
    print("-" * 60)
    print(f"{'Dataset':15s} {'True Label':12s} {'Simple':>8s} {'Advanced':>10s}")
    print("-" * 60)
    
    for _, row in results_df.iterrows():
        print(f"{row['dataset']:15s} {row['true_label']:12s} "
              f"{row['simple_accuracy']:>7.1%} {row['advanced_accuracy']:>9.1%}")
    
    # Summary statistics
    overall_simple = results_df['simple_accuracy'].mean()
    overall_advanced = results_df['advanced_accuracy'].mean()
    
    print(f"\nOverall Performance:")
    print(f"  Simple Method:   {overall_simple:.1%} accuracy")
    print(f"  Advanced Method: {overall_advanced:.1%} accuracy")
    
    # Analyze feature distributions
    print("\n" + "="*60)
    print("FEATURE ANALYSIS ON REAL DATA")
    print("="*60)
    
    # Calculate key features for each dataset
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (data, name, true_label) in enumerate(datasets):
        ax = axes[i//2, i%2]
        
        # Extract PWM 240 readings (primary discriminator)
        pwm_240_readings = data[:, 0]  # First column is PWM 240
        
        # Plot distribution
        ax.hist(pwm_240_readings, bins=30, alpha=0.7, 
                color='blue' if 'ethanol' in true_label.lower() else 'red')
        ax.axvline(classifier.primary_threshold, color='black', linestyle='--', 
                   label=f'Threshold ({classifier.primary_threshold})')
        ax.set_title(f'{name}\nPWM 240 Distribution')
        ax.set_xlabel('Resistance (Ω)')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Statistics
        mean_val = np.mean(pwm_240_readings)
        std_val = np.std(pwm_240_readings)
        ax.text(0.05, 0.95, f'Mean: {mean_val:.0f}\nStd: {std_val:.0f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results_df

def demonstrate_real_time_usage():
    """Demonstrate how to use the classifier in real-time"""
    print("\n" + "="*60)
    print("REAL-TIME USAGE DEMONSTRATION")
    print("="*60)
    
    classifier = GasSensorClassifier()
    
    # Simulate real-time sensor readings
    print("\nSimulating real-time gas detection:")
    print("-" * 40)
    
    # Example scenarios
    scenarios = [
        {
            'name': 'Pure Ethanol (low resistance)',
            'readings': [5000000, 15000, 250000, 180000, 12000, 3500, 1100, 200, 90, 45, 20]
        },
        {
            'name': 'Pure 2-Propanol (high resistance)', 
            'readings': [25000000, 18000, 300000, 200000, 15000, 4000, 1200, 250, 100, 50, 25]
        },
        {
            'name': 'Mixed/Unknown gas',
            'readings': [12000000, 16000, 275000, 190000, 13500, 3750, 1150, 225, 95, 47, 22]
        }
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        readings = scenario['readings']
        
        # Classify using both methods
        simple_result, simple_conf = classifier.classify_simple(readings)
        advanced_result, advanced_conf = classifier.classify_advanced(readings)
        
        print(f"  Readings: PWM240={readings[0]:,}, PWM220={readings[9]:,}")
        print(f"  Simple:   {simple_result} (confidence: {simple_conf:.3f})")
        print(f"  Advanced: {advanced_result} (confidence: {advanced_conf:.3f})")
        
        # Extract key features for interpretation
        features = classifier.extract_key_features(readings)
        print(f"  Key ratio 240/40: {features[3]:.1f}")

def generate_integration_examples():
    """Generate examples for system integration"""
    print("\n" + "="*60)
    print("SYSTEM INTEGRATION EXAMPLES")
    print("="*60)
    
    # Arduino integration example
    arduino_code = '''
// Arduino integration example
#include "gas_classifier_embedded.h"

// PWM settings array (from your sensor configuration)
uint32_t resistance_readings[NUM_PWM_SETTINGS];

void setup() {
    Serial.begin(115200);
    // Initialize your sensor system here
}

void loop() {
    // Read resistance values from your sensor at each PWM setting
    // (Replace with your actual sensor reading code)
    for (int i = 0; i < NUM_PWM_SETTINGS; i++) {
        resistance_readings[i] = read_sensor_at_pwm(pwm_settings[i]);
        delay(10); // Adjust timing as needed
    }
    
    // Classify the gas
    ClassificationResult result = classify_gas_fast(resistance_readings);
    
    // Output results
    Serial.print("Gas Type: ");
    Serial.print(gas_type_to_string(result.gas_type));
    Serial.print(", Confidence: ");
    Serial.println(result.confidence);
    
    delay(1000); // Classify every second
}

// Your sensor reading function (implement based on your hardware)
uint32_t read_sensor_at_pwm(uint16_t pwm_value) {
    // Set PWM value
    // Read ADC
    // Convert to resistance
    // Return resistance value
    return 0; // Placeholder
}
'''
    
    # Python integration example
    python_integration = '''
# Python integration example
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'analysis'))
    from gas_classifier_realtime import GasSensorClassifier
    import time

class SensorSystem:
    def __init__(self):
        self.classifier = GasSensorClassifier()
        self.pwm_settings = [240, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]
    
    def read_sensor_cycle(self):
        """Read one complete cycle of all PWM settings"""
        readings = []
        for pwm in self.pwm_settings:
            # Set PWM value on your sensor
            # self.set_pwm(pwm)
            # time.sleep(0.01)  # Settling time
            
            # Read resistance value
            # resistance = self.read_adc_to_resistance()
            resistance = 10000000  # Placeholder
            readings.append(resistance)
        
        return readings
    
    def continuous_monitoring(self):
        """Continuous gas monitoring"""
        while True:
            try:
                readings = self.read_sensor_cycle()
                
                # Classify gas
                gas_type, confidence = self.classifier.classify_advanced(readings)
                
                # Log results
                timestamp = time.time()
                print(f"{timestamp:.0f}: {gas_type} (conf: {confidence:.3f})")
                
                # Optional: Save to database, trigger alerts, etc.
                if confidence > 0.8:
                    self.handle_confident_detection(gas_type, confidence)
                
                time.sleep(1)  # Sample every second
                
            except KeyboardInterrupt:
                break
    
    def handle_confident_detection(self, gas_type, confidence):
        """Handle high-confidence detections"""
        print(f"HIGH CONFIDENCE DETECTION: {gas_type} ({confidence:.3f})")
        # Add your alert/logging logic here

# Usage
if __name__ == "__main__":
    system = SensorSystem()
    system.continuous_monitoring()
'''
    
    # Save integration examples
    with open('arduino_integration_example.cpp', 'w') as f:
        f.write(arduino_code)
    
    with open('python_integration_example.py', 'w') as f:
        f.write(python_integration)
    
    print("Generated integration examples:")
    print("  - arduino_integration_example.cpp")
    print("  - python_integration_example.py")

if __name__ == "__main__":
    # Run validation
    results = validate_classifier()
    
    # Demonstrate real-time usage
    demonstrate_real_time_usage()
    
    # Generate integration examples
    generate_integration_examples()
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\nFiles generated:")
    print("  ✓ validation_results.png - Feature distribution plots")
    print("  ✓ arduino_integration_example.cpp - Arduino code")
    print("  ✓ python_integration_example.py - Python integration")
    print("\nNext steps:")
    print("  1. Review validation results and adjust thresholds if needed")
    print("  2. Test with your actual sensor hardware")
    print("  3. Implement the integration examples in your system")
    print("  4. Collect more data to improve the model") 