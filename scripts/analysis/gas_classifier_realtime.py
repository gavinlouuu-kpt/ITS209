
#!/usr/bin/env python3
"""
Real-time Gas Sensor Classifier
Generated from comprehensive analysis
"""

import numpy as np

class GasSensorClassifier:
    def __init__(self):
        # PWM settings in order
        self.pwm_settings = [240, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]
        
        # Optimal thresholds from analysis
        self.primary_threshold = 14261
        self.secondary_threshold = 23376
        
        # Feature normalization parameters (you may need to adjust)
        self.feature_means = np.array([1e7, 1e4, 1e5])  # Approximate means
        self.feature_stds = np.array([5e6, 5e3, 5e4])   # Approximate stds
    
    def extract_key_features(self, resistance_readings):
        """Extract the most discriminative features"""
        if len(resistance_readings) != 11:
            raise ValueError(f"Expected 11 resistance readings")
        
        features = []
        
        # Primary discriminators (from analysis)
        features.append(resistance_readings[0])  # PWM 240 (strongest)
        features.append(resistance_readings[1])  # PWM 40
        features.append(resistance_readings[9])  # PWM 220
        
        # Key ratios
        if resistance_readings[1] > 0:
            features.append(resistance_readings[0] / resistance_readings[1])  # 240/40
        else:
            features.append(0)
        
        if resistance_readings[2] > 0:
            features.append(resistance_readings[9] / resistance_readings[2])  # 220/60
        else:
            features.append(0)
        
        # Normalized fingerprint
        total = sum(resistance_readings)
        if total > 0:
            features.append(resistance_readings[0] / total)  # Normalized PWM 240
            features.append(resistance_readings[3] / total)  # Normalized PWM 80
        else:
            features.extend([0, 0])
        
        return np.array(features)
    
    def classify_simple(self, resistance_readings):
        """Simple threshold-based classification (fast)"""
        primary_reading = resistance_readings[0]  # PWM 240
        secondary_reading = resistance_readings[9]  # PWM 220
        
        # Simple decision tree
        if primary_reading > self.primary_threshold:
            confidence = min(1.0, abs(primary_reading - self.primary_threshold) / (self.primary_threshold * 0.2))
            return "2-propanol", confidence
        else:
            confidence = min(1.0, abs(primary_reading - self.primary_threshold) / (self.primary_threshold * 0.2))
            return "ethanol", confidence
    
    def classify_advanced(self, resistance_readings):
        """Advanced feature-based classification"""
        features = self.extract_key_features(resistance_readings)
        
        # Multiple decision criteria
        pwm_240 = resistance_readings[0]
        pwm_220 = resistance_readings[9]
        ratio_240_40 = features[3]
        
        # Advanced decision logic
        score = 0
        
        # Criterion 1: PWM 240 threshold
        if pwm_240 > self.primary_threshold:
            score += 1
        
        # Criterion 2: PWM 220 threshold  
        if pwm_220 > self.secondary_threshold:
            score += 1
        
        # Criterion 3: Ratio analysis
        if ratio_240_40 > 800:  # Approximate threshold from analysis
            score += 1
        
        # Final classification
        if score >= 2:
            prediction = "2-propanol"
            confidence = score / 3.0
        else:
            prediction = "ethanol"
            confidence = (3 - score) / 3.0
        
        return prediction, confidence
    
    def get_feature_importance(self):
        """Return feature importance for interpretation"""
        return {
            "PWM_240": "Primary discriminator (Cohen's d = 1.657)",
            "PWM_220": "Secondary discriminator (Cohen's d = 1.316)",
            "Ratio_240_40": "Key ratio feature for robust classification",
            "Normalized_patterns": "Response fingerprints for pattern recognition"
        }

# Example usage
if __name__ == "__main__":
    classifier = GasSensorClassifier()
    
    # Example readings (replace with actual sensor data)
    example_readings = [15000000, 18000, 300000, 200000, 15000, 4000, 1200, 250, 100, 50, 25]
    
    # Simple classification
    result_simple, conf_simple = classifier.classify_simple(example_readings)
    print(f"Simple classification: {result_simple} (confidence: {conf_simple:.3f})")
    
    # Advanced classification
    result_advanced, conf_advanced = classifier.classify_advanced(example_readings)
    print(f"Advanced classification: {result_advanced} (confidence: {conf_advanced:.3f})")
    
    # Feature importance
    print("\nFeature Importance:")
    for feature, importance in classifier.get_feature_importance().items():
        print(f"  {feature}: {importance}")
