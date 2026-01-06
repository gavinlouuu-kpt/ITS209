#!/usr/bin/env python3
"""
Comprehensive analysis of static PWM-resistance data for gas sensor discrimination
PWM settings: 240, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220
Values in columns: Resistance readings in Ohms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_align_data():
    """Load and align all static data files"""
    print("Loading static data files...")
    
    # Load ethanol data
    etoh_1 = pd.read_csv('results/static/static_etoh_1.csv')
    etoh_2 = pd.read_csv('results/static/static_etoh_2.csv')
    
    # Load 2-propanol data  
    prop_1 = pd.read_csv('results/static/static_2pol_1.csv')
    prop_2 = pd.read_csv('results/static/static_2pol_2.csv')
    
    print(f"Loaded: Ethanol-1 {etoh_1.shape}, Ethanol-2 {etoh_2.shape}")
    print(f"        2-Propanol-1 {prop_1.shape}, 2-Propanol-2 {prop_2.shape}")
    
    # Get PWM columns
    pwm_cols = [col for col in etoh_1.columns if col.startswith('setting_')]
    pwm_values = [int(col.split('_')[1]) for col in pwm_cols]
    print(f"PWM settings: {pwm_values}")
    
    def align_cycling_data(df):
        """Align cycling PWM data into bins"""
        df = df.copy()
        
        # Convert timestamp and PWM columns to numeric
        df['timestamp(ms)'] = pd.to_numeric(df['timestamp(ms)'])
        for col in pwm_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Normalize timestamp
        df['timestamp(ms)'] = df['timestamp(ms)'] - df['timestamp(ms)'].min()
        
        # Group into bins (each bin should have readings from all PWM settings)
        num_settings = len(pwm_cols)
        df['bin_id'] = df.index // num_settings
        
        aligned_data = []
        for bin_id in df['bin_id'].unique():
            bin_data = df[df['bin_id'] == bin_id].copy()
            
            if len(bin_data) < num_settings:
                continue
                
            bin_timestamp = bin_data['timestamp(ms)'].iloc[0]
            bin_row = {'timestamp(ms)': bin_timestamp}
            
            # Extract the non-NaN value for each PWM setting in this bin
            for col in pwm_cols:
                non_nan_values = bin_data[col].dropna()
                if len(non_nan_values) > 0:
                    bin_row[col] = non_nan_values.iloc[0]
                else:
                    bin_row[col] = np.nan
            
            aligned_data.append(bin_row)
        
        aligned_df = pd.DataFrame(aligned_data)
        aligned_df = aligned_df.dropna()
        return aligned_df
    
    # Align all datasets
    print("Aligning cycling PWM data...")
    etoh_1_aligned = align_cycling_data(etoh_1)
    etoh_2_aligned = align_cycling_data(etoh_2)
    prop_1_aligned = align_cycling_data(prop_1)
    prop_2_aligned = align_cycling_data(prop_2)
    
    print(f"Aligned: Ethanol-1 {len(etoh_1_aligned)}, Ethanol-2 {len(etoh_2_aligned)}")
    print(f"         2-Propanol-1 {len(prop_1_aligned)}, 2-Propanol-2 {len(prop_2_aligned)}")
    
    return etoh_1_aligned, etoh_2_aligned, prop_1_aligned, prop_2_aligned, pwm_cols, pwm_values

def extract_features(df, sample_name, pwm_cols, pwm_values):
    """Extract comprehensive features from PWM-resistance data"""
    resistance_data = df[pwm_cols].values
    features = {}
    
    # 1. Basic statistical features for each PWM setting
    for i, col in enumerate(pwm_cols):
        pwm_setting = pwm_values[i]
        resistance = resistance_data[:, i]
        
        features[f'mean_pwm_{pwm_setting}'] = np.mean(resistance)
        features[f'std_pwm_{pwm_setting}'] = np.std(resistance)
        features[f'max_pwm_{pwm_setting}'] = np.max(resistance)
        features[f'min_pwm_{pwm_setting}'] = np.min(resistance)
        features[f'median_pwm_{pwm_setting}'] = np.median(resistance)
        features[f'cv_pwm_{pwm_setting}'] = np.std(resistance) / np.mean(resistance)
        features[f'q75_pwm_{pwm_setting}'] = np.percentile(resistance, 75)
        features[f'q25_pwm_{pwm_setting}'] = np.percentile(resistance, 25)
    
    # 2. Cross-PWM ratios (key for gas discrimination)
    important_ratios = [(240, 40), (100, 60), (200, 80), (180, 120), (220, 140)]
    for pwm1, pwm2 in important_ratios:
        if pwm1 in pwm_values and pwm2 in pwm_values:
            idx1 = pwm_values.index(pwm1)
            idx2 = pwm_values.index(pwm2)
            ratio = np.mean(resistance_data[:, idx1]) / np.mean(resistance_data[:, idx2])
            features[f'ratio_pwm_{pwm1}_{pwm2}'] = ratio
    
    # 3. Overall response characteristics
    features['total_mean'] = np.mean(resistance_data)
    features['total_std'] = np.std(resistance_data)
    features['total_cv'] = features['total_std'] / features['total_mean']
    features['total_range'] = np.max(resistance_data) - np.min(resistance_data)
    features['dynamic_range'] = np.max(resistance_data) / np.min(resistance_data)
    
    # 4. Response pattern features
    mean_responses = np.mean(resistance_data, axis=0)
    features['dominant_pwm'] = pwm_values[np.argmax(mean_responses)]
    features['max_response'] = np.max(mean_responses)
    features['min_response'] = np.min(mean_responses)
    
    # Normalized fingerprint
    normalized_pattern = mean_responses / np.sum(mean_responses)
    for i, pwm in enumerate(pwm_values):
        features[f'fingerprint_pwm_{pwm}'] = normalized_pattern[i]
    
    # 5. PWM-resistance relationship
    slope, intercept, r_value, p_value, std_err = stats.linregress(pwm_values, mean_responses)
    features['pwm_response_slope'] = slope
    features['pwm_response_r2'] = r_value**2
    features['pwm_response_correlation'] = np.corrcoef(pwm_values, mean_responses)[0, 1]
    
    # 6. Temporal stability
    features['temporal_stability'] = np.mean([np.std(resistance_data[:, i]) for i in range(len(pwm_values))])
    features['max_temporal_variation'] = np.max([np.std(resistance_data[:, i]) for i in range(len(pwm_values))])
    
    # 7. Higher-order moments
    features['skewness'] = stats.skew(resistance_data.flatten())
    features['kurtosis'] = stats.kurtosis(resistance_data.flatten())
    
    features['sample'] = sample_name
    return features

def calculate_discrimination_power(etoh_data, prop_data, pwm_cols):
    """Calculate how well each PWM setting can discriminate between samples"""
    discrimination_results = {}
    
    for col in pwm_cols:
        pwm_setting = col.split('_')[1]
        
        etoh_values = etoh_data[col].values
        prop_values = prop_data[col].values
        
        # Basic statistics
        etoh_mean, etoh_std = np.mean(etoh_values), np.std(etoh_values)
        prop_mean, prop_std = np.mean(prop_values), np.std(prop_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(etoh_values)-1)*etoh_std**2 + 
                             (len(prop_values)-1)*prop_std**2) / 
                            (len(etoh_values) + len(prop_values) - 2))
        cohens_d = abs(etoh_mean - prop_mean) / pooled_std if pooled_std > 0 else 0
        
        # Signal-to-noise ratio
        signal = abs(etoh_mean - prop_mean)
        noise = (etoh_std + prop_std) / 2
        snr = signal / noise if noise > 0 else 0
        
        # Statistical significance
        t_stat, p_value = stats.ttest_ind(etoh_values, prop_values)
        
        # Overlap coefficient (simplified)
        overlap = min(etoh_mean + etoh_std, prop_mean + prop_std) - max(etoh_mean - etoh_std, prop_mean - prop_std)
        total_range = max(etoh_mean + etoh_std, prop_mean + prop_std) - min(etoh_mean - etoh_std, prop_mean - prop_std)
        overlap_coeff = max(0, overlap / total_range) if total_range > 0 else 1
        
        discrimination_results[f'pwm_{pwm_setting}'] = {
            'cohens_d': cohens_d,
            'snr': snr,
            'p_value': p_value,
            'overlap_coeff': overlap_coeff,
            'etoh_mean': etoh_mean,
            'prop_mean': prop_mean,
            'etoh_std': etoh_std,
            'prop_std': prop_std,
            'discriminability': cohens_d * (1 - overlap_coeff)  # Combined metric
        }
    
    return discrimination_results

def create_visualizations(etoh_data_list, prop_data_list, pwm_cols, pwm_values):
    """Create comprehensive visualizations"""
    
    # Combine all data
    all_etoh = pd.concat(etoh_data_list, ignore_index=True)
    all_prop = pd.concat(prop_data_list, ignore_index=True)
    
    # Calculate statistics
    etoh_means = all_etoh[pwm_cols].mean()
    prop_means = all_prop[pwm_cols].mean()
    etoh_stds = all_etoh[pwm_cols].std()
    prop_stds = all_prop[pwm_cols].std()
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Comprehensive PWM-Resistance Analysis: Ethanol vs 2-Propanol', fontsize=16)
    
    # Plot 1: Mean resistance patterns
    ax = axes[0, 0]
    ax.errorbar(pwm_values, etoh_means.values, yerr=etoh_stds.values, 
                marker='o', linewidth=2, markersize=8, label='Ethanol', capsize=5)
    ax.errorbar(pwm_values, prop_means.values, yerr=prop_stds.values, 
                marker='s', linewidth=2, markersize=8, label='2-Propanol', capsize=5)
    ax.set_xlabel('PWM Setting')
    ax.set_ylabel('Mean Resistance (Ω)')
    ax.set_title('Mean Resistance vs PWM Setting')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Normalized fingerprint
    ax = axes[0, 1]
    etoh_norm = etoh_means.values / np.sum(etoh_means.values)
    prop_norm = prop_means.values / np.sum(prop_means.values)
    ax.plot(pwm_values, etoh_norm, 'o-', linewidth=2, markersize=8, label='Ethanol')
    ax.plot(pwm_values, prop_norm, 's-', linewidth=2, markersize=8, label='2-Propanol')
    ax.set_xlabel('PWM Setting')
    ax.set_ylabel('Normalized Response')
    ax.set_title('Response Fingerprint (Normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Coefficient of variation
    ax = axes[0, 2]
    etoh_cv = etoh_stds.values / etoh_means.values
    prop_cv = prop_stds.values / prop_means.values
    ax.plot(pwm_values, etoh_cv, 'o-', linewidth=2, markersize=8, label='Ethanol')
    ax.plot(pwm_values, prop_cv, 's-', linewidth=2, markersize=8, label='2-Propanol')
    ax.set_xlabel('PWM Setting')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Response Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Key resistance ratios
    ax = axes[1, 0]
    key_pairs = [(240, 40), (100, 60), (200, 80)]
    ratios_etoh, ratios_prop, labels = [], [], []
    
    for pwm1, pwm2 in key_pairs:
        col1, col2 = f'setting_{pwm1}_ohms', f'setting_{pwm2}_ohms'
        if col1 in pwm_cols and col2 in pwm_cols:
            ratio_etoh = etoh_means[col1] / etoh_means[col2]
            ratio_prop = prop_means[col1] / prop_means[col2]
            ratios_etoh.append(ratio_etoh)
            ratios_prop.append(ratio_prop)
            labels.append(f'{pwm1}/{pwm2}')
    
    x_pos = np.arange(len(labels))
    width = 0.35
    ax.bar(x_pos - width/2, ratios_etoh, width, label='Ethanol', alpha=0.8)
    ax.bar(x_pos + width/2, ratios_prop, width, label='2-Propanol', alpha=0.8)
    ax.set_xlabel('PWM Ratios')
    ax.set_ylabel('Resistance Ratio')
    ax.set_title('Key Resistance Ratios')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Discrimination power
    ax = axes[1, 1]
    discrimination = calculate_discrimination_power(all_etoh, all_prop, pwm_cols)
    pwm_labels = [key.split('_')[1] for key in discrimination.keys()]
    cohens_d_values = [discrimination[key]['cohens_d'] for key in discrimination.keys()]
    
    bars = ax.bar(pwm_labels, cohens_d_values, alpha=0.8, color='skyblue')
    ax.set_xlabel('PWM Setting')
    ax.set_ylabel("Cohen's d (Effect Size)")
    ax.set_title('Discrimination Power by PWM')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Highlight best discriminators
    max_d = max(cohens_d_values)
    for i, (bar, d) in enumerate(zip(bars, cohens_d_values)):
        if d > max_d * 0.8:  # Top 20%
            bar.set_color('orange')
    
    # Plot 6: Time series comparison
    ax = axes[1, 2]
    n_samples = min(50, len(all_etoh))
    for pwm in [240, 100, 60]:  # Show key PWM settings
        col = f'setting_{pwm}_ohms'
        if col in pwm_cols:
            ax.plot(all_etoh[col][:n_samples], alpha=0.7, label=f'ETH-{pwm}')
            ax.plot(all_prop[col][:n_samples], alpha=0.7, linestyle='--', label=f'PROP-{pwm}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Resistance (Ω)')
    ax.set_title('Time Series Comparison')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Distribution of most discriminative PWM
    ax = axes[2, 0]
    best_pwm = max(discrimination.keys(), key=lambda x: discrimination[x]['cohens_d'])
    best_col = f'setting_{best_pwm.split("_")[1]}_ohms'
    
    ax.hist(all_etoh[best_col], bins=30, alpha=0.7, label='Ethanol', density=True, color='blue')
    ax.hist(all_prop[best_col], bins=30, alpha=0.7, label='2-Propanol', density=True, color='red')
    ax.set_xlabel(f'Resistance (Ω) - {best_pwm}')
    ax.set_ylabel('Density')
    ax.set_title(f'Best Discriminator: {best_pwm}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Feature space PCA
    ax = axes[2, 1]
    # Create feature matrix for PCA
    etoh_features = [extract_features(data, 'ethanol', pwm_cols, pwm_values) for data in etoh_data_list]
    prop_features = [extract_features(data, '2-propanol', pwm_cols, pwm_values) for data in prop_data_list]
    
    all_features = etoh_features + prop_features
    feature_df = pd.DataFrame(all_features)
    feature_cols = [col for col in feature_df.columns if col != 'sample']
    
    X = feature_df[feature_cols].values
    y = feature_df['sample'].values
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    colors = ['blue' if label == 'ethanol' else 'red' for label in y]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7, s=150)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title('PCA - Sample Separation')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Ethanol'),
                      Patch(facecolor='red', label='2-Propanol')]
    ax.legend(handles=legend_elements)
    
    # Plot 9: Correlation heatmap
    ax = axes[2, 2]
    correlation_data = pd.concat([all_etoh[pwm_cols], all_prop[pwm_cols]])
    corr_matrix = correlation_data.corr()
    
    # Create abbreviated labels for better visibility
    short_labels = [f'PWM{col.split("_")[1]}' for col in pwm_cols]
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, 
                fmt='.2f', ax=ax, xticklabels=short_labels, yticklabels=short_labels)
    ax.set_title('PWM Setting Correlations')
    
    plt.tight_layout()
    plt.show()
    
    return discrimination, feature_df

def machine_learning_analysis(feature_df):
    """Perform machine learning classification analysis"""
    print("\n" + "="*60)
    print("MACHINE LEARNING CLASSIFICATION")
    print("="*60)
    
    feature_cols = [col for col in feature_df.columns if col != 'sample']
    X = feature_df[feature_cols].values
    y = feature_df['sample'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Leave-one-out cross-validation (since we have small dataset)
    predictions = []
    for i in range(len(X_scaled)):
        train_idx = [j for j in range(len(X_scaled)) if j != i]
        test_idx = [i]
        
        rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_cv.fit(X_scaled[train_idx], y[train_idx])
        pred = rf_cv.predict(X_scaled[test_idx])
        predictions.append(pred[0])
    
    accuracy = accuracy_score(y, predictions)
    print(f"Leave-One-Out Cross-Validation Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y, predictions))
    
    # Feature importance
    feature_importance = rf.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-15:]  # Top 15 features
    
    print("\nTop 15 Most Important Features:")
    for i, idx in enumerate(reversed(top_features_idx)):
        print(f"{i+1:2d}. {feature_cols[idx]:35s} {feature_importance[idx]:.4f}")
    
    return rf, feature_importance, feature_cols

def main():
    """Main analysis function"""
    print("="*80)
    print("COMPREHENSIVE PWM-RESISTANCE ANALYSIS")
    print("Static Gas Sensor Data: Ethanol vs 2-Propanol")
    print("="*80)
    
    # Load and align data
    etoh_1, etoh_2, prop_1, prop_2, pwm_cols, pwm_values = load_and_align_data()
    
    # Extract features
    print("\nExtracting features...")
    etoh_1_features = extract_features(etoh_1, 'ethanol_1', pwm_cols, pwm_values)
    etoh_2_features = extract_features(etoh_2, 'ethanol_2', pwm_cols, pwm_values)
    prop_1_features = extract_features(prop_1, 'propanol_1', pwm_cols, pwm_values)
    prop_2_features = extract_features(prop_2, 'propanol_2', pwm_cols, pwm_values)
    
    all_features = [etoh_1_features, etoh_2_features, prop_1_features, prop_2_features]
    feature_df = pd.DataFrame(all_features)
    
    print(f"Total features extracted: {len([col for col in feature_df.columns if col != 'sample'])}")
    
    # Sample comparison
    print("\nSample Comparison:")
    comparison_cols = ['sample', 'total_mean', 'total_cv', 'dominant_pwm', 'dynamic_range', 'pwm_response_correlation']
    print(feature_df[comparison_cols].round(3))
    
    # Discrimination analysis
    print("\n" + "="*60)
    print("DISCRIMINATION ANALYSIS")
    print("="*60)
    
    all_etoh = pd.concat([etoh_1, etoh_2], ignore_index=True)
    all_prop = pd.concat([prop_1, prop_2], ignore_index=True)
    
    discrimination = calculate_discrimination_power(all_etoh, all_prop, pwm_cols)
    
    print("\nDiscrimination Power by PWM Setting:")
    print("-" * 70)
    print(f"{'PWM':>8} {'Cohen\\'s d':>10} {'SNR':>8} {'Overlap':>8} {'Combined':>10} {'p-value':>12}")
    print("-" * 70)
    
    for pwm, metrics in sorted(discrimination.items(), key=lambda x: x[1]['discriminability'], reverse=True):
        print(f"{pwm:>8} {metrics['cohens_d']:>10.3f} {metrics['snr']:>8.3f} "
              f"{metrics['overlap_coeff']:>8.3f} {metrics['discriminability']:>10.3f} "
              f"{metrics['p_value']:>12.2e}")
    
    # Find best discriminators
    best_overall = max(discrimination.keys(), key=lambda x: discrimination[x]['discriminability'])
    best_cohens = max(discrimination.keys(), key=lambda x: discrimination[x]['cohens_d'])
    
    print(f"\nBest overall discriminator: {best_overall}")
    print(f"Best by Cohen's d: {best_cohens}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    discrimination, feature_df = create_visualizations([etoh_1, etoh_2], [prop_1, prop_2], pwm_cols, pwm_values)
    
    # Machine learning analysis
    rf, feature_importance, feature_cols = machine_learning_analysis(feature_df)
    
    # Recommendations
    print("\n" + "="*60)
    print("FEATURE SELECTION RECOMMENDATIONS")
    print("="*60)
    
    print("1. MOST DISCRIMINATIVE PWM SETTINGS:")
    top_pwm = sorted(discrimination.items(), key=lambda x: x[1]['discriminability'], reverse=True)[:5]
    for i, (pwm, metrics) in enumerate(top_pwm):
        print(f"   {i+1}. {pwm} (Combined score: {metrics['discriminability']:.3f})")
    
    print("\n2. RECOMMENDED FEATURE TYPES:")
    feature_types = {}
    for col in feature_cols:
        if feature_importance[feature_cols.index(col)] > 0.01:  # Only important features
            if 'ratio' in col:
                feature_types['ratios'] = feature_types.get('ratios', 0) + 1
            elif 'fingerprint' in col:
                feature_types['fingerprints'] = feature_types.get('fingerprints', 0) + 1
            elif 'mean' in col:
                feature_types['means'] = feature_types.get('means', 0) + 1
            elif 'std' in col or 'cv' in col:
                feature_types['variability'] = feature_types.get('variability', 0) + 1
    
    for ftype, count in sorted(feature_types.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {ftype.capitalize()}: {count} important features")
    
    print("\n3. IMPLEMENTATION RECOMMENDATIONS:")
    print("   - Focus on PWM settings with high discrimination scores")
    print("   - Use resistance ratios for robust gas discrimination")
    print("   - Include normalized fingerprints for pattern recognition")
    print("   - Apply logarithmic scaling for resistance values")
    print("   - Consider ensemble methods for classification")
    
    return feature_df, discrimination, rf

if __name__ == "__main__":
    features, discrimination, model = main() 