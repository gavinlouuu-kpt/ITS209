import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_static_data():
    """
    Load all static data files
    """
    # Load ethanol data
    etoh_1 = pd.read_csv('/home/gavin/Documents/PlatformIO/Projects/async_ads/results/static/static_etoh_1.csv')
    etoh_2 = pd.read_csv('/home/gavin/Documents/PlatformIO/Projects/async_ads/results/static/static_etoh_2.csv')
    
    # Load 2-propanol data
    prop_1 = pd.read_csv('/home/gavin/Documents/PlatformIO/Projects/async_ads/results/static/static_2pol_1.csv')
    prop_2 = pd.read_csv('/home/gavin/Documents/PlatformIO/Projects/async_ads/results/static/static_2pol_2.csv')
    
    return etoh_1, etoh_2, prop_1, prop_2

def bin_align_static(df):
    """
    Align the cycling sensor data into bins where each bin contains one reading from each PWM setting
    PWM settings cycle: 240, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220
    """
    # Make a copy
    df = df.copy()
    
    # Convert timestamp to numeric
    df['timestamp(ms)'] = pd.to_numeric(df['timestamp(ms)'])
    
    # Normalize timestamp to start at 0
    df['timestamp(ms)'] = df['timestamp(ms)'] - df['timestamp(ms)'].min()
    
    # Get PWM setting columns (excluding timestamp)
    pwm_cols = [col for col in df.columns if col.startswith('setting_')]
    
    # Convert setting columns to numeric, replacing empty strings with NaN
    for col in pwm_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Number of PWM settings
    num_settings = len(pwm_cols)
    
    # Group rows into bins based on the cycling pattern
    df['bin_id'] = df.index // num_settings
    
    # Create aligned dataframe
    aligned_data = []
    
    for bin_id in df['bin_id'].unique():
        bin_data = df[df['bin_id'] == bin_id].copy()
        
        # Skip incomplete bins
        if len(bin_data) < num_settings:
            continue
        
        # Take the first timestamp of the bin as the bin timestamp
        bin_timestamp = bin_data['timestamp(ms)'].iloc[0]
        
        # Create a row for this bin
        bin_row = {'timestamp(ms)': bin_timestamp}
        
        # For each PWM setting, find the non-NaN value in this bin
        for col in pwm_cols:
            non_nan_values = bin_data[col].dropna()
            if len(non_nan_values) > 0:
                bin_row[col] = non_nan_values.iloc[0]
            else:
                bin_row[col] = np.nan
        
        aligned_data.append(bin_row)
    
    # Create aligned dataframe and drop rows with NaN
    aligned_df = pd.DataFrame(aligned_data)
    aligned_df = aligned_df.dropna()
    
    return aligned_df

def extract_pwm_features(df, sample_name):
    """
    Extract comprehensive features from PWM-resistance data
    """
    # Get PWM setting columns
    pwm_cols = [col for col in df.columns if col.startswith('setting_')]
    
    # Extract PWM values from column names (240, 40, 60, etc.)
    pwm_values = [int(col.split('_')[1]) for col in pwm_cols]
    
    # Convert to numpy array for easier manipulation
    resistance_data = df[pwm_cols].values
    
    features = {}
    
    # 1. Statistical features for each PWM setting
    for i, col in enumerate(pwm_cols):
        pwm_setting = pwm_values[i]
        resistance_readings = resistance_data[:, i]
        
        features[f'mean_pwm_{pwm_setting}'] = np.mean(resistance_readings)
        features[f'std_pwm_{pwm_setting}'] = np.std(resistance_readings)
        features[f'max_pwm_{pwm_setting}'] = np.max(resistance_readings)
        features[f'min_pwm_{pwm_setting}'] = np.min(resistance_readings)
        features[f'median_pwm_{pwm_setting}'] = np.median(resistance_readings)
        features[f'range_pwm_{pwm_setting}'] = np.max(resistance_readings) - np.min(resistance_readings)
        features[f'cv_pwm_{pwm_setting}'] = np.std(resistance_readings) / np.mean(resistance_readings)
        features[f'q75_pwm_{pwm_setting}'] = np.percentile(resistance_readings, 75)
        features[f'q25_pwm_{pwm_setting}'] = np.percentile(resistance_readings, 25)
        features[f'iqr_pwm_{pwm_setting}'] = np.percentile(resistance_readings, 75) - np.percentile(resistance_readings, 25)
    
    # 2. Cross-PWM ratios (important for gas sensor discrimination)
    for i in range(len(pwm_cols)):
        for j in range(i+1, len(pwm_cols)):
            pwm1 = pwm_values[i]
            pwm2 = pwm_values[j]
            
            # Calculate mean ratio
            mean_ratio = np.mean(resistance_data[:, i]) / np.mean(resistance_data[:, j])
            features[f'ratio_pwm_{pwm1}_{pwm2}'] = mean_ratio
            
            # Calculate max ratio
            max_ratio = np.max(resistance_data[:, i]) / np.max(resistance_data[:, j])
            features[f'max_ratio_pwm_{pwm1}_{pwm2}'] = max_ratio
    
    # 3. Overall response patterns
    features['total_mean_resistance'] = np.mean(resistance_data)
    features['total_std_resistance'] = np.std(resistance_data)
    features['total_cv'] = features['total_std_resistance'] / features['total_mean_resistance']
    features['total_max'] = np.max(resistance_data)
    features['total_min'] = np.min(resistance_data)
    features['total_range'] = features['total_max'] - features['total_min']
    
    # 4. Principal response characteristics
    max_responses = np.max(resistance_data, axis=0)
    mean_responses = np.mean(resistance_data, axis=0)
    
    features['dominant_pwm_setting'] = pwm_values[np.argmax(max_responses)]
    features['max_response_value'] = np.max(max_responses)
    features['min_response_value'] = np.min(mean_responses)
    features['response_dynamic_range'] = np.max(mean_responses) / np.min(mean_responses)
    
    # 5. Response stability (variance across time for each PWM)
    for i, col in enumerate(pwm_cols):
        pwm_setting = pwm_values[i]
        features[f'stability_pwm_{pwm_setting}'] = np.var(resistance_data[:, i])
        features[f'rel_stability_pwm_{pwm_setting}'] = np.var(resistance_data[:, i]) / np.mean(resistance_data[:, i])**2
    
    # 6. Response fingerprint (normalized pattern)
    normalized_pattern = mean_responses / np.sum(mean_responses)
    for i, col in enumerate(pwm_cols):
        pwm_setting = pwm_values[i]
        features[f'fingerprint_pwm_{pwm_setting}'] = normalized_pattern[i]
    
    # 7. PWM-response relationship features
    # Correlation between PWM setting and response
    pwm_response_corr = np.corrcoef(pwm_values, mean_responses)[0, 1]
    features['pwm_response_correlation'] = pwm_response_corr
    
    # Slope of PWM vs response
    slope, intercept, r_value, p_value, std_err = stats.linregress(pwm_values, mean_responses)
    features['pwm_response_slope'] = slope
    features['pwm_response_r2'] = r_value**2
    
    # 8. Temporal features
    features['temporal_stability'] = np.mean([np.std(resistance_data[:, i]) for i in range(len(pwm_cols))])
    features['temporal_drift'] = np.mean([np.abs(resistance_data[-10:, i].mean() - resistance_data[:10, i].mean()) 
                                        for i in range(len(pwm_cols))])
    
    # 9. High-order statistical moments
    features['skewness_total'] = stats.skew(resistance_data.flatten())
    features['kurtosis_total'] = stats.kurtosis(resistance_data.flatten())
    
    # Add sample identifier
    features['sample'] = sample_name
    
    return features

def calculate_discrimination_power(etoh_data, prop_data):
    """
    Calculate discrimination power for each PWM setting
    """
    pwm_cols = [col for col in etoh_data.columns if col.startswith('setting_')]
    discrimination_metrics = {}
    
    for col in pwm_cols:
        pwm_setting = col.split('_')[1]
        
        etoh_values = etoh_data[col].values
        prop_values = prop_data[col].values
        
        # Statistical separation metrics
        etoh_mean = np.mean(etoh_values)
        prop_mean = np.mean(prop_values)
        etoh_std = np.std(etoh_values)
        prop_std = np.std(prop_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(etoh_values)-1)*etoh_std**2 + 
                             (len(prop_values)-1)*prop_std**2) / 
                            (len(etoh_values) + len(prop_values) - 2))
        cohens_d = abs(etoh_mean - prop_mean) / pooled_std if pooled_std > 0 else 0
        
        # Signal-to-noise ratio
        signal = abs(etoh_mean - prop_mean)
        noise = (etoh_std + prop_std) / 2
        snr = signal / noise if noise > 0 else 0
        
        # Separation index
        separation = abs(etoh_mean - prop_mean) / (etoh_std + prop_std) if (etoh_std + prop_std) > 0 else 0
        
        # Statistical significance (t-test)
        t_stat, p_value = stats.ttest_ind(etoh_values, prop_values)
        
        discrimination_metrics[f'pwm_{pwm_setting}'] = {
            'cohens_d': cohens_d,
            'snr': snr,
            'separation': separation,
            'p_value': p_value,
            'etoh_mean': etoh_mean,
            'prop_mean': prop_mean,
            'etoh_std': etoh_std,
            'prop_std': prop_std
        }
    
    return discrimination_metrics

def plot_comprehensive_analysis(etoh_data_list, prop_data_list):
    """
    Create comprehensive visualization of the PWM-resistance patterns
    """
    # Combine all data
    all_etoh = pd.concat(etoh_data_list, ignore_index=True)
    all_prop = pd.concat(prop_data_list, ignore_index=True)
    
    pwm_cols = [col for col in all_etoh.columns if col.startswith('setting_')]
    pwm_values = [int(col.split('_')[1]) for col in pwm_cols]
    
    # Calculate statistics
    etoh_means = all_etoh[pwm_cols].mean()
    prop_means = all_prop[pwm_cols].mean()
    etoh_stds = all_etoh[pwm_cols].std()
    prop_stds = all_prop[pwm_cols].std()
    
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Mean resistance vs PWM setting
    plt.subplot(3, 4, 1)
    plt.errorbar(pwm_values, etoh_means.values, yerr=etoh_stds.values, 
                marker='o', linewidth=2, markersize=8, label='Ethanol', capsize=5)
    plt.errorbar(pwm_values, prop_means.values, yerr=prop_stds.values, 
                marker='s', linewidth=2, markersize=8, label='2-Propanol', capsize=5)
    plt.xlabel('PWM Setting')
    plt.ylabel('Mean Resistance (Ω)')
    plt.title('Mean Resistance vs PWM Setting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Normalized fingerprint
    plt.subplot(3, 4, 2)
    etoh_norm = etoh_means.values / np.sum(etoh_means.values)
    prop_norm = prop_means.values / np.sum(prop_means.values)
    plt.plot(pwm_values, etoh_norm, 'o-', linewidth=2, markersize=8, label='Ethanol')
    plt.plot(pwm_values, prop_norm, 's-', linewidth=2, markersize=8, label='2-Propanol')
    plt.xlabel('PWM Setting')
    plt.ylabel('Normalized Response')
    plt.title('Normalized Response Fingerprint')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Coefficient of variation
    plt.subplot(3, 4, 3)
    etoh_cv = etoh_stds.values / etoh_means.values
    prop_cv = prop_stds.values / prop_means.values
    plt.plot(pwm_values, etoh_cv, 'o-', linewidth=2, markersize=8, label='Ethanol')
    plt.plot(pwm_values, prop_cv, 's-', linewidth=2, markersize=8, label='2-Propanol')
    plt.xlabel('PWM Setting')
    plt.ylabel('Coefficient of Variation')
    plt.title('Response Variability (CV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Response ratios (key discriminative ratios)
    plt.subplot(3, 4, 4)
    key_ratios_etoh = []
    key_ratios_prop = []
    ratio_labels = []
    
    # Select a few key ratios
    key_pairs = [(240, 40), (100, 60), (200, 80), (180, 120)]
    for pwm1, pwm2 in key_pairs:
        if f'setting_{pwm1}_ohms' in pwm_cols and f'setting_{pwm2}_ohms' in pwm_cols:
            ratio_etoh = etoh_means[f'setting_{pwm1}_ohms'] / etoh_means[f'setting_{pwm2}_ohms']
            ratio_prop = prop_means[f'setting_{pwm1}_ohms'] / prop_means[f'setting_{pwm2}_ohms']
            key_ratios_etoh.append(ratio_etoh)
            key_ratios_prop.append(ratio_prop)
            ratio_labels.append(f'{pwm1}/{pwm2}')
    
    x_pos = np.arange(len(ratio_labels))
    width = 0.35
    plt.bar(x_pos - width/2, key_ratios_etoh, width, label='Ethanol', alpha=0.8)
    plt.bar(x_pos + width/2, key_ratios_prop, width, label='2-Propanol', alpha=0.8)
    plt.xlabel('PWM Ratios')
    plt.ylabel('Resistance Ratio')
    plt.title('Key Resistance Ratios')
    plt.xticks(x_pos, ratio_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Heatmap of resistance patterns
    plt.subplot(3, 4, 5)
    combined_data = np.array([etoh_means.values, prop_means.values])
    sns.heatmap(combined_data, 
                xticklabels=[f'PWM{pwm}' for pwm in pwm_values],
                yticklabels=['Ethanol', '2-Propanol'],
                annot=False, cmap='viridis', cbar_kws={'label': 'Resistance (Ω)'})
    plt.title('Resistance Pattern Heatmap')
    
    # Plot 6: Time series comparison (first 100 samples)
    plt.subplot(3, 4, 6)
    n_samples = min(100, len(all_etoh))
    for i, pwm in enumerate([240, 100, 60]):  # Show a few representative PWM settings
        if f'setting_{pwm}_ohms' in pwm_cols:
            plt.plot(all_etoh[f'setting_{pwm}_ohms'][:n_samples], 
                    alpha=0.7, label=f'ETH PWM{pwm}')
            plt.plot(all_prop[f'setting_{pwm}_ohms'][:n_samples], 
                    alpha=0.7, linestyle='--', label=f'PROP PWM{pwm}')
    plt.xlabel('Sample Index')
    plt.ylabel('Resistance (Ω)')
    plt.title('Time Series Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Distribution comparison for most discriminative PWM
    discrimination_metrics = calculate_discrimination_power(all_etoh, all_prop)
    best_pwm = max(discrimination_metrics.keys(), 
                  key=lambda x: discrimination_metrics[x]['cohens_d'])
    best_pwm_col = f'setting_{best_pwm.split("_")[1]}_ohms'
    
    plt.subplot(3, 4, 7)
    plt.hist(all_etoh[best_pwm_col], bins=30, alpha=0.7, label='Ethanol', density=True)
    plt.hist(all_prop[best_pwm_col], bins=30, alpha=0.7, label='2-Propanol', density=True)
    plt.xlabel(f'Resistance (Ω) - {best_pwm}')
    plt.ylabel('Density')
    plt.title(f'Most Discriminative PWM: {best_pwm}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Correlation matrix
    plt.subplot(3, 4, 8)
    corr_data = pd.concat([all_etoh[pwm_cols], all_prop[pwm_cols]])
    correlation_matrix = corr_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('PWM Setting Correlations')
    
    # Plot 9: Discrimination power
    plt.subplot(3, 4, 9)
    pwm_labels = [key.split('_')[1] for key in discrimination_metrics.keys()]
    cohens_d_values = [discrimination_metrics[key]['cohens_d'] for key in discrimination_metrics.keys()]
    
    plt.bar(pwm_labels, cohens_d_values, alpha=0.8)
    plt.xlabel('PWM Setting')
    plt.ylabel("Cohen's d")
    plt.title('Discrimination Power by PWM')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 10: PCA visualization
    plt.subplot(3, 4, 10)
    # Prepare data for PCA
    etoh_features_list = [extract_pwm_features(data, 'ethanol') for data in etoh_data_list]
    prop_features_list = [extract_pwm_features(data, '2-propanol') for data in prop_data_list]
    
    all_features = etoh_features_list + prop_features_list
    feature_df = pd.DataFrame(all_features)
    
    # Separate features and labels
    feature_cols = [col for col in feature_df.columns if col != 'sample']
    X = feature_df[feature_cols].values
    y = feature_df['sample'].values
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    colors = ['red' if label == 'ethanol' else 'blue' for label in y]
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7, s=100)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA - Sample Separation')
    plt.grid(True, alpha=0.3)
    
    # Create custom legend
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='Ethanol')
    blue_patch = mpatches.Patch(color='blue', label='2-Propanol')
    plt.legend(handles=[red_patch, blue_patch])
    
    # Plot 11: Feature importance from Random Forest
    plt.subplot(3, 4, 11)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Get top 10 features
    feature_importance = rf.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:]
    
    plt.barh(range(len(top_features_idx)), 
             feature_importance[top_features_idx])
    plt.yticks(range(len(top_features_idx)), 
               [feature_cols[i][:20] for i in top_features_idx])  # Truncate long names
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Discriminative Features')
    
    # Plot 12: PWM response relationship
    plt.subplot(3, 4, 12)
    plt.scatter(pwm_values, etoh_means.values, s=100, alpha=0.7, label='Ethanol')
    plt.scatter(pwm_values, prop_means.values, s=100, alpha=0.7, label='2-Propanol')
    
    # Fit trend lines
    etoh_slope, etoh_intercept, etoh_r, _, _ = stats.linregress(pwm_values, etoh_means.values)
    prop_slope, prop_intercept, prop_r, _, _ = stats.linregress(pwm_values, prop_means.values)
    
    x_trend = np.array(pwm_values)
    plt.plot(x_trend, etoh_slope * x_trend + etoh_intercept, 'r--', 
             label=f'ETH trend (R²={etoh_r**2:.3f})')
    plt.plot(x_trend, prop_slope * x_trend + prop_intercept, 'b--', 
             label=f'PROP trend (R²={prop_r**2:.3f})')
    
    plt.xlabel('PWM Setting')
    plt.ylabel('Mean Resistance (Ω)')
    plt.title('PWM-Resistance Relationship')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return discrimination_metrics, feature_df

def main():
    """
    Main analysis function
    """
    print("="*80)
    print("COMPREHENSIVE STATIC DATA ANALYSIS")
    print("PWM Settings vs Resistance Readings (Ohms)")
    print("="*80)
    
    # Load data
    print("Loading static data files...")
    etoh_1, etoh_2, prop_1, prop_2 = load_static_data()
    
    # Bin align all datasets
    print("Aligning PWM cycling data...")
    etoh_1_aligned = bin_align_static(etoh_1)
    etoh_2_aligned = bin_align_static(etoh_2)
    prop_1_aligned = bin_align_static(prop_1)
    prop_2_aligned = bin_align_static(prop_2)
    
    print(f"Ethanol 1: {len(etoh_1_aligned)} aligned samples")
    print(f"Ethanol 2: {len(etoh_2_aligned)} aligned samples")
    print(f"2-Propanol 1: {len(prop_1_aligned)} aligned samples")
    print(f"2-Propanol 2: {len(prop_2_aligned)} aligned samples")
    
    # Extract features
    print("\nExtracting comprehensive features...")
    etoh_1_features = extract_pwm_features(etoh_1_aligned, 'ethanol_1')
    etoh_2_features = extract_pwm_features(etoh_2_aligned, 'ethanol_2')
    prop_1_features = extract_pwm_features(prop_1_aligned, 'propanol_1')
    prop_2_features = extract_pwm_features(prop_2_aligned, 'propanol_2')
    
    # Create feature comparison
    all_features = [etoh_1_features, etoh_2_features, prop_1_features, prop_2_features]
    feature_df = pd.DataFrame(all_features)
    
    print("\nFeature Summary:")
    print(f"Total features extracted: {len([col for col in feature_df.columns if col != 'sample'])}")
    print("\nSample comparison:")
    print(feature_df[['sample', 'total_mean_resistance', 'total_cv', 'dominant_pwm_setting', 
                     'response_dynamic_range', 'pwm_response_correlation']].round(3))
    
    # Calculate discrimination metrics
    print("\n" + "="*60)
    print("DISCRIMINATION ANALYSIS")
    print("="*60)
    
    # Combine datasets for discrimination analysis
    all_etoh = pd.concat([etoh_1_aligned, etoh_2_aligned], ignore_index=True)
    all_prop = pd.concat([prop_1_aligned, prop_2_aligned], ignore_index=True)
    
    discrimination_metrics = calculate_discrimination_power(all_etoh, all_prop)
    
    print("\nDiscrimination Power by PWM Setting:")
    print("-" * 50)
    for pwm, metrics in discrimination_metrics.items():
        print(f"{pwm:>8}: Cohen's d={metrics['cohens_d']:6.3f}, "
              f"SNR={metrics['snr']:6.3f}, p-value={metrics['p_value']:8.2e}")
    
    # Find best discriminative PWM settings
    best_cohens = max(discrimination_metrics.keys(), 
                     key=lambda x: discrimination_metrics[x]['cohens_d'])
    best_snr = max(discrimination_metrics.keys(), 
                  key=lambda x: discrimination_metrics[x]['snr'])
    
    print(f"\nBest PWM by Cohen's d: {best_cohens} (d = {discrimination_metrics[best_cohens]['cohens_d']:.3f})")
    print(f"Best PWM by SNR: {best_snr} (SNR = {discrimination_metrics[best_snr]['snr']:.3f})")
    
    # Machine learning classification
    print("\n" + "="*60)
    print("MACHINE LEARNING CLASSIFICATION")
    print("="*60)
    
    # Prepare ML features
    etoh_data_list = [etoh_1_aligned, etoh_2_aligned]
    prop_data_list = [prop_1_aligned, prop_2_aligned]
    
    etoh_ml_features = [extract_pwm_features(data, 'ethanol') for data in etoh_data_list]
    prop_ml_features = [extract_pwm_features(data, '2-propanol') for data in prop_data_list]
    
    ml_features = etoh_ml_features + prop_ml_features
    ml_df = pd.DataFrame(ml_features)
    
    feature_cols = [col for col in ml_df.columns if col != 'sample']
    X = ml_df[feature_cols].values
    y = ml_df['sample'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Random Forest classification
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Cross-validation prediction (leave-one-out for small dataset)
    predictions = []
    for i in range(len(X_scaled)):
        train_idx = [j for j in range(len(X_scaled)) if j != i]
        test_idx = [i]
        
        rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_cv.fit(X_scaled[train_idx], y[train_idx])
        pred = rf_cv.predict(X_scaled[test_idx])
        predictions.append(pred[0])
    
    print("Random Forest Classification (Leave-One-Out CV):")
    print(classification_report(y, predictions))
    
    # Feature importance
    feature_importance = rf.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:]
    
    print("\nTop 10 Most Important Features:")
    for i, idx in enumerate(reversed(top_features_idx)):
        print(f"{i+1:2d}. {feature_cols[idx]:40s} {feature_importance[idx]:.4f}")
    
    # Create comprehensive visualizations
    print("\nGenerating comprehensive visualizations...")
    discrimination_metrics, feature_df = plot_comprehensive_analysis(etoh_data_list, prop_data_list)
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR FEATURE SELECTION")
    print("="*60)
    
    print("1. MOST DISCRIMINATIVE PWM SETTINGS:")
    top_pwm = sorted(discrimination_metrics.items(), 
                    key=lambda x: x[1]['cohens_d'], reverse=True)[:3]
    for i, (pwm, metrics) in enumerate(top_pwm):
        print(f"   {i+1}. {pwm} (Cohen's d = {metrics['cohens_d']:.3f})")
    
    print("\n2. RECOMMENDED FEATURE COMBINATIONS:")
    print("   - Use resistance ratios between high-discriminating PWM settings")
    print("   - Include normalized response patterns (fingerprints)")
    print("   - Add statistical moments and variability metrics")
    print("   - Consider PWM-response correlation features")
    
    print("\n3. SIGNAL PROCESSING RECOMMENDATIONS:")
    print("   - Apply logarithmic transformation for resistance values")
    print("   - Use standardization for machine learning features")
    print("   - Consider temporal filtering for noise reduction")
    
    return feature_df, discrimination_metrics

if __name__ == "__main__":
    features, discrimination = main() 