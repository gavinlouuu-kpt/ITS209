import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

etoh_df = pd.read_csv(r'/home/gavin/Documents/PlatformIO/Projects/async_ads/results/etoh.csv')
prop_df = pd.read_csv(r'/home/gavin/Documents/PlatformIO/Projects/async_ads/results/2pol.csv')

def bin_align(df):
    '''
    normalise timestamp to start at 0
    the settings go in a loop where at one timestamp there is only one setting with reading
    we will bin the timestamp into a range where all settings have one reading to get rid off NaN
    i.e., t1:setting 1, t2:setting 2, t3:setting 3, t4:setting 4
    the binning will be t1-t4: setting [1,2,3,4]   
    '''
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Convert timestamp column to numeric if it's not already
    df['timestamp(ms)'] = pd.to_numeric(df['timestamp(ms)'])
    
    # Normalize timestamp to start at 0
    df['timestamp(ms)'] = df['timestamp(ms)'] - df['timestamp(ms)'].min()
    
    # Get all setting columns (excluding timestamp)
    setting_cols = [col for col in df.columns if col.startswith('setting_')]
    
    # Convert setting columns to numeric, replacing empty strings with NaN
    for col in setting_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Find the number of unique settings
    num_settings = len(setting_cols)
    
    # Group rows into bins based on the cycling pattern
    # Each bin should contain num_settings consecutive rows
    df['bin_id'] = df.index // num_settings
    
    # Create the aligned dataframe
    aligned_data = []
    
    for bin_id in df['bin_id'].unique():
        bin_data = df[df['bin_id'] == bin_id].copy()
        
        # Skip incomplete bins (less than num_settings rows)
        if len(bin_data) < num_settings:
            continue
            
        # Take the first timestamp of the bin as the bin timestamp
        bin_timestamp = bin_data['timestamp(ms)'].iloc[0]
        
        # Create a row for this bin
        bin_row = {'timestamp(ms)': bin_timestamp}
        
        # For each setting, find the non-NaN value in this bin
        for col in setting_cols:
            non_nan_values = bin_data[col].dropna()
            if len(non_nan_values) > 0:
                bin_row[col] = non_nan_values.iloc[0]
            else:
                bin_row[col] = np.nan
        
        aligned_data.append(bin_row)
    
    # Create the final aligned dataframe
    aligned_df = pd.DataFrame(aligned_data)
    # Drop rows with NaN
    aligned_df = aligned_df.dropna()
    
    return aligned_df

def extract_features(df, sample_name):
    """
    Extract distinguishing features from sensor data
    """
    # Get setting columns
    setting_cols = [col for col in df.columns if col.startswith('setting_')]
    
    # Convert to numpy array for easier manipulation
    sensor_data = df[setting_cols].values
    
    features = {}
    
    # 1. Statistical features for each sensor
    for i, col in enumerate(setting_cols):
        resistance = col.split('_')[1]  # Extract resistance value
        sensor_values = sensor_data[:, i]
        
        features[f'mean_{resistance}'] = np.mean(sensor_values)
        features[f'std_{resistance}'] = np.std(sensor_values)
        features[f'max_{resistance}'] = np.max(sensor_values)
        features[f'min_{resistance}'] = np.min(sensor_values)
        features[f'median_{resistance}'] = np.median(sensor_values)
        features[f'range_{resistance}'] = np.max(sensor_values) - np.min(sensor_values)
    
    # 2. Cross-sensor ratios (important for gas sensor discrimination)
    for i in range(len(setting_cols)):
        for j in range(i+1, len(setting_cols)):
            res1 = setting_cols[i].split('_')[1]
            res2 = setting_cols[j].split('_')[1]
            
            # Calculate mean ratio
            mean_ratio = np.mean(sensor_data[:, i]) / np.mean(sensor_data[:, j])
            features[f'ratio_{res1}_{res2}'] = mean_ratio
    
    # 3. Response patterns over time
    features['total_mean'] = np.mean(sensor_data)
    features['total_std'] = np.std(sensor_data)
    features['cv'] = features['total_std'] / features['total_mean']  # Coefficient of variation
    
    # 4. Principal response (highest responding sensor)
    max_responses = np.max(sensor_data, axis=0)
    features['max_response_sensor'] = np.argmax(max_responses)
    features['max_response_value'] = np.max(max_responses)
    
    # 5. Response stability (variance across time for each sensor)
    for i, col in enumerate(setting_cols):
        resistance = col.split('_')[1]
        features[f'stability_{resistance}'] = np.var(sensor_data[:, i])
    
    # 6. Response fingerprint (normalized pattern)
    mean_responses = np.mean(sensor_data, axis=0)
    normalized_pattern = mean_responses / np.sum(mean_responses)
    for i, col in enumerate(setting_cols):
        resistance = col.split('_')[1]
        features[f'fingerprint_{resistance}'] = normalized_pattern[i]
    
    # Add sample identifier
    features['sample'] = sample_name
    
    return features

def plot_sensor_patterns(etoh_df, prop_df):
    """
    Visualize the sensor response patterns for both samples
    """
    setting_cols = [col for col in etoh_df.columns if col.startswith('setting_')]
    
    # Calculate mean responses
    etoh_means = etoh_df[setting_cols].mean()
    prop_means = prop_df[setting_cols].mean()
    
    # Extract resistance values for x-axis
    resistances = [int(col.split('_')[1]) for col in setting_cols]
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Mean response comparison
    plt.subplot(2, 2, 1)
    plt.plot(resistances, etoh_means.values, 'o-', label='Ethanol', linewidth=2, markersize=6)
    plt.plot(resistances, prop_means.values, 's-', label='2-Propanol', linewidth=2, markersize=6)
    plt.xlabel('Resistance Setting (Ohms)')
    plt.ylabel('Mean Response')
    plt.title('Mean Sensor Response Patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Normalized fingerprint
    plt.subplot(2, 2, 2)
    etoh_norm = etoh_means.values / np.sum(etoh_means.values)
    prop_norm = prop_means.values / np.sum(prop_means.values)
    plt.plot(resistances, etoh_norm, 'o-', label='Ethanol', linewidth=2, markersize=6)
    plt.plot(resistances, prop_norm, 's-', label='2-Propanol', linewidth=2, markersize=6)
    plt.xlabel('Resistance Setting (Ohms)')
    plt.ylabel('Normalized Response')
    plt.title('Normalized Response Fingerprint')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Response variability
    plt.subplot(2, 2, 3)
    etoh_std = etoh_df[setting_cols].std()
    prop_std = prop_df[setting_cols].std()
    plt.plot(resistances, etoh_std.values, 'o-', label='Ethanol', linewidth=2, markersize=6)
    plt.plot(resistances, prop_std.values, 's-', label='2-Propanol', linewidth=2, markersize=6)
    plt.xlabel('Resistance Setting (Ohms)')
    plt.ylabel('Standard Deviation')
    plt.title('Response Variability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Response ratios (example with first few sensors)
    plt.subplot(2, 2, 4)
    ratios_etoh = []
    ratios_prop = []
    ratio_labels = []
    
    for i in range(min(3, len(resistances)-1)):
        ratio_etoh = etoh_means.iloc[i] / etoh_means.iloc[i+1]
        ratio_prop = prop_means.iloc[i] / prop_means.iloc[i+1]
        ratios_etoh.append(ratio_etoh)
        ratios_prop.append(ratio_prop)
        ratio_labels.append(f'{resistances[i]}/{resistances[i+1]}')
    
    x_pos = np.arange(len(ratio_labels))
    width = 0.35
    plt.bar(x_pos - width/2, ratios_etoh, width, label='Ethanol', alpha=0.8)
    plt.bar(x_pos + width/2, ratios_prop, width, label='2-Propanol', alpha=0.8)
    plt.xlabel('Sensor Ratios')
    plt.ylabel('Response Ratio')
    plt.title('Sensor Response Ratios')
    plt.xticks(x_pos, ratio_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

etoh_df = bin_align(etoh_df)
prop_df = bin_align(prop_df)

print(etoh_df.head())
print(prop_df.head())

# Extract features for both samples
etoh_features = extract_features(etoh_df, 'ethanol')
prop_features = extract_features(prop_df, '2-propanol')

# Create feature comparison dataframe
feature_df = pd.DataFrame([etoh_features, prop_features])
print("\nFeature Comparison:")
print(feature_df.T)  # Transpose for better readability

# Visualize the patterns
plot_sensor_patterns(etoh_df, prop_df)

def create_ml_features(etoh_df, prop_df):
    """
    Create a combined feature matrix suitable for machine learning
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    # Extract features for both samples
    etoh_features = extract_features(etoh_df, 'ethanol')
    prop_features = extract_features(prop_df, '2-propanol')
    
    # Create feature dataframe
    features_df = pd.DataFrame([etoh_features, prop_features])
    
    # Separate features and labels
    feature_cols = [col for col in features_df.columns if col != 'sample']
    X = features_df[feature_cols].values
    y = features_df['sample'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for dimensionality reduction and visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Print PCA results
    print("\nPCA Analysis:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Plot PCA results
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue']
    for i, sample in enumerate(['ethanol', '2-propanol']):
        mask = y == sample
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i], label=sample, s=100, alpha=0.7)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA - Sample Separation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature importance from PCA
    plt.subplot(1, 2, 2)
    feature_importance = np.abs(pca.components_[0])  # First PC
    top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
    
    plt.barh(range(len(top_features_idx)), 
             feature_importance[top_features_idx])
    plt.yticks(range(len(top_features_idx)), 
               [feature_cols[i] for i in top_features_idx])
    plt.xlabel('Feature Importance (PC1)')
    plt.title('Most Discriminative Features')
    plt.tight_layout()
    plt.show()
    
    return X_scaled, y, feature_cols, scaler, pca

def calculate_discrimination_metrics(etoh_df, prop_df):
    """
    Calculate metrics to quantify how well samples can be distinguished
    """
    setting_cols = [col for col in etoh_df.columns if col.startswith('setting_')]
    
    discrimination_metrics = {}
    
    for col in setting_cols:
        resistance = col.split('_')[1]
        
        etoh_values = etoh_df[col].values
        prop_values = prop_df[col].values
        
        # Statistical separation metrics
        etoh_mean = np.mean(etoh_values)
        prop_mean = np.mean(prop_values)
        etoh_std = np.std(etoh_values)
        prop_std = np.std(etoh_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(etoh_values)-1)*etoh_std**2 + 
                             (len(prop_values)-1)*prop_std**2) / 
                            (len(etoh_values) + len(prop_values) - 2))
        cohens_d = abs(etoh_mean - prop_mean) / pooled_std
        
        # Signal-to-noise ratio
        signal = abs(etoh_mean - prop_mean)
        noise = (etoh_std + prop_std) / 2
        snr = signal / noise if noise > 0 else 0
        
        # Separation index
        separation = abs(etoh_mean - prop_mean) / (etoh_std + prop_std)
        
        discrimination_metrics[f'{resistance}_cohens_d'] = cohens_d
        discrimination_metrics[f'{resistance}_snr'] = snr
        discrimination_metrics[f'{resistance}_separation'] = separation
    
    return discrimination_metrics

def recommend_best_features(etoh_df, prop_df):
    """
    Recommend the best features for sample discrimination
    """
    # Calculate discrimination metrics
    metrics = calculate_discrimination_metrics(etoh_df, prop_df)
    
    # Extract Cohen's d values
    cohens_d_metrics = {k: v for k, v in metrics.items() if 'cohens_d' in k}
    snr_metrics = {k: v for k, v in metrics.items() if 'snr' in k}
    sep_metrics = {k: v for k, v in metrics.items() if 'separation' in k}
    
    print("\nDiscrimination Analysis:")
    print("=" * 50)
    
    # Best sensors by different metrics
    best_cohens = max(cohens_d_metrics, key=cohens_d_metrics.get)
    best_snr = max(snr_metrics, key=snr_metrics.get)
    best_sep = max(sep_metrics, key=sep_metrics.get)
    
    print(f"Best sensor by Cohen's d: {best_cohens} (d = {cohens_d_metrics[best_cohens]:.3f})")
    print(f"Best sensor by SNR: {best_snr} (SNR = {snr_metrics[best_snr]:.3f})")
    print(f"Best sensor by Separation: {best_sep} (Sep = {sep_metrics[best_sep]:.3f})")
    
    # Recommendations
    print("\nRecommendations:")
    print("=" * 50)
    
    # High Cohen's d (> 0.8 is large effect)
    high_d = [k for k, v in cohens_d_metrics.items() if v > 0.8]
    if high_d:
        print(f"Sensors with large effect size (d > 0.8): {[k.split('_')[0] for k in high_d]}")
    
    # High SNR
    high_snr = [k for k, v in snr_metrics.items() if v > 2.0]
    if high_snr:
        print(f"Sensors with high SNR (> 2.0): {[k.split('_')[0] for k in high_snr]}")
    
    # Feature combination recommendations
    print("\nFeature Combination Recommendations:")
    print("1. Use sensor response ratios between high-performing sensors")
    print("2. Combine normalized response patterns (fingerprints)")
    print("3. Include stability/variability metrics for robust classification")
    print("4. Consider PCA features for dimensionality reduction")
    
    return metrics

# Run the analysis
print("\n" + "="*60)
print("COMPREHENSIVE FEATURE ANALYSIS")
print("="*60)

# Machine learning features
X_scaled, y, feature_cols, scaler, pca = create_ml_features(etoh_df, prop_df)

# Discrimination analysis
discrimination_metrics = recommend_best_features(etoh_df, prop_df)

def apply_temporal_pca(df, window_size=50, overlap=0.5):
    """
    Apply PCA to time series data using different temporal approaches
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    setting_cols = [col for col in df.columns if col.startswith('setting_')]
    
    print("\n" + "="*60)
    print("TEMPORAL PCA ANALYSIS")
    print("="*60)
    
    # Approach 1: Direct PCA on sensor array at each time point
    print("\n1. SENSOR ARRAY PCA (across sensors at each time)")
    print("-" * 50)
    
    # Each row is a time point, each column is a sensor
    sensor_data = df[setting_cols].values
    
    # Standardize across sensors (columns)
    scaler_sensors = StandardScaler()
    sensor_data_scaled = scaler_sensors.fit_transform(sensor_data)
    
    # PCA across sensors
    pca_sensors = PCA(n_components=min(len(setting_cols), 3))
    sensor_pca = pca_sensors.fit_transform(sensor_data_scaled)
    
    print(f"Explained variance ratio: {pca_sensors.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca_sensors.explained_variance_ratio_):.3f}")
    
    # Plot sensor PCA over time
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(df['timestamp(ms)'], sensor_pca[:, 0], 'b-', linewidth=2, label='PC1')
    plt.plot(df['timestamp(ms)'], sensor_pca[:, 1], 'r-', linewidth=2, label='PC2')
    if sensor_pca.shape[1] > 2:
        plt.plot(df['timestamp(ms)'], sensor_pca[:, 2], 'g-', linewidth=2, label='PC3')
    plt.xlabel('Time (ms)')
    plt.ylabel('PC Score')
    plt.title('Sensor PCA Components Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Approach 2: Sliding Window PCA
    print("\n2. SLIDING WINDOW PCA")
    print("-" * 50)
    
    step_size = int(window_size * (1 - overlap))
    n_windows = (len(df) - window_size) // step_size + 1
    
    window_features = []
    window_times = []
    
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        if end_idx > len(df):
            break
            
        window_data = sensor_data_scaled[start_idx:end_idx, :]
        
        # Calculate features for this window
        window_mean = np.mean(window_data, axis=0)
        window_std = np.std(window_data, axis=0)
        window_max = np.max(window_data, axis=0)
        window_min = np.min(window_data, axis=0)
        
        # Combine features
        window_feature = np.concatenate([window_mean, window_std, window_max, window_min])
        window_features.append(window_feature)
        window_times.append(df['timestamp(ms)'].iloc[start_idx + window_size//2])
    
    window_features = np.array(window_features)
    
    # Apply PCA to window features
    pca_windows = PCA(n_components=3)
    window_pca = pca_windows.fit_transform(window_features)
    
    print(f"Window PCA explained variance: {pca_windows.explained_variance_ratio_}")
    
    plt.subplot(2, 3, 2)
    plt.plot(window_times, window_pca[:, 0], 'b-', linewidth=2, label='PC1')
    plt.plot(window_times, window_pca[:, 1], 'r-', linewidth=2, label='PC2')
    plt.plot(window_times, window_pca[:, 2], 'g-', linewidth=2, label='PC3')
    plt.xlabel('Time (ms)')
    plt.ylabel('PC Score')
    plt.title(f'Sliding Window PCA (window={window_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Approach 3: Trajectory PCA (each sensor as a trajectory)
    print("\n3. TRAJECTORY PCA (across time for each sensor)")
    print("-" * 50)
    
    # Transpose: each row is a sensor trajectory over time
    trajectory_data = sensor_data_scaled.T
    
    # Standardize across time (columns)
    scaler_time = StandardScaler()
    trajectory_data_scaled = scaler_time.fit_transform(trajectory_data)
    
    # PCA across time
    pca_trajectory = PCA(n_components=min(len(df), 5))
    trajectory_pca = pca_trajectory.fit_transform(trajectory_data_scaled)
    
    print(f"Trajectory PCA explained variance: {pca_trajectory.explained_variance_ratio_[:3]}")
    
    plt.subplot(2, 3, 3)
    resistances = [int(col.split('_')[1]) for col in setting_cols]
    plt.plot(resistances, trajectory_pca[:, 0], 'bo-', linewidth=2, markersize=6, label='PC1')
    plt.plot(resistances, trajectory_pca[:, 1], 'ro-', linewidth=2, markersize=6, label='PC2')
    plt.plot(resistances, trajectory_pca[:, 2], 'go-', linewidth=2, markersize=6, label='PC3')
    plt.xlabel('Resistance Setting (Ohms)')
    plt.ylabel('PC Score')
    plt.title('Trajectory PCA (Sensor Loadings)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Approach 4: Phase Space PCA (embedding)
    print("\n4. PHASE SPACE EMBEDDING PCA")
    print("-" * 50)
    
    def create_embedding(signal, embedding_dim=3, delay=1):
        """Create phase space embedding"""
        n_points = len(signal) - (embedding_dim - 1) * delay
        embedded = np.zeros((n_points, embedding_dim))
        
        for i in range(embedding_dim):
            embedded[:, i] = signal[i * delay:i * delay + n_points]
        
        return embedded
    
    # Use the first principal component from sensor PCA
    main_signal = sensor_pca[:, 0]
    
    # Create phase space embedding
    embedding = create_embedding(main_signal, embedding_dim=3, delay=5)
    
    # Apply PCA to embedded data
    pca_embedding = PCA(n_components=3)
    embedding_pca = pca_embedding.fit_transform(embedding)
    
    print(f"Phase space PCA explained variance: {pca_embedding.explained_variance_ratio_}")
    
    plt.subplot(2, 3, 4)
    plt.plot(embedding_pca[:, 0], 'b-', linewidth=2, label='PC1')
    plt.plot(embedding_pca[:, 1], 'r-', linewidth=2, label='PC2')
    plt.plot(embedding_pca[:, 2], 'g-', linewidth=2, label='PC3')
    plt.xlabel('Time Points')
    plt.ylabel('PC Score')
    plt.title('Phase Space Embedding PCA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Approach 5: Frequency Domain PCA
    print("\n5. FREQUENCY DOMAIN PCA")
    print("-" * 50)
    
    # Apply FFT to each sensor
    freq_data = []
    for col in setting_cols:
        signal = df[col].values
        fft_signal = np.abs(np.fft.fft(signal))
        # Take first half (positive frequencies)
        freq_data.append(fft_signal[:len(fft_signal)//2])
    
    freq_data = np.array(freq_data)
    
    # Apply PCA to frequency domain
    pca_freq = PCA(n_components=3)
    freq_pca = pca_freq.fit_transform(freq_data)
    
    print(f"Frequency PCA explained variance: {pca_freq.explained_variance_ratio_}")
    
    plt.subplot(2, 3, 5)
    resistances = [int(col.split('_')[1]) for col in setting_cols]
    plt.plot(resistances, freq_pca[:, 0], 'bo-', linewidth=2, markersize=6, label='PC1')
    plt.plot(resistances, freq_pca[:, 1], 'ro-', linewidth=2, markersize=6, label='PC2')
    plt.plot(resistances, freq_pca[:, 2], 'go-', linewidth=2, markersize=6, label='PC3')
    plt.xlabel('Resistance Setting (Ohms)')
    plt.ylabel('PC Score')
    plt.title('Frequency Domain PCA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Summary plot: 3D trajectory of sensor PCA
    plt.subplot(2, 3, 6)
    # Create a 3D-like visualization using 2D projection
    if sensor_pca.shape[1] >= 3:
        scatter = plt.scatter(sensor_pca[:, 0], sensor_pca[:, 1], 
                            c=range(len(sensor_pca)), cmap='viridis', 
                            s=20, alpha=0.6)
        plt.colorbar(scatter, label='Time')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Sensor Response Trajectory\n(colored by time)')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'sensor_pca': sensor_pca,
        'window_pca': window_pca,
        'trajectory_pca': trajectory_pca,
        'embedding_pca': embedding_pca,
        'freq_pca': freq_pca,
        'pca_models': {
            'sensor': pca_sensors,
            'window': pca_windows,
            'trajectory': pca_trajectory,
            'embedding': pca_embedding,
            'frequency': pca_freq
        }
    }

def compare_samples_temporal_pca(etoh_df, prop_df):
    """
    Compare both samples using temporal PCA approaches
    """
    print("\n" + "="*60)
    print("TEMPORAL PCA COMPARISON BETWEEN SAMPLES")
    print("="*60)
    
    # Apply temporal PCA to both samples
    etoh_results = apply_temporal_pca(etoh_df, window_size=30)
    prop_results = apply_temporal_pca(prop_df, window_size=30)
    
    # Compare the PCA spaces
    plt.figure(figsize=(15, 8))
    
    # Compare sensor PCA trajectories
    plt.subplot(2, 3, 1)
    plt.plot(etoh_results['sensor_pca'][:, 0], etoh_results['sensor_pca'][:, 1], 
             'b-', alpha=0.7, linewidth=2, label='Ethanol')
    plt.plot(prop_results['sensor_pca'][:, 0], prop_results['sensor_pca'][:, 1], 
             'r-', alpha=0.7, linewidth=2, label='2-Propanol')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Sensor PCA Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare window PCA
    plt.subplot(2, 3, 2)
    if len(etoh_results['window_pca']) > 0 and len(prop_results['window_pca']) > 0:
        plt.plot(etoh_results['window_pca'][:, 0], etoh_results['window_pca'][:, 1], 
                 'bo', alpha=0.7, markersize=4, label='Ethanol')
        plt.plot(prop_results['window_pca'][:, 0], prop_results['window_pca'][:, 1], 
                 'ro', alpha=0.7, markersize=4, label='2-Propanol')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Window PCA Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Compare trajectory PCA
    plt.subplot(2, 3, 3)
    setting_cols = [col for col in etoh_df.columns if col.startswith('setting_')]
    resistances = [int(col.split('_')[1]) for col in setting_cols]
    
    plt.plot(resistances, etoh_results['trajectory_pca'][:, 0], 
             'bo-', linewidth=2, markersize=6, label='Ethanol PC1')
    plt.plot(resistances, prop_results['trajectory_pca'][:, 0], 
             'ro-', linewidth=2, markersize=6, label='2-Propanol PC1')
    plt.xlabel('Resistance Setting (Ohms)')
    plt.ylabel('PC1 Score')
    plt.title('Trajectory PCA Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare frequency PCA
    plt.subplot(2, 3, 4)
    plt.plot(resistances, etoh_results['freq_pca'][:, 0], 
             'bo-', linewidth=2, markersize=6, label='Ethanol PC1')
    plt.plot(resistances, prop_results['freq_pca'][:, 0], 
             'ro-', linewidth=2, markersize=6, label='2-Propanol PC1')
    plt.xlabel('Resistance Setting (Ohms)')
    plt.ylabel('PC1 Score')
    plt.title('Frequency PCA Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistical comparison
    plt.subplot(2, 3, 5)
    etoh_pc1_stats = [np.mean(etoh_results['sensor_pca'][:, 0]), 
                      np.std(etoh_results['sensor_pca'][:, 0])]
    prop_pc1_stats = [np.mean(prop_results['sensor_pca'][:, 0]), 
                      np.std(prop_results['sensor_pca'][:, 0])]
    
    x_pos = [0, 1]
    plt.bar([x-0.2 for x in x_pos], etoh_pc1_stats, 0.4, 
            label='Ethanol', alpha=0.7)
    plt.bar([x+0.2 for x in x_pos], prop_pc1_stats, 0.4, 
            label='2-Propanol', alpha=0.7)
    plt.xticks(x_pos, ['Mean', 'Std'])
    plt.ylabel('PC1 Value')
    plt.title('PC1 Statistics Comparison')
    plt.legend()
    
    # Discrimination potential
    plt.subplot(2, 3, 6)
    # Calculate separation between samples in PC space
    etoh_centroid = np.mean(etoh_results['sensor_pca'][:, :2], axis=0)
    prop_centroid = np.mean(prop_results['sensor_pca'][:, :2], axis=0)
    separation = np.linalg.norm(etoh_centroid - prop_centroid)
    
    plt.scatter(*etoh_centroid, s=200, c='blue', marker='o', 
                label=f'Ethanol Centroid', alpha=0.8)
    plt.scatter(*prop_centroid, s=200, c='red', marker='s', 
                label=f'2-Propanol Centroid', alpha=0.8)
    plt.plot([etoh_centroid[0], prop_centroid[0]], 
             [etoh_centroid[1], prop_centroid[1]], 'k--', linewidth=2)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Sample Separation = {separation:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nSample separation in PC space: {separation:.3f}")
    print("Higher separation indicates better discriminability")
    
    return etoh_results, prop_results

# Run temporal PCA analysis
print("\n" + "="*60)
print("RUNNING TEMPORAL PCA ANALYSIS")
print("="*60)

# Analyze ethanol data
print("\nAnalyzing Ethanol Data:")
etoh_pca_results = apply_temporal_pca(etoh_df, window_size=30)

# Compare both samples
etoh_results, prop_results = compare_samples_temporal_pca(etoh_df, prop_df)