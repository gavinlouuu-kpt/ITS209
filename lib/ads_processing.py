import pandas as pd

# Constants for resistance calculations
LOAD_RESISTANCE = 10000  # Load resistance in ohms
INPUT_VOLTAGE = 3.3  # Input voltage in volts

def ads_to_voltage(ads_value):
    """Convert ADS value to voltage in volts."""
    return ads_value * 0.000125

def ads_to_resistance(ads_value):
    """Convert ADS value to resistance in ohms."""
    voltage = ads_to_voltage(ads_value)
    return (LOAD_RESISTANCE / voltage) * ((INPUT_VOLTAGE - voltage) / voltage)

def process_ads_columns(df, columns):
    """
    Apply ads_to_resistance to specified columns in a DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process
        
    Returns:
        Modified DataFrame
    """
    for col in columns:
        df[col] = df[col].apply(ads_to_resistance)
    return df

def normalize_columns(df, columns):
    """
    Normalize specified columns by dividing by their minimum value.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize
        
    Returns:
        Modified DataFrame
    """
    for col in columns:
        df[col] = df[col] / df[col].min()
    return df

def normalize_timestamp(df, timestamp_col='timestamp(ms)'):
    """
    Normalize timestamp column by subtracting minimum value.
    
    Args:
        df: pandas DataFrame
        timestamp_col: name of timestamp column
        
    Returns:
        Modified DataFrame
    """
    df[timestamp_col] = df[timestamp_col] - df[timestamp_col].min()
    return df

