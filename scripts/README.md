# Scripts Directory

This directory contains all Python scripts and analysis tools for the async_ads project, organized into logical subdirectories.

## Directory Structure

### `analysis/`
Contains Python scripts for data analysis and processing:

- **`comprehensive_analysis.py`** - Comprehensive analysis of gas sensor data
- **`comprehensive_gas_analysis.py`** - Advanced gas classification analysis  
- **`analyze_static_features.py`** - Static feature analysis and extraction
- **`static_features_analysis.py`** - Additional static analysis tools
- **`static_analysis.py`** - Core static data analysis
- **`mux_explore.py`** - Multiplexer exploration and analysis
- **`gas_classifier_realtime.py`** - Real-time gas classification system
- **`realtime_receiver.py`** - Real-time data receiver and processor

### `examples/`
Contains example code and integration demonstrations:

- **`python_integration_example.py`** - Python integration example
- **`arduino_integration_example.cpp`** - Arduino integration example

### `validation/`
Contains system validation and testing scripts:

- **`validate_system.py`** - System validation and testing suite
- **`validation_results.png`** - Validation results visualization

## Usage

All analysis scripts can be run from their respective directories. The import paths have been configured to work correctly with the new structure.

Example:
```bash
cd scripts/analysis
python comprehensive_analysis.py

cd ../validation  
python validate_system.py
```

## Dependencies

Scripts may require various Python packages including:
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn
- scipy

Install required packages with:
```bash
pip install pandas numpy matplotlib scikit-learn seaborn scipy
```

