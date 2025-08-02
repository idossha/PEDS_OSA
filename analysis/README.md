# PEDS OSA Sleep Spindle Analysis Pipeline

Comprehensive analysis pipeline for the PEDS OSA study examining the relationship between sleep spindle characteristics and cognitive performance in children with and without Obstructive Sleep Apnea (OSA).

## Overview

This repository contains multiple statistical analysis pipelines:

1. **Regional Spindle Analysis** (RECOMMENDED) - Complete anterior vs posterior spindle density comparison with TOVA scores
2. **Regional AHI Analysis** (RECOMMENDED) - Complete anterior vs posterior spindle density comparison with sleep apnea severity

## Quick Start

### 1. Setup

```bash
# Navigate to analysis directory
cd analysis

# Install dependencies
pip install -r analysis_requirements.txt

# Configure project path
# Edit config.py and set PROJECT_DIR to your data location
```

### 2. Run Analysis

```bash
# Run complete regional spindle analysis (recommended for TOVA)
python run_regional_analysis.py --project-dir /path/to/your/002_PEDS_OSA

# Run complete regional AHI analysis (recommended for AHI)  
python run_regional_ahi_analysis.py --project-dir /path/to/your/002_PEDS_OSA

# Alternative: Run legacy individual analyses
python run_ahi_analysis.py --project-dir /path/to/your/002_PEDS_OSA
python run_tova_analysis.py --project-dir /path/to/your/002_PEDS_OSA
```

## Data Structure

Your project directory should be organized as follows:

```
PROJECT_DIR/
├── derivatives/
│   └── sub-XXX/
│       └── spindles/
│           └── results/
│               └── all_spindles_detailed.csv
├── demographics.csv
└── TOVA.csv
```

### Data Files Description

- **Spindle Data**: Individual subject spindle detection results in `derivatives/sub-XXX/spindles/results/all_spindles_detailed.csv`
- **Demographics**: Subject information including age, sex, group (Control/OSA), and sleep parameters in `demographics.csv`
- **TOVA**: Cognitive performance scores including Cohen's d values in `TOVA.csv`

## Analysis Pipelines

### 1. Regional Spindle Analysis (Recommended)

**Streamlined Complete Pipeline**

This is the **recommended approach** that performs a comprehensive regional comparison in a single script.

**Research Questions:**
1. Does anterior spindle density predict cognitive performance improvement?
2. Does posterior spindle density predict cognitive performance improvement?  
3. Which brain region shows stronger associations with cognitive performance?

**Key Features:**
- Single script with all analysis logic
- Analyzes both anterior and posterior regions simultaneously
- Direct statistical comparison between regions (Steiger's Z-test)
- Generates only the essential publication figures
- Controls for age and sex in all models

**Usage:**
```bash
python run_regional_analysis.py --project-dir /path/to/your/002_PEDS_OSA
```

**Output:** `PROJECT_DIR/regional_analysis_results/`
- `main_results_figure.png` - 3-panel results figure
- `demographics_dashboard.png` - 6-panel demographics overview
- `analysis_summary.txt` - Comprehensive statistical summary

### 2. Regional AHI Analysis (Recommended)

**Streamlined Complete Pipeline**

This is the **recommended approach** for AHI analysis that performs a comprehensive regional comparison in a single script.

**Research Questions:**
1. Does log(AHI) predict anterior spindle density, controlling for age and sex?
2. Does log(AHI) predict posterior spindle density, controlling for age and sex?
3. Which brain region shows stronger associations with sleep apnea severity?

**Key Features:**
- Single script with all analysis logic
- Analyzes both anterior and posterior regions simultaneously
- Direct statistical comparison between regions (Steiger's Z-test)
- Uses log-transformed AHI for better model fit
- Generates only the essential publication figures
- Controls for age and sex in all models

**Usage:**
```bash
python run_regional_ahi_analysis.py --project-dir /path/to/your/002_PEDS_OSA
```

**Output:** `PROJECT_DIR/regional_ahi_results/`
- `main_results_figure.png` - 3-panel results figure
- `demographics_dashboard.png` - 6-panel demographics overview
- `analysis_summary.txt` - Comprehensive statistical summary

### 3. Individual AHI Analysis (Legacy)

**Research Questions:**
1. Does AHI predict spindle density, controlling for age and sex?
2. Does AHI disrupt normal spindle development (age × AHI interaction)?

**Key Features:**
- Uses log-transformed AHI to address non-linearity
- Examines both main effects and interaction models  
- Regional analysis comparing frontal vs parietal spindles
- Group comparisons between Control and OSA groups

**Statistical Models:**
```python
# Main effects model
spindle_density ~ age + AHI + sex

# Interaction model  
spindle_density ~ age * AHI + sex

# Log-transformed model
spindle_density ~ age + log1p(AHI) + sex
```

**Usage:**
```bash
python run_ahi_analysis.py [OPTIONS]
```

**Output Location:** `PROJECT_DIR/ahi_analysis_results/`

**Key Output Files:**
- `ahi_analysis_results.txt` - Statistical results summary
- `ahi_analysis_plots.png` - Visualization of relationships
- `ahi_analysis_data.csv` - Processed dataset
- `ahi_regional_comparison.png` - Frontal vs parietal analysis

### 4. Individual TOVA Analysis (Legacy)

**Research Questions:**
1. Does anterior spindle density predict cognitive performance improvement as measured by TOVA scores?
2. Does posterior spindle density predict cognitive performance improvement?
3. Which brain region shows stronger associations with cognitive performance?

**Key Features:**
- Analyzes both anterior and posterior brain regions separately
- Analyzes Cohen's d values as outcome measure (improvement = negative values)
- Direct statistical comparison between regional effects
- Controls for age and sex in all statistical models

**Statistical Models:**
```python
# Regional models
cohens_d ~ anterior_spindle_density + age + sex
cohens_d ~ posterior_spindle_density + age + sex

# Combined regional model
cohens_d ~ anterior_spindle_density + posterior_spindle_density + age + sex

# Regional comparison
Steiger's Z-test for correlation differences
```

**Usage:**
```bash
python run_tova_analysis.py [OPTIONS]
```

**Output Location:** `PROJECT_DIR/tova_analysis_results/`

**Key Output Files:**
- `analysis_dataset_anterior.csv` - Anterior region analysis dataset
- `analysis_dataset_posterior.csv` - Posterior region analysis dataset
- `analysis_dataset_regional_comparison.csv` - Combined regional dataset
- `regional_comparison_results.txt` - Statistical comparison between regions
- `regional_comparison_plot.png` - Comprehensive regional visualization
- `anterior_analysis_summary.txt` - Anterior region results
- `posterior_analysis_summary.txt` - Posterior region results

## Configuration

### Basic Configuration

Edit `config.py` to set your project directory:

```python
# Path to your project directory (MODIFY THIS)
PROJECT_DIR = "/path/to/your/002_PEDS_OSA"
```

### Advanced Configuration

The `config.py` file contains numerous parameters that can be customized:

**Spindle Detection Parameters:**
- `SPINDLE_FREQUENCY_THRESHOLD` - Threshold for slow vs fast spindles (default: 12 Hz)
- `MIN_SPINDLE_DURATION` - Minimum duration (default: 0.5 seconds)
- `MAX_SPINDLE_DURATION` - Maximum duration (default: 3.0 seconds)

**Analysis Parameters:**
- `AHI_THRESHOLD` - OSA diagnosis threshold (default: 1.0)
- `USE_ANTERIOR_CHANNELS_ONLY` - For TOVA analysis (default: True)
- `ALPHA_LEVEL` - Significance level (default: 0.05)

**Output Settings:**
- `AHI_OUTPUT_DIR` - AHI analysis results directory
- `TOVA_OUTPUT_DIR` - TOVA analysis results directory
- `DPI` - Plot resolution (default: 300)

## Command Line Options

Both analysis scripts support the following options:

```bash
# Basic usage
python run_ahi_analysis.py
python run_tova_analysis.py

# Specify project directory
python run_ahi_analysis.py --project-dir /path/to/data

# Use custom configuration
python run_ahi_analysis.py --config my_config.py

# Validate setup without running analysis
python run_ahi_analysis.py --check-only
```

## Data Processing Details

### Spindle Density Calculation

**AHI Analysis:**
- Uses all available channels for spindle density calculation
- Calculates spindles per minute of recording time
- Applies duration filters (0.5-3.0 seconds)

**TOVA Analysis:**
- **Anterior region**: Uses anterior/frontal channels (defined in `reference/segment_net.txt`)
- **Posterior region**: Uses posterior channels (defined in `reference/segment_net.txt`)
- Direct comparison between regions for comprehensive analysis
- Same duration and frequency criteria as AHI analysis

### Quality Control

- Minimum recording duration: 100 minutes
- Maximum missing data allowed: 20%
- Automatic exclusion of subjects not meeting criteria
- Validation of data file structure and contents

### Statistical Approach

**Model Selection:**
- Linear regression for continuous predictors
- ANCOVA for group comparisons with covariates
- Log transformation for skewed variables (AHI)
- Robust standard errors when appropriate

**Multiple Comparisons:**
- Bonferroni correction for multiple tests
- False Discovery Rate (FDR) control when applicable
- Clear reporting of corrected vs uncorrected p-values

## Output Interpretation

### AHI Analysis Results

**Key Metrics:**
- Main effect of AHI on spindle density
- Age × AHI interaction effects
- Group differences (Control vs OSA)
- Regional differences (frontal vs parietal)

**Clinical Significance:**
- Positive AHI effects indicate decreased spindle density with higher AHI
- Significant interactions suggest AHI disrupts normal development
- Effect sizes (Cohen's d) provided for clinical interpretation

## Summary of Recommended Pipelines

### Current Best Practices

**For TOVA Analysis (Cognitive Performance):**
- Use `run_regional_analysis.py` for complete anterior vs posterior analysis
- Generates publication-ready figures and comprehensive statistical summary
- Controls for age and sex in all models
- Direct regional comparison using Steiger's Z-test

**For AHI Analysis (Sleep Apnea Severity):**
- Use `run_regional_ahi_analysis.py` for complete anterior vs posterior analysis  
- Uses log-transformed AHI for better model fit
- Generates publication-ready figures and comprehensive statistical summary
- Controls for age and sex in all models
- Direct regional comparison using Steiger's Z-test

**Legacy Scripts:**
- `run_ahi_analysis.py` and `run_tova_analysis.py` are maintained for compatibility
- New analyses should use the regional versions for better insights

### TOVA Regional Analysis Results

**Key Metrics:**
- Association between anterior spindle density and cognitive improvement
- Association between posterior spindle density and cognitive improvement
- Regional comparison of effect sizes (Steiger's Z-test)
- Group differences in improvement rates by region
- Age and sex effects on regional relationships

**Clinical Significance:**
- Negative Cohen's d values indicate improvement (PM to AM)
- Larger magnitude indicates greater improvement
- Regional comparison identifies brain areas most relevant to cognitive resilience
- Combined model reveals independent contributions of each region
- Effect size differences inform targeted intervention strategies

## Dependencies

Required Python packages (see `analysis_requirements.txt`):

```
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
statsmodels>=0.13.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
reportlab>=3.6.0  # For PDF report generation
```

Install with:
```bash
pip install -r analysis_requirements.txt
```

## Troubleshooting

### Common Issues

**Missing Data Files:**
- Verify project directory structure matches expected format
- Check that spindle detection has been run for all subjects
- Ensure demographics.csv and TOVA.csv are in correct locations

**Import Errors:**
- Install missing packages: `pip install -r analysis_requirements.txt`
- Verify Python version compatibility (3.8+)
- Check virtual environment activation

**Analysis Failures:**
- Run with `--check-only` flag to validate setup
- Check log output for specific error messages
- Verify data file formats and column names

### Getting Help

**Validation Commands:**
```bash
# Check dependencies and data structure
python run_ahi_analysis.py --check-only
python run_tova_analysis.py --check-only
```

**Debug Mode:**
Add detailed logging by modifying config.py:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If using this analysis pipeline, please cite:

1. The PEDS OSA study and relevant publications
2. Software dependencies (pandas, scipy, statsmodels, etc.)
3. Spindle detection methodology references

## Development Notes

### File Organization

**Core Analysis Files:**
- `run_ahi_analysis.py` - AHI analysis entry point
- `run_tova_analysis.py` - TOVA analysis entry point
- `config.py` - Unified configuration system
- `analyze_spindle_density_vs_ahi.py` - AHI analysis implementation
- `analyze_spindle_density_vs_tova.py` - TOVA analysis implementation

**Supporting Files:**
- `reference/` - Reference data files for development/testing
- `analysis_requirements.txt` - Python package dependencies


### Version History

**Current Version:**
- Streamlined pipeline with unified configuration
- Clean entry points for both analyses
- Comprehensive validation and error handling
- Improved documentation and usability

**Previous Versions:**
- Multiple configuration files and runners
- Separate documentation for each analysis
- Limited validation and error handling