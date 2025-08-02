# PEDS OSA: Spindle Density vs AHI Analysis

This analysis script examines the relationship between sleep spindle density and AHI (Apnea-Hypopnea Index) in children with and without OSA, following the methodology outlined in the reference document.

## Overview

This analysis addresses two main research questions:

1. **Question 1**: Does AHI predict spindle density, and how does its impact compare to that of age, given age's known developmental effect on spindle density?

2. **Question 2**: Does AHI change the relationship between age and spindle density — i.e., does AHI disrupt normal spindle development?

## Key Features

- **Spindle Density Focus**: Uses spindle density (spindles per minute) instead of raw spindle count
- **AHI Analysis**: Examines the relationship between AHI and spindle density
- **Log-Transformed AHI**: Uses log1p(AHI) as recommended in the reference document
- **Interaction Analysis**: Tests whether AHI moderates the age-spindle density relationship
- **Regional Analysis**: Compares frontal vs parietal spindle density (if data available)
- **Group Comparisons**: Control vs OSA group differences

## Statistical Models

### 1. Main Effects Model
```
spindle_density ~ age + AHI + sex
```
Tests the independent effects of age and AHI on spindle density.

### 2. Interaction Model
```
spindle_density ~ age * AHI + sex
```
Tests whether AHI moderates the relationship between age and spindle density.

### 3. Log-Transformed AHI Model
```
spindle_density ~ age + log1p(AHI) + sex
```
Addresses non-linearity and heteroscedasticity in AHI effects.

### 4. Regional Analysis
```
spindle_density ~ Region * age + Region * AHI + sex
```
Compares frontal vs parietal spindle density effects.

## Data Requirements

### Input Files
- **Spindle Data**: `002_PEDS_OSA/derivatives/sub-XXX/spindles/results/all_spindles_detailed.csv`
- **Demographics**: `demographics.csv` (must include AHI column)
- **Group Information**: Control vs OSA group assignments

### Required Columns in Demographics
- `Subject`: Subject identifier
- `age`: Age in years
- `sex`: Sex (M/F)
- `AHI`: Apnea-Hypopnea Index
- `group`: Group assignment (Control/OSA)

## Output Files

All results are saved to `project_dir/analysis_2/`:

- `analysis_dataset.csv`: Complete analysis dataset
- `main_effects_analysis_results.txt`: Main effects model results
- `interaction_analysis_results.txt`: Interaction model results
- `log_ahi_analysis_results.txt`: Log-transformed AHI analysis
- `regional_analysis_results.txt`: Regional comparison results
- `group_analysis_results.txt`: Group comparison results
- `spindle_density_analysis_plots.png`: Visualization plots
- `analysis_summary.txt`: Comprehensive summary report

## Usage

1. **Configure the script**:
   ```bash
   # Copy and modify the configuration file
   cp config_ahi_analysis.py config.py
   # Edit config.py with your project directory path
   ```

2. **Run the analysis**:
   ```bash
   python run_ahi_analysis.py --project-dir /path/to/your/002_PEDS_OSA
   ```

## Key Differences from Original Analysis

| Aspect | Original Analysis | Density Analysis |
|--------|------------------|------------------|
| **Outcome Variable** | TOVA performance | Spindle density |
| **Main Predictor** | Spindle density | AHI |
| **Focus** | Cognitive performance | Sleep disorder severity |
| **Models** | ANCOVA, LMM | Linear models, interactions |
| **Transformation** | None | Log-transformed AHI |
| **Output Directory** | `analysis/` | `analysis_2/` |

## Interpretation Guidelines

### Main Effects
- **Age effect**: Expected positive relationship (spindle density increases with age)
- **AHI effect**: Negative relationship suggests OSA reduces spindle density
- **Standardized coefficients**: Compare relative importance of age vs AHI

### Interaction Effects
- **Significant interaction**: AHI moderates age-spindle relationship
- **Negative interaction**: Higher AHI reduces age-related spindle density increase
- **Positive interaction**: Higher AHI enhances age-related spindle density increase

### Log-Transformed AHI
- **Interpretation**: Change in spindle density per unit increase in log(AHI)
- **Practical meaning**: 1-unit increase in log(AHI) ≈ 2.7-fold increase in raw AHI

## Troubleshooting

### Common Issues
1. **Missing demographics file**: Script will create dummy data for testing
2. **No regional data**: Regional analysis will be skipped
3. **Missing AHI data**: Analysis will exclude subjects without AHI values

### Data Quality Checks
- Minimum recording duration: 100 minutes
- Maximum missing data: 20%
- AHI range validation
- Age range validation

## References

This analysis follows the methodology outlined in `reference/OSA_impact_on_spindlles.txt`, which recommends:
- Using log1p(AHI) transformation
- Testing age × AHI interactions
- Comparing relative importance of age vs AHI effects
- Regional analysis (frontal vs parietal)

## Contact

For questions or issues with this analysis, please refer to the main project documentation or contact the development team. 