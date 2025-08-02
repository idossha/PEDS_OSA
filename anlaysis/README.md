# PEDS OSA Spindle Analysis

Production analysis pipeline for the PEDS OSA study examining sleep spindle density and TOVA performance in children with and without Obstructive Sleep Apnea (OSA).

## Overview

This pipeline analyzes the relationship between sleep spindle characteristics and cognitive performance (TOVA test) in children with OSA compared to healthy controls.

## Data Requirements

### Input Data Structure
```
project_directory/
├── derivatives/
│   └── sub-XXX/
│       └── spindles/
│           └── results/
│               └── all_spindles_detailed.csv
├── demographics.csv
└── TOVA.csv
```

### Data Files
- **Spindle data**: Individual subject spindle detection results
- **Demographics**: Age, sex, group (Control/OSA), sleep parameters
- **TOVA**: Performance scores (PM/AM testing)

## Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r analysis_requirements.txt
```

## Usage

1. Update the project directory path in `peds_osa_spindle_analysis.py`:
```python
project_dir = "/path/to/your/002_PEDS_OSA"
```

2. Run the analysis:
```bash
python peds_osa_spindle_analysis.py
```

3. Generate PDF report (optional):
```bash
python report_create.py
```

## Output

All results are saved to `project_dir/analysis/`:

### Data Files
- `analysis_dataset_wide.csv` - Wide format dataset
- `analysis_dataset_long.csv` - Long format dataset for mixed models

### Statistical Results
- `ancova_results.txt` - ANCOVA analysis results
- `lmm_results.txt` - Linear Mixed Model results
- `group_analysis_results.txt` - Group comparison results
- `comprehensive_analysis_summary.txt` - Complete analysis summary

### Visualizations
- `publication_interaction_plot.png` - Publication-ready interaction plot (2 panels):
  - Panel A: TOVA Performance: Time × Spindle Density Interaction
  - Panel B: Spindle Density vs TOVA Performance Change (PM - AM)
- `descriptive_info.png` - Comprehensive descriptive statistics (7 panels):
  - Age distribution by group
  - Sex distribution
  - Group distribution
  - Spindle density by group
  - TOVA PM scores by group
  - TOVA AM scores by group
  - Overall sample characteristics summary

### Reports
- `PEDS_OSA_Spindle_Analysis_Report_YYYYMMDD_HHMMSS.pdf` - Comprehensive PDF report including:
  - Executive summary
  - Methods section
  - Descriptive statistics with figures
  - Interaction analysis with visualizations
  - Statistical results from all models
  - Conclusions and clinical implications
  - Appendix with file descriptions

## Statistical Models

### 1. ANCOVA Model
**Formula**: `TOVA_AM ~ TOVA_PM + spindle_density + age + sex`
**Purpose**: Test effects of spindle density on morning TOVA performance

### 2. Linear Mixed Model
**Formula**: `TOVA_score ~ Time * spindle_density + age + sex + (1|ID)`
**Purpose**: Test time × spindle density interaction

### 3. Group Comparisons
**Tests**: Independent t-tests between Control and OSA groups
**Variables**: Spindle density, TOVA improvement

## Key Findings

- **Significant interaction** between time and spindle density (p = 0.020)
- Spindle density moderates the effect of time on TOVA performance
- No significant group differences in spindle density or TOVA improvement
- Baseline TOVA performance significantly predicts morning performance

## Requirements

- Python 3.8+
- pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn, reportlab
- See `analysis_requirements.txt` for specific versions

## Citation

If using this analysis pipeline, please cite the PEDS OSA study and relevant methodological references. 