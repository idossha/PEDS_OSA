# PEDS OSA Analysis - File Organization Guide

This document explains the file naming convention and organization for the PEDS OSA analysis scripts.

## 📁 File Naming Convention

All files follow a clear, descriptive naming pattern that indicates their purpose:

### **Analysis Scripts**
- `analyze_spindle_density_vs_ahi.py` - Main analysis for spindle density vs AHI
- `analyze_spindle_density_vs_tova.py` - Main analysis for spindle density vs TOVA performance

### **Execution Scripts**
- `run_ahi_analysis.py` - Command-line interface for AHI analysis
- `run_tova_analysis.py` - Command-line interface for TOVA analysis

### **Configuration Files**
- `config_ahi_analysis.py` - Configuration for AHI analysis
- `config_tova_analysis.py` - Configuration for TOVA analysis

### **Documentation**
- `README_ahi_analysis.md` - Documentation for AHI analysis
- `README.md` - General project documentation

## 🔍 Analysis Types

### **1. Spindle Density vs AHI Analysis**
**Purpose**: Examine relationship between sleep spindle density and AHI (Apnea-Hypopnea Index)

**Files**:
- `analyze_spindle_density_vs_ahi.py` - Main analysis script
- `run_ahi_analysis.py` - Execution script
- `config_ahi_analysis.py` - Configuration
- `README_ahi_analysis.md` - Documentation

**Research Questions**:
1. Does AHI predict spindle density, and how does its impact compare to age?
2. Does AHI change the relationship between age and spindle density (disrupt development)?

**Output**: `analysis_2/` directory

### **2. Spindle Density vs TOVA Analysis**
**Purpose**: Examine relationship between sleep spindle density and TOVA cognitive performance

**Files**:
- `analyze_spindle_density_vs_tova.py` - Main analysis script
- `run_tova_analysis.py` - Execution script
- `config_tova_analysis.py` - Configuration

**Research Questions**:
1. Does spindle density predict morning TOVA performance?
2. Is there a time × spindle density interaction?

**Output**: `analysis/` directory

## 🚀 Quick Start Guide

### **For AHI Analysis**:
```bash
# 1. Configure the analysis
cp config_ahi_analysis.py config.py
# Edit config.py with your project directory

# 2. Run the analysis
python run_ahi_analysis.py --project-dir /path/to/your/002_PEDS_OSA

# 3. Check setup only
python run_ahi_analysis.py --check-only
```

### **For TOVA Analysis**:
```bash
# 1. Configure the analysis
cp config_tova_analysis.py config.py
# Edit config.py with your project directory

# 2. Run the analysis
python run_tova_analysis.py --project-dir /path/to/your/002_PEDS_OSA
```

## 📊 Key Differences Between Analyses

| Aspect | AHI Analysis | TOVA Analysis |
|--------|-------------|---------------|
| **Outcome Variable** | Spindle density | TOVA performance |
| **Main Predictor** | AHI | Spindle density |
| **Focus** | Sleep disorder severity | Cognitive performance |
| **Models** | Linear models, interactions | ANCOVA, LMM |
| **Transformation** | Log-transformed AHI | None |
| **Output Directory** | `analysis_2/` | `analysis/` |

## 🛠️ File Structure

```
anlaysis/
├── analyze_spindle_density_vs_ahi.py      # AHI analysis main script
├── analyze_spindle_density_vs_tova.py     # TOVA analysis main script
├── run_ahi_analysis.py                    # AHI analysis execution
├── run_tova_analysis.py                   # TOVA analysis execution
├── config_ahi_analysis.py                 # AHI analysis configuration
├── config_tova_analysis.py                # TOVA analysis configuration
├── README_ahi_analysis.md                 # AHI analysis documentation
├── README.md                              # General documentation
├── README_file_organization.md            # This file
├── analysis_requirements.txt              # Python dependencies
├── reference/                             # Reference data and documents
└── report_create.py                       # Report generation utilities
```

## 📝 Naming Convention Benefits

1. **Clarity**: File names immediately indicate their purpose
2. **Consistency**: All files follow the same naming pattern
3. **Professional**: Clean, descriptive names suitable for collaboration
4. **Maintainable**: Easy to understand and modify
5. **Scalable**: Easy to add new analysis types following the same pattern

## 🔧 Adding New Analyses

To add a new analysis type, follow this pattern:

1. **Main Script**: `analyze_[outcome]_vs_[predictor].py`
2. **Run Script**: `run_[analysis_type]_analysis.py`
3. **Config**: `config_[analysis_type]_analysis.py`
4. **Documentation**: `README_[analysis_type]_analysis.md`

Example for a new analysis:
- `analyze_spindle_density_vs_bmi.py`
- `run_bmi_analysis.py`
- `config_bmi_analysis.py`
- `README_bmi_analysis.md`

## 📋 Best Practices

1. **Always use descriptive names** that indicate the analysis purpose
2. **Keep file names consistent** across related files
3. **Update documentation** when renaming files
4. **Test scripts** after renaming to ensure imports work correctly
5. **Use version control** to track file name changes

## 🎯 Summary

The new file naming convention makes it immediately clear what each file does:
- **`analyze_*`** = Main analysis scripts
- **`run_*`** = Execution scripts with command-line interface
- **`config_*`** = Configuration files
- **`README_*`** = Documentation files

This organization makes the codebase much more professional and easier for peers to understand and use. 