# Sleep Spindle Analysis Pipeline

A high-throughput pipeline for detecting and analyzing sleep spindles in EEG data.

## Installation

### 1. System Dependencies

First, install the required system dependencies:

```bash
# Install pandoc for PDF report generation
brew install pandoc

# Install BasicTeX (smaller than full TeX Live)
brew install --cask basictex

# Restart your terminal or add TeX to your PATH
export PATH=$PATH:/Library/TeX/texbin
```

### 2. Python Environment

Create and activate a Python virtual environment:

```bash
python -m venv spindles
source spindles/bin/activate  # On Unix/macOS
```

### 3. Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Process a single subject
python main.py --project_dir /path/to/project --subjects sub-001

# Process multiple subjects
python main.py --project_dir /path/to/project --subjects sub-001 sub-002

# Process all available subjects
python main.py --project_dir /path/to/project --all

# Interactive subject selection
python main.py --project_dir /path/to/project --interactive
```

### Output Structure

For each subject, the pipeline creates:

```
derivatives/
└── sub-XXX/
    └── spindles/
        ├── results/          # CSV files and analysis results
        ├── visualizations/   # Generated plots and figures
        ├── logs/            # Processing logs
        └── report/          # Individual subject report (MD + PDF)
```

### Reports

The pipeline automatically generates:
- Individual subject reports (Markdown + PDF)
- Comprehensive visualizations
- Detailed processing logs

## Features

- High-throughput spindle detection using YASA
- Comprehensive visualization suite
- Detailed per-subject logging
- Automatic PDF report generation
- Sequential processing with progress tracking

## Dependencies

- **System Dependencies**:
  - `pandoc`: For PDF report generation
  - `basictex`: For PDF report generation (LaTeX engine)

- **Python Dependencies**: See `requirements.txt`

## License

[Your License Here]


