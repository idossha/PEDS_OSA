#!/usr/bin/env python3
"""
Report Module
=============

Creates markdown and PDF reports from spindle analysis results.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json
import demographics

def generate_subject_report(project_dir: Path, derivatives_dir: Path, 
                           subject_id: str) -> Optional[Path]:
    """
    Generate markdown and PDF report for a single subject.
    
    Parameters
    ----------
    project_dir : Path
        Path to project directory
    derivatives_dir : Path
        Path to derivatives directory
    subject_id : str
        Subject identifier
    
    Returns
    -------
    Optional[Path]
        Path to generated markdown file, or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[{subject_id}] Generating subject report...")
        
        # Get subject directories
        subject_output_dir = derivatives_dir / subject_id / 'spindles'
        report_dir = subject_output_dir / 'report'
        report_dir.mkdir(exist_ok=True)
        
        # Load subject data
        summary_file = subject_output_dir / 'results' / 'spindle_summary.csv'
        if not summary_file.exists():
            logger.error(f"[{subject_id}] Summary file not found: {summary_file}")
            return None
        
        # Load summary data (simplified approach)
        import pandas as pd
        summary_df = pd.read_csv(summary_file, index_col=0)
        stats = summary_df.iloc[0].to_dict()
        
        # Load demographics information
        subject_demographics = demographics.load_subject_demographics(project_dir, subject_id)
        
        # Generate markdown content
        markdown_content = _generate_subject_markdown(subject_id, stats, subject_output_dir, subject_demographics)
        
        # Save markdown file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_file = report_dir / f"{subject_id}_report_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"[{subject_id}] ✓ Markdown report saved: {md_file}")
        
        # Try to generate PDF
        pdf_file = md_file.with_suffix('.pdf')
        if _generate_pdf(md_file, pdf_file, subject_id):
            logger.info(f"[{subject_id}] ✓ PDF report saved: {pdf_file}")
        
        return md_file
        
    except Exception as e:
        logger.error(f"[{subject_id}] Error generating report: {str(e)}")
        return None

def _generate_subject_markdown(subject_id: str, stats: Dict, output_dir: Path, subject_demographics: Optional[Dict] = None) -> str:
    """Generate markdown content for subject report."""
    timestamp = datetime.now().strftime("%B %d, %Y")
    
    # Generate demographics section
    demographics_section = demographics.format_demographics_for_report(subject_demographics)
    
    # Use relative paths for images (relative to report directory)
    content = f"""# Sleep Spindle Analysis Report - {subject_id}
*Generated: {timestamp}*

{demographics_section}## Subject Overview
- **Subject ID**: {subject_id}
- **Analysis Date**: {timestamp}
- **Detection Range**: 10-16 Hz
- **Spindle Classification**: Slow (10-12 Hz), Fast (12-16 Hz)

## Technical Details

### Analysis Parameters
- **Frequency Range**: 10-16 Hz
- **Slow Spindles**: 10-12 Hz
- **Fast Spindles**: 12-16 Hz
- **Minimum Duration**: 0.5 seconds
- **Maximum Duration**: 2.0 seconds


## Visualizations

### Comprehensive Analysis
![{subject_id} Comprehensive Analysis](../visualizations/comprehensive_analysis.png)
*2×3 comprehensive analysis showing topographic distributions (top row) and channel counts (bottom row).*

## Analysis Results

### Summary Statistics
- **Total Spindles Detected**: {int(stats.get('Total_Spindles', 0)):,}
- **Slow Spindles**: {int(stats.get('Slow_Spindles', 0)):,} ({stats.get('Slow_Percentage', 0):.1f}%)
- **Fast Spindles**: {int(stats.get('Fast_Spindles', 0)):,} ({stats.get('Fast_Percentage', 0):.1f}%)



### Spindle Density Over Time
![{subject_id} Spindle Density](../visualizations/spindle_density.png)
*Spindle density plot showing spindles per minute throughout the recording.*

### Additional Visualizations
![{subject_id} Spectrogram](../visualizations/spectrogram.png)
*Spectrogram of EEG data showing frequency content over time.*

![{subject_id} Sensor Layout](../visualizations/sensor_layout_2d.png)
*2D layout of EEG sensor positions.*

---

### Software
- **YASA**: Yet Another Spindle Algorithm for spindle detection
- **MNE-Python**: EEG data loading and preprocessing
- **Analysis Pipeline**: Custom high-throughput spindle analysis pipeline

---

*Analysis completed on {timestamp}*
"""
    
    return content

def _generate_pdf(md_file: Path, pdf_file: Path, subject_id: str) -> bool:
    """Generate PDF from markdown using pandoc."""
    logger = logging.getLogger(__name__)
    
    try:
        # Check if pandoc is available
        result = subprocess.run(['pandoc', '--version'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            logger.warning(f"[{subject_id}] pandoc not available, skipping PDF generation")
            return False
        
        # Generate PDF
        result = subprocess.run([
            'pandoc', str(md_file),
            '-o', str(pdf_file),
            '--pdf-engine=pdflatex',
            '--variable', 'geometry:margin=1in'
        ], capture_output=True, text=True, cwd=md_file.parent, timeout=60)
        
        if result.returncode == 0 and pdf_file.exists():
            return True
        else:
            logger.warning(f"[{subject_id}] PDF generation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.warning(f"[{subject_id}] PDF generation error: {str(e)}")
        return False

def generate_batch_report(project_dir: Path, derivatives_dir: Path, 
                         processed_subjects: List[str], 
                         report_name: Optional[str] = None) -> Optional[Path]:
    """Generate batch report for multiple subjects."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Generating batch report for {len(processed_subjects)} subjects...")
        
        # Create report directory
        report_dir = derivatives_dir / 'reports'
        report_dir.mkdir(exist_ok=True)
        
        # Generate report name
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"spindle_analysis_report_{timestamp}"
        
        # Generate markdown content
        markdown_content = _generate_batch_markdown(processed_subjects, derivatives_dir, project_dir)
        
        # Save markdown file
        md_file = report_dir / f"{report_name}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"✓ Batch markdown report saved: {md_file}")
        
        # Try to generate PDF
        pdf_file = md_file.with_suffix('.pdf')
        if _generate_pdf(md_file, pdf_file, "batch"):
            logger.info(f"✓ Batch PDF report saved: {pdf_file}")
        
        return md_file
        
    except Exception as e:
        logger.error(f"Error generating batch report: {str(e)}")
        return None

def _generate_batch_markdown(processed_subjects: List[str], derivatives_dir: Path, project_dir: Path) -> str:
    """Generate markdown content for batch report."""
    timestamp = datetime.now().strftime("%B %d, %Y")
    
    # Load demographics data once for all subjects
    demographics_df = demographics.load_demographics(project_dir)
    
    content = f"""# Sleep Spindle Analysis Batch Report
*Generated: {timestamp}*

## Dataset Overview
- **Analysis Date**: {timestamp}
- **Total Subjects Processed**: {len(processed_subjects)}
- **Detection Range**: 10-16 Hz
- **Spindle Classification**: Slow (10-12 Hz), Fast (12-16 Hz)

---

## Processed Subjects
"""
    
    # Add detailed subject information including demographics
    for i, subject_id in enumerate(processed_subjects, 1):
        subject_demographics = None
        if demographics_df is not None:
            subject_demographics = demographics.get_subject_demographics(demographics_df, subject_id)
        
        content += f"### {i}. {subject_id}\n"
        
        if subject_demographics:
            content += f"- **Age**: {subject_demographics['age_rounded']} years\n"
            content += f"- **Group**: {subject_demographics['group']}\n"
            content += f"- **NREM OAI**: {subject_demographics['nrem_oai']}\n"
            content += f"- **NREM AHI**: {subject_demographics['nrem_ahi']}\n"
        else:
            content += "- *Demographics information not available*\n"
        
        content += "\n"
    
    content += f"""
---

## Analysis Summary

This batch analysis processed {len(processed_subjects)} subjects using automated sleep spindle detection. Each subject was analyzed using the YASA algorithm with consistent parameters across all recordings.

### Methodology
- **Data Processing**: EEG data loaded from EEGLAB .set files
- **Spindle Detection**: YASA algorithm with 10-16 Hz frequency range
- **Classification**: Spindles classified as slow (10-12 Hz) or fast (12-16 Hz)
- **Quality Control**: Automated validation and error checking

---

*Batch analysis completed on {timestamp}*
"""
    
    return content 