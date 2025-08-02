#!/usr/bin/env python3
"""
PEDS OSA Spindle Analysis Report Generator
==========================================

This script generates a comprehensive PDF report from the analysis outputs.
It combines visualizations, statistical results, and creates a professional document.

Author: PEDS OSA Analysis Team
Date: 2025
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Import reportlab components
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas

def read_text_file(file_path):
    """Read and return the contents of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def create_title_page(canvas, doc):
    """Create a professional title page."""
    canvas.saveState()
    
    # Set font and size for title
    canvas.setFont("Helvetica-Bold", 24)
    canvas.setFillColor(colors.darkblue)
    
    # Title
    title = "PEDS OSA Spindle Analysis Report"
    canvas.drawCentredString(4.25*inch, 9*inch, title)
    
    # Subtitle
    canvas.setFont("Helvetica", 16)
    canvas.setFillColor(colors.grey)
    subtitle = "Sleep Spindle Density and TOVA Performance in Children"
    canvas.drawCentredString(4.25*inch, 8.5*inch, subtitle)
    
    # Date
    canvas.setFont("Helvetica", 12)
    canvas.setFillColor(colors.black)
    date_str = f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    canvas.drawCentredString(4.25*inch, 7.5*inch, date_str)
    
    # Analysis information
    canvas.setFont("Helvetica", 10)
    info_lines = [
        "Analysis Pipeline: peds_osa_spindle_analysis.py",
        "Report Generator: report_create.py",
        "Statistical Models: ANCOVA, Linear Mixed Models",
        "Key Finding: Time × Spindle Density Interaction (p = 0.020)"
    ]
    
    y_pos = 6.5*inch
    for line in info_lines:
        canvas.drawCentredString(4.25*inch, y_pos, line)
        y_pos -= 0.3*inch
    
    canvas.restoreState()

def create_header_page(canvas, doc):
    """Create a header for content pages."""
    canvas.saveState()
    
    # Add a simple header with report title and page number
    canvas.setFont("Helvetica-Bold", 10)
    canvas.setFillColor(colors.darkblue)
    canvas.drawString(72, 750, "PEDS OSA Spindle Analysis Report")
    
    # Add page number
    canvas.drawRightString(522, 750, f"Page {doc.page}")
    
    # Add a line separator
    canvas.setStrokeColor(colors.darkblue)
    canvas.setLineWidth(1)
    canvas.line(72, 745, 522, 745)
    
    canvas.restoreState()

def create_executive_summary(styles):
    """Create executive summary section."""
    story = []
    
    # Section header
    story.append(Paragraph("Executive Summary", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    # Summary text
    summary_text = """
    This report presents the analysis of sleep spindle characteristics and their relationship 
    with cognitive performance (TOVA test) in children with Obstructive Sleep Apnea (OSA) 
    compared to healthy controls. The study examined 53 subjects (34 OSA, 19 Control) aged 
    5-11 years, analyzing spindle density and TOVA performance across morning and evening testing sessions.
    
    <b>Key Findings:</b>
    • Significant interaction between time and spindle density (p = 0.020)
    • Spindle density moderates the effect of time on TOVA performance
    • No significant group differences in spindle density or TOVA improvement
    • Baseline TOVA performance significantly predicts morning performance (p = 0.002)
    
    The analysis employed both ANCOVA and Linear Mixed Models to examine the relationships 
    between spindle characteristics, cognitive performance, and clinical group status.
    """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    return story

def create_methods_section(styles):
    """Create methods section."""
    story = []
    
    story.append(Paragraph("Methods", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    methods_text = """
    <b>Study Design:</b> Cross-sectional analysis of sleep spindle characteristics and cognitive performance.
    
    <b>Participants:</b> 53 children (5-11 years) including 34 with OSA and 19 healthy controls.
    
    <b>Data Collection:</b>
    • Sleep spindle detection from overnight polysomnography
    • TOVA performance testing (PM and AM sessions)
    • Demographic and clinical information
    
    <b>Statistical Analysis:</b>
    • ANCOVA: TOVA_AM ~ TOVA_PM + spindle_density + age + sex
    • Linear Mixed Model: TOVA_score ~ Time × spindle_density + age + sex + (1|ID)
    • Group comparisons: Independent t-tests
    • Visualization: Publication-ready plots with statistical annotations
    """
    
    story.append(Paragraph(methods_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    return story

def create_variable_definitions(styles):
    """Create detailed variable definitions section."""
    story = []
    
    story.append(Paragraph("Variable Definitions and Analysis Parameters", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    definitions_text = """
    <b>Spindle Density Variables:</b>
    
    <b>• Spindle Density (spindles/min):</b>
    - Definition: Number of detected spindles per minute of sleep
    - Calculation: Total spindles / Total sleep time (minutes)
    - Range in dataset: 8.377 - 513.071 spindles/min
    - Mean ± SD: 196.879 ± 114.297 spindles/min
    
    <b>• High vs Low Spindle Density Categories:</b>
    - Method: Equal-frequency binning (2 categories)
    - Cutoff: Median split of spindle density values
    - Low category: ≤ 196.879 spindles/min (median)
    - High category: > 196.879 spindles/min (median)
    - Rationale: Ensures equal sample sizes in each category for balanced analysis
    
    <b>• Spindle Type Classification:</b>
    - Slow spindles: 11-13 Hz frequency range
    - Fast spindles: 13-16 Hz frequency range
    - Classification based on peak frequency of each detected spindle
    
    <b>TOVA Performance Variables:</b>
    
    <b>• TOVA Score (DPRIME values):</b>
    - Source: DPRIMEQ1 and DPRIMEQ4 columns from TOVA.csv
    - Definition: D-prime (d') values from Test of Variables of Attention
    - DPRIMEQ1: First quarter performance (PM/evening session)
    - DPRIMEQ4: Fourth quarter performance (AM/morning session)
    - Scale: Continuous variable (range in dataset: 0.67 - 8.53)
    - Higher d' values indicate better signal detection performance
    - D-prime measures sensitivity to target stimuli vs. distractors
    
    <b>• TOVA Score Mapping:</b>
    - TOVA_PM = DPRIMEQ1 (evening baseline performance)
    - TOVA_AM = DPRIMEQ4 (morning post-sleep performance)
    - Rationale: Q1 represents initial performance, Q4 represents final performance
    
    <b>• TOVA Improvement:</b>
    - Calculation: TOVA_PM - TOVA_AM (DPRIMEQ1 - DPRIMEQ4)
    - Interpretation: Positive values indicate performance decline from PM to AM
    - Negative values indicate performance improvement from PM to AM
    - Range in dataset: -6.11 to +4.20
    
    <b>• Additional TOVA Metrics Available:</b>
    - DPRIMEQ2, DPRIMEQ3: Intermediate quarter performances
    - Simple Difference: DPRIMEQ1 - DPRIMEQ4 (same as our TOVA improvement)
    - Average Score Change: Mean change across quarters
    - Overall Percentage Change: Percentage change from Q1 to Q4
    - Z-Score Change: Standardized change relative to group
    - Cohen's d: Effect size of the change
    
    <b>Demographic Variables:</b>
    
    <b>• Age:</b>
    - Range: 5.0 - 11.0 years
    - Mean ± SD: 8.0 ± 1.6 years
    - Continuous variable in years
    
    <b>• Sex:</b>
    - Categories: M (Male), F (Female)
    - Encoding: M = 1, F = 0 for statistical analysis
    
    <b>• Group:</b>
    - Categories: Control, OSA (Obstructive Sleep Apnea)
    - Based on clinical diagnosis and polysomnography results
    
    <b>Statistical Analysis Parameters:</b>
    
    <b>• ANCOVA Model:</b>
    - Dependent variable: TOVA_AM (morning performance)
    - Covariates: TOVA_PM, spindle_density, age, sex
    - Assumptions: Normality, homogeneity of variance, linearity
    
    <b>• Linear Mixed Model:</b>
    - Dependent variable: TOVA_score (both PM and AM)
    - Fixed effects: Time (PM/AM), spindle_density, age, sex, Time × spindle_density
    - Random effect: Subject ID (1|ID)
    - Method: REML (Restricted Maximum Likelihood)
    
    <b>• Group Comparisons:</b>
    - Test: Independent t-tests
    - Variables: spindle_density, TOVA_improvement
    - Assumption: Equal variances (Levene's test)
    
    <b>• Visualization Parameters:</b>
    - Color scheme: #2E86AB (blue), #A23B72 (magenta)
    - Figure resolution: 300 DPI
    - Statistical annotations: p-values, correlation coefficients
    - Error bars: Standard error of the mean
    """
    
    story.append(Paragraph(definitions_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    return story

def create_descriptive_statistics(styles, analysis_dir):
    """Create descriptive statistics section with the visualization."""
    story = []
    
    story.append(Paragraph("Descriptive Statistics", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    # Add the descriptive statistics figure
    desc_fig_path = os.path.join(analysis_dir, 'descriptive_info.png')
    if os.path.exists(desc_fig_path):
        img = Image(desc_fig_path, width=7*inch, height=5.25*inch)
        img.hAlign = 'CENTER'
        story.append(img)
        story.append(Spacer(1, 12))
        
        caption = """
        <b>Figure 1: Descriptive Statistics Overview</b><br/>
        Panel A: Age distribution by group; Panel B: Sex distribution; Panel C: Group distribution;
        Panel D: Spindle density by group; Panel E: TOVA PM scores by group; Panel F: TOVA AM scores by group;
        Panel G: Overall sample characteristics summary.
        """
        story.append(Paragraph(caption, styles['Normal']))
        story.append(Spacer(1, 12))
    
    return story

def create_interaction_analysis(styles, analysis_dir):
    """Create interaction analysis section with the visualization."""
    story = []
    
    story.append(Paragraph("Interaction Analysis", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    # Add the interaction figure
    interaction_fig_path = os.path.join(analysis_dir, 'publication_interaction_plot.png')
    if os.path.exists(interaction_fig_path):
        img = Image(interaction_fig_path, width=7*inch, height=3*inch)
        img.hAlign = 'CENTER'
        story.append(img)
        story.append(Spacer(1, 12))
        
        caption = """
        <b>Figure 2: Time × Spindle Density Interaction</b><br/>
        Panel A: TOVA Performance interaction plot showing how spindle density moderates the effect of time on performance.
        Panel B: Scatter plot showing the relationship between spindle density and TOVA performance change (PM - AM).
        Significant interaction effect (p = 0.020) indicates that spindle density moderates the time effect on performance.
        """
        story.append(Paragraph(caption, styles['Normal']))
        story.append(Spacer(1, 12))
    
    return story

def create_statistical_results(styles, analysis_dir):
    """Create statistical results section with text files."""
    story = []
    
    story.append(Paragraph("Statistical Results", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    # ANCOVA Results
    story.append(Paragraph("ANCOVA Analysis", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    ancova_path = os.path.join(analysis_dir, 'ancova_results.txt')
    ancova_text = read_text_file(ancova_path)
    story.append(Paragraph(f"<pre>{ancova_text}</pre>", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # LMM Results
    story.append(Paragraph("Linear Mixed Model Analysis", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    lmm_path = os.path.join(analysis_dir, 'lmm_results.txt')
    lmm_text = read_text_file(lmm_path)
    story.append(Paragraph(f"<pre>{lmm_text}</pre>", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Group Analysis
    story.append(Paragraph("Group Comparison Analysis", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    group_path = os.path.join(analysis_dir, 'group_analysis_results.txt')
    group_text = read_text_file(group_path)
    story.append(Paragraph(f"<pre>{group_text}</pre>", styles['Normal']))
    story.append(Spacer(1, 12))
    
    return story

def create_comprehensive_summary(styles, analysis_dir):
    """Create comprehensive summary section."""
    story = []
    
    story.append(Paragraph("Comprehensive Analysis Summary", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    summary_path = os.path.join(analysis_dir, 'comprehensive_analysis_summary.txt')
    summary_text = read_text_file(summary_path)
    story.append(Paragraph(f"<pre>{summary_text}</pre>", styles['Normal']))
    story.append(Spacer(1, 12))
    
    return story

def create_conclusions(styles):
    """Create conclusions section."""
    story = []
    
    story.append(Paragraph("Conclusions", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    conclusions_text = """
    <b>Primary Findings:</b>
    • A significant Time × Spindle Density interaction (p = 0.020) was observed
    • Spindle density moderates the effect of time on TOVA performance
    • Higher spindle density is associated with different patterns of performance change
    
    <b>Clinical Implications:</b>
    • Sleep spindle characteristics may serve as biomarkers for cognitive performance
    • Individual differences in spindle density may predict cognitive outcomes
    • Further research is needed to understand the mechanisms underlying this interaction
    
    <b>Methodological Strengths:</b>
    • Comprehensive statistical modeling with multiple approaches
    • Publication-ready visualizations with statistical annotations
    • Robust data processing and quality control measures
    
    <b>Future Directions:</b>
    • Longitudinal studies to examine temporal relationships
    • Intervention studies targeting spindle characteristics
    • Integration with other sleep and cognitive measures
    """
    
    story.append(Paragraph(conclusions_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    return story

def create_appendix(styles, analysis_dir):
    """Create appendix with data files information."""
    story = []
    
    story.append(Paragraph("Appendix: Data Files", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    appendix_text = """
    <b>Generated Data Files:</b>
    • analysis_dataset_wide.csv - Wide format dataset with one row per subject
    • analysis_dataset_long.csv - Long format dataset for mixed models
    • spindle_density.csv - Intermediate spindle density calculations
    
    <b>Statistical Output Files:</b>
    • ancova_results.txt - ANCOVA analysis results
    • lmm_results.txt - Linear Mixed Model results
    • group_analysis_results.txt - Group comparison statistics
    • comprehensive_analysis_summary.txt - Complete analysis summary
    
    <b>Visualization Files:</b>
    • publication_interaction_plot.png - Publication-ready interaction plot
    • descriptive_info.png - Comprehensive descriptive statistics visualization
    
    All files are located in the analysis directory and can be used for further analysis or publication.
    """
    
    story.append(Paragraph(appendix_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    return story

def generate_report(analysis_dir, output_dir):
    """Generate the complete PDF report."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"PEDS_OSA_Spindle_Analysis_Report_{timestamp}.pdf"
    report_path = os.path.join(output_dir, report_filename)
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        report_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkblue
    ))
    
    # Build the story
    story = []
    
    # Add page break for title page
    story.append(PageBreak())
    
    # Executive Summary
    story.extend(create_executive_summary(styles))
    story.append(PageBreak())
    
    # Methods
    story.extend(create_methods_section(styles))
    story.append(PageBreak())
    
    # Variable Definitions
    story.extend(create_variable_definitions(styles))
    story.append(PageBreak())
    
    # Descriptive Statistics
    story.extend(create_descriptive_statistics(styles, analysis_dir))
    story.append(PageBreak())
    
    # Interaction Analysis
    story.extend(create_interaction_analysis(styles, analysis_dir))
    story.append(PageBreak())
    
    # Statistical Results
    story.extend(create_statistical_results(styles, analysis_dir))
    story.append(PageBreak())
    
    # Comprehensive Summary
    story.extend(create_comprehensive_summary(styles, analysis_dir))
    story.append(PageBreak())
    
    # Conclusions
    story.extend(create_conclusions(styles))
    story.append(PageBreak())
    
    # Appendix
    story.extend(create_appendix(styles, analysis_dir))
    
    # Build the PDF
    doc.build(story, onFirstPage=create_title_page, onLaterPages=create_header_page)
    
    print(f"✓ Report generated successfully: {report_path}")
    return report_path

def main():
    """Main function to generate the report."""
    print("PEDS OSA Spindle Analysis Report Generator")
    print("=" * 50)
    
    # Get the current directory (analysis directory)
    current_dir = os.getcwd()
    
    # Check if we're in the analysis directory
    if not os.path.exists(os.path.join(current_dir, 'peds_osa_spindle_analysis.py')):
        print("Error: This script should be run from the analysis directory.")
        print("Please navigate to the analysis directory and run again.")
        sys.exit(1)
    
    # Define paths - look for analysis files in the project analysis directory
    analysis_dir = "/Volumes/Ido/002_PEDS_OSA/analysis"
    output_dir = analysis_dir  # Save report to the same directory as analysis outputs
    
    # Check if analysis files exist
    required_files = [
        'descriptive_info.png',
        'publication_interaction_plot.png',
        'ancova_results.txt',
        'lmm_results.txt',
        'group_analysis_results.txt',
        'comprehensive_analysis_summary.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(analysis_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print("Error: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run peds_osa_spindle_analysis.py first to generate the analysis outputs.")
        sys.exit(1)
    
    print("✓ All required analysis files found.")
    print("Generating PDF report...")
    
    try:
        report_path = generate_report(analysis_dir, output_dir)
        print(f"\n✓ Report generation completed successfully!")
        print(f"Report saved to: {report_path}")
        print(f"File size: {os.path.getsize(report_path) / 1024:.1f} KB")
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 