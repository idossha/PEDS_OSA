#!/usr/bin/env python3
"""
Standalone Report Generator
==========================

Generate markdown and PDF reports from existing spindle analysis results.
This script can be run independently of the main pipeline.

Usage:
    python generate_report.py --project_dir /path/to/project --subject sub-001
    python generate_report.py --project_dir /path/to/project --subject sub-001 --batch-report
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from project_manager import ProjectManager
from report_generator import ReportGenerator

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def create_bash_script(project_dir: str, subject_id: str):
    """Create a bash script for PDF generation using pandoc."""
    
    # Determine paths
    project_path = Path(project_dir)
    subject_report_dir = project_path / 'derivatives' / subject_id / 'spindles' / 'report'
    viz_dir = project_path / 'derivatives' / subject_id / 'spindles' / 'visualizations'
    
    # Find the most recent markdown file
    md_files = list(subject_report_dir.glob('*.md'))
    if not md_files:
        print(f"No markdown files found in {subject_report_dir}")
        return None
    
    # Get the most recent file
    latest_md = max(md_files, key=lambda x: x.stat().st_mtime)
    pdf_file = latest_md.with_suffix('.pdf')
    
    # Create bash script with image copying
    script_content = f"""#!/bin/bash
# Automatic PDF Report Generation Script
# Generated for {subject_id}

set -e  # Exit on any error

echo "=== PDF Report Generation for {subject_id} ==="
echo "Project Directory: {project_dir}"
echo "Report Directory: {subject_report_dir}"
echo "Visualization Directory: {viz_dir}"
echo "Markdown File: {latest_md.name}"
echo "Output PDF: {pdf_file.name}"
echo ""

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "ERROR: pandoc is not installed!"
    echo "Please install it with: brew install pandoc"
    exit 1
fi

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex is not installed!"
    echo "Please install BasicTeX with: brew install --cask basictex"
    echo "Then restart your terminal."
    exit 1
fi

# Change to the report directory
cd "{subject_report_dir}"

# Create images subdirectory in report folder
mkdir -p images

# Copy visualization files to report directory for reliable access
echo "Copying visualization files..."
if [ -f "{viz_dir}/comprehensive_analysis.png" ]; then
    cp "{viz_dir}/comprehensive_analysis.png" images/
    echo "✓ Copied comprehensive_analysis.png"
fi

if [ -f "{viz_dir}/temporal_analysis.png" ]; then
    cp "{viz_dir}/temporal_analysis.png" images/
    echo "✓ Copied temporal_analysis.png"
fi

if [ -f "{viz_dir}/frequency_distribution.png" ]; then
    cp "{viz_dir}/frequency_distribution.png" images/
    echo "✓ Copied frequency_distribution.png"
fi

if [ -f "{viz_dir}/summary_statistics.png" ]; then
    cp "{viz_dir}/summary_statistics.png" images/
    echo "✓ Copied summary_statistics.png"
fi

if [ -f "{viz_dir}/sensor_layout_2d.png" ]; then
    cp "{viz_dir}/sensor_layout_2d.png" images/
    echo "✓ Copied sensor_layout_2d.png"
fi

echo ""
echo "Converting markdown to PDF..."

# Method 1: Try with resource path for images
pandoc "{latest_md.name}" -o "{pdf_file.name}" \\
    --pdf-engine=pdflatex \\
    --variable geometry:margin=1in \\
    --variable fontsize=11pt \\
    --variable documentclass=article \\
    --resource-path=.:images:{viz_dir} \\
    2>/dev/null || {{

    echo "First attempt failed, trying alternative method..."
    
    # Method 2: Create a modified markdown with local image paths
    sed 's|{viz_dir}/|images/|g' "{latest_md.name}" > temp_report.md
    
    pandoc temp_report.md -o "{pdf_file.name}" \\
        --pdf-engine=pdflatex \\
        --variable geometry:margin=1in \\
        --variable fontsize=11pt \\
        --variable documentclass=article
    
    rm temp_report.md
}}

if [ -f "{pdf_file.name}" ]; then
    echo "✓ PDF generated successfully: {pdf_file}"
    echo "✓ Opening PDF..."
    open "{pdf_file.name}" 2>/dev/null || echo "PDF saved (could not open automatically)"
else
    echo "✗ PDF generation failed"
    echo "Trying one more method with HTML intermediate..."
    
    # Method 3: Convert via HTML
    pandoc "{latest_md.name}" -o temp_report.html --resource-path=.:images:{viz_dir}
    pandoc temp_report.html -o "{pdf_file.name}" --pdf-engine=pdflatex
    rm temp_report.html 2>/dev/null || true
    
    if [ -f "{pdf_file.name}" ]; then
        echo "✓ PDF generated via HTML conversion"
        open "{pdf_file.name}" 2>/dev/null || echo "PDF saved"
    else
        echo "✗ All PDF generation methods failed"
        exit 1
    fi
fi

echo "=== Report generation completed ==="
"""
    
    # Save bash script
    script_file = subject_report_dir / f"generate_pdf_{subject_id}.sh"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_file.chmod(0o755)
    
    return script_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate reports from existing spindle analysis results")
    parser.add_argument("--project_dir", required=True, help="Path to project directory")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-001)")
    parser.add_argument("--batch-report", action="store_true", help="Also generate batch report")
    parser.add_argument("--create-bash-script", action="store_true", help="Create bash script for PDF generation")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    try:
        # Initialize project manager
        project_manager = ProjectManager(args.project_dir)
        logger.info(f"Project directory: {args.project_dir}")
        
        # Validate subject
        if not project_manager.validate_subject(args.subject):
            logger.error(f"Subject {args.subject} not found or invalid")
            return 1
        
        # Check if subject has results
        if not project_manager.subject_results_exist(args.subject):
            logger.error(f"No analysis results found for {args.subject}")
            logger.info("Please run the main pipeline first to generate results")
            return 1
        
        # Initialize report generator
        report_generator = ReportGenerator(project_manager)
        
        # Generate individual subject report
        logger.info(f"Generating individual report for {args.subject}")
        success = report_generator.generate_subject_report(args.subject)
        
        if success:
            logger.info("✓ Individual report generated successfully")
        else:
            logger.error("✗ Individual report generation failed")
            return 1
        
        # Generate batch report if requested
        if args.batch_report:
            logger.info("Generating batch report")
            batch_success = report_generator.generate_comprehensive_report([args.subject])
            if batch_success:
                logger.info("✓ Batch report generated successfully")
            else:
                logger.warning("✗ Batch report generation failed")
        
        # Create bash script if requested
        if args.create_bash_script:
            logger.info("Creating bash script for PDF generation")
            script_file = create_bash_script(args.project_dir, args.subject)
            if script_file:
                logger.info(f"✓ Bash script created: {script_file}")
                logger.info(f"Run it with: {script_file}")
            else:
                logger.error("✗ Failed to create bash script")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 