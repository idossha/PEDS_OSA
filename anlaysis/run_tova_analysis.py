#!/usr/bin/env python3
"""
Complete PEDS OSA Spindle Analysis Workflow
==========================================

This script demonstrates the complete analysis workflow:
1. Run the main analysis pipeline
2. Generate the PDF report

Usage: python run_complete_analysis.py
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        print("✓ Command completed successfully!")
        print(f"Duration: {end_time - start_time:.1f} seconds")
        
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

def main():
    """Run the complete analysis workflow."""
    print("PEDS OSA Spindle Analysis - Complete Workflow")
    print("=" * 60)
    print("This script will run the complete analysis pipeline and generate a PDF report.")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('peds_osa_spindle_analysis.py'):
        print("Error: This script should be run from the analysis directory.")
        print("Please navigate to the analysis directory and run again.")
        sys.exit(1)
    
    # Step 1: Run the main analysis
    print("Step 1: Running main analysis pipeline...")
    success1 = run_command(
        "source ../venv/bin/activate && python peds_osa_spindle_analysis.py",
        "Main Analysis Pipeline"
    )
    
    if not success1:
        print("\n✗ Analysis pipeline failed. Stopping workflow.")
        sys.exit(1)
    
    # Step 2: Generate PDF report
    print("\nStep 2: Generating PDF report...")
    success2 = run_command(
        "source ../venv/bin/activate && python report_create.py",
        "PDF Report Generation"
    )
    
    if not success2:
        print("\n✗ Report generation failed.")
        sys.exit(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("✓ COMPLETE WORKFLOW SUCCESSFUL!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    
    # List generated files
    analysis_dir = "/Volumes/Ido/002_PEDS_OSA/analysis"
    if os.path.exists(analysis_dir):
        print(f"\nAnalysis files in {analysis_dir}:")
        for file in os.listdir(analysis_dir):
            if file.endswith(('.png', '.txt', '.csv')):
                file_path = os.path.join(analysis_dir, file)
                size = os.path.getsize(file_path) / 1024  # KB
                print(f"  - {file} ({size:.1f} KB)")
    
    # List PDF reports in current directory
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    if pdf_files:
        print(f"\nPDF reports in current directory:")
        for pdf_file in pdf_files:
            size = os.path.getsize(pdf_file) / 1024  # KB
            print(f"  - {pdf_file} ({size:.1f} KB)")
    
    print(f"\nWorkflow completed successfully!")
    print("You can now review the analysis results and PDF report.")

if __name__ == "__main__":
    main() 