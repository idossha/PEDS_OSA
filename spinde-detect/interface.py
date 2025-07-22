#!/usr/bin/env python3
"""
Interface Module
================

Handles command line interface and user interaction.
"""

import argparse
from typing import List, Tuple

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="High-throughput spindle analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=get_usage_examples()
    )
    
    # Required arguments
    parser.add_argument(
        "--project_dir", 
        type=str, 
        required=True,
        help="Path to the project directory containing subject folders"
    )
    
    # Processing mode arguments (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        "--all", 
        action="store_true",
        help="Process all available subjects"
    )
    
    group.add_argument(
        "--subjects", 
        nargs="+",
        help="List of specific subjects to process (e.g., sub-001 sub-002)"
    )
    
    group.add_argument(
        "--interactive", 
        action="store_true",
        help="Interactive mode to select subjects"
    )
    
    group.add_argument(
        "--list-subjects", 
        action="store_true",
        help="List all available subjects and exit"
    )
    
    # Analysis parameter arguments
    parser.add_argument(
        "--freq_range", 
        nargs=2, 
        type=float, 
        default=[10.0, 16.0],
        help="Frequency range for spindle detection (default: 10 16)"
    )
    
    parser.add_argument(
        "--slow_range", 
        nargs=2, 
        type=float, 
        default=[10.0, 12.0],
        help="Frequency range for slow spindles (default: 10 12)"
    )
    
    parser.add_argument(
        "--fast_range", 
        nargs=2, 
        type=float, 
        default=[12.0, 16.0],
        help="Frequency range for fast spindles (default: 12 16)"
    )
    
    # Preprocessing arguments
    parser.add_argument(
        "--downsample", 
        type=float,
        help="Target sampling rate for downsampling (Hz)"
    )
    
    # Output and reporting arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="derivatives",
        help="Output directory name within project (default: derivatives)"
    )
    
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="Overwrite existing results"
    )
    
    # Report generation arguments
    parser.add_argument(
        "--generate-report", 
        action="store_true",
        help="Generate comprehensive batch markdown and PDF report after processing all subjects"
    )
    
    parser.add_argument(
        "--report-name", 
        type=str, 
        default=None,
        help="Custom name for the generated batch report (default: auto-generated)"
    )
    
    parser.add_argument(
        "--no-individual-subjects", 
        action="store_true",
        help="Exclude individual subject details from the batch report"
    )
    
    parser.add_argument(
        "--no-batch-summary", 
        action="store_true",
        help="Exclude batch summary from the batch report"
    )
    
    parser.add_argument(
        "--no-individual-reports",
        action="store_true", 
        help="Disable automatic individual subject report generation"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log_level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    return parser

def get_usage_examples() -> str:
    """Get usage examples for the help text."""
    return """
Examples:
  # Process all subjects with automatic individual reports
  python main.py --project_dir /data/sleep_study --all
  
  # Process specific subjects with batch report
  python main.py --project_dir /data/sleep_study --subjects sub-001 sub-002 --generate-report
  
  # Interactive mode with custom batch report name
  python main.py --project_dir /data/sleep_study --interactive --generate-report --report-name my_analysis
  
  # List available subjects
  python main.py --project_dir /data/sleep_study --list-subjects
  
  # Process with downsampling and detailed logging
  python main.py --project_dir /data/sleep_study --all --downsample 200 --log_level DEBUG
    """

def validate_arguments(args) -> List[str]:
    """
    Validate parsed arguments and return any validation errors.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments
    
    Returns
    -------
    List[str]
        List of validation errors (empty if no errors)
    """
    errors = []
    
    # Validate frequency ranges
    if args.freq_range[0] >= args.freq_range[1]:
        errors.append("Invalid freq_range: start frequency must be less than end frequency")
    
    if args.slow_range[0] >= args.slow_range[1]:
        errors.append("Invalid slow_range: start frequency must be less than end frequency")
    
    if args.fast_range[0] >= args.fast_range[1]:
        errors.append("Invalid fast_range: start frequency must be less than end frequency")
    
    # Validate that slow and fast ranges don't overlap inappropriately
    if args.slow_range[1] > args.fast_range[0]:
        errors.append("Warning: slow_range and fast_range overlap")
    
    # Validate that ranges are within the detection range
    if args.slow_range[0] < args.freq_range[0] or args.slow_range[1] > args.freq_range[1]:
        errors.append("slow_range must be within freq_range")
    
    if args.fast_range[0] < args.freq_range[0] or args.fast_range[1] > args.freq_range[1]:
        errors.append("fast_range must be within freq_range")
    
    # Validate downsampling rate
    if args.downsample and args.downsample <= 0:
        errors.append("Downsampling rate must be positive")
    
    # Validate report arguments
    if (args.no_individual_subjects or args.no_batch_summary or args.report_name) and not args.generate_report:
        errors.append("Batch report options require --generate-report flag")
    
    return errors

def list_subjects(available_subjects: List[str], project_dir: str):
    """List available subjects and their EEG files."""
    print(f"\nFound {len(available_subjects)} subjects in {project_dir}:")
    for subject in available_subjects:
        print(f"  {subject}")

def interactive_subject_selection(available_subjects: List[str]) -> List[str]:
    """
    Interactive selection of subjects to process.
    
    Parameters
    ----------
    available_subjects : List[str]
        List of available subject IDs
    
    Returns
    -------
    List[str]
        Selected subject IDs
    """
    print("\nAvailable subjects:")
    for i, subject in enumerate(available_subjects, 1):
        print(f"  {i:2d}. {subject}")
    
    print("\nSelect subjects to process:")
    print("  - Enter numbers separated by spaces (e.g., 1 3 5)")
    print("  - Enter 'all' to select all subjects")
    print("  - Enter 'quit' to exit")
    
    while True:
        selection = input("\nYour choice: ").strip().lower()
        
        if selection == 'quit':
            return []
        elif selection == 'all':
            return available_subjects
        else:
            try:
                indices = [int(x) - 1 for x in selection.split()]
                selected = [available_subjects[i] for i in indices 
                           if 0 <= i < len(available_subjects)]
                
                if selected:
                    print(f"\nSelected subjects: {', '.join(selected)}")
                    confirm = input("Proceed? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        return selected
                else:
                    print("Invalid selection. Please try again.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter numbers, 'all', or 'quit'.")

def validate_requested_subjects(requested_subjects: List[str], 
                               available_subjects: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate that requested subjects exist.
    
    Parameters
    ----------
    requested_subjects : List[str]
        List of requested subject IDs
    available_subjects : List[str]
        List of available subject IDs
    
    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple of (valid_subjects, invalid_subjects)
    """
    valid_subjects = [s for s in requested_subjects if s in available_subjects]
    invalid_subjects = [s for s in requested_subjects if s not in available_subjects]
    
    return valid_subjects, invalid_subjects

def get_argument_summary(args) -> str:
    """
    Get a formatted summary of the parsed arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments
    
    Returns
    -------
    str
        Formatted argument summary
    """
    summary = "=== Pipeline Configuration ===\n"
    summary += f"Project Directory: {args.project_dir}\n"
    summary += f"Output Directory: {args.output_dir}\n"
    summary += f"Detection Range: {args.freq_range[0]}-{args.freq_range[1]} Hz\n"
    summary += f"Slow Spindles: {args.slow_range[0]}-{args.slow_range[1]} Hz\n"
    summary += f"Fast Spindles: {args.fast_range[0]}-{args.fast_range[1]} Hz\n"
    
    if args.downsample:
        summary += f"Downsampling: {args.downsample} Hz\n"
    
    summary += f"Log Level: {args.log_level}\n"
    summary += f"Overwrite: {args.overwrite}\n"
    
    # Processing mode
    if args.all:
        summary += "Processing Mode: All subjects\n"
    elif args.subjects:
        summary += f"Processing Mode: Specific subjects ({', '.join(args.subjects)})\n"
    elif args.interactive:
        summary += "Processing Mode: Interactive selection\n"
    elif getattr(args, 'list_subjects', False):
        summary += "Processing Mode: List subjects only\n"
    
    # Report options
    no_individual_reports = getattr(args, 'no_individual_reports', False)
    summary += f"Individual Reports: {'Disabled' if no_individual_reports else 'Enabled'}\n"
    if args.generate_report:
        summary += "Batch Report Generation: Enabled\n"
        if args.report_name:
            summary += f"Batch Report Name: {args.report_name}\n"
        if args.no_individual_subjects:
            summary += "Batch Report: Exclude individual subjects\n"
        if args.no_batch_summary:
            summary += "Batch Report: Exclude batch summary\n"
    else:
        summary += "Batch Report Generation: Disabled\n"
    
    summary += "=" * 30 + "\n"
    return summary 