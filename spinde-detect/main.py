#!/usr/bin/env python3
"""
High-throughput Spindle Analysis Pipeline
==========================================

A functional pipeline for batch processing of EEG data to detect and analyze sleep spindles.
Following the SW-detect information flow pattern with spindle-detect structure.

Author: Spindle Analysis Pipeline
"""

import sys
import os
import logging
from pathlib import Path

# Import only the interface and parser modules at startup
import interface
import parser


def setup_logging(output_dir):
    """
    Set up logging to both console and a log file in the output directory.
    Configures the root logger to capture all logs from all modules.
    Following SW-detect pattern.
    """
    log_file = os.path.join(output_dir, "spindle_processing.log")

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers to prevent duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode='w')  # 'w' overwrites each run
    fh.setLevel(logging.INFO)
    
    # Console (stdout) handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Common formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    root_logger.addHandler(fh)
    root_logger.addHandler(ch)

    root_logger.info("Logger has been set up. Writing logs to: %s", log_file)


def process_single_subject(subject_id: str, project_dir: Path, derivatives_dir: Path, args) -> bool:
    """Process a single subject following SW-detect information flow pattern."""
    
    # 1. Setup subject output structure
    output_dirs = parser.create_subject_output_structure(derivatives_dir, subject_id)
    
    # 2. Set up logging per subject (following SW-detect pattern)
    setup_logging(output_dirs['logs'])
    logger = logging.getLogger(__name__)

    logger.info("================================================")
    logger.info("Starting spindle analysis for subject: '%s'", subject_id)

    try:
        # 3. Validate subject
        if not parser.validate_subject(project_dir, subject_id):
            logger.error("Subject validation failed for '%s'", subject_id)
            return False
        
        # 4. Check if results already exist
        if parser.subject_results_exist(derivatives_dir, subject_id) and not args.overwrite:
            logger.info("Results already exist for '%s', skipping (use --overwrite to force)", subject_id)
            return True

        # 5. Import modules AFTER logging is set up (following SW-detect pattern)
        import loader
        import pre_process
        import pre_detection
        import spindle_detect
        import df_create
        import post_detection
        import report

        # 6. STEP 1: Load EEG data
        logger.info("STEP 1) Loading EEG data...")
        eeg_file = parser.get_subject_eeg_file(project_dir, subject_id)
        raw, data = loader.load_and_validate_subject_data(eeg_file, subject_id)
        
        if raw is None or data is None:
            logger.error("Failed to load EEG data for subject '%s'", subject_id)
            return False

        # 7. STEP 2: Preprocessing
        logger.info("STEP 2) Preprocessing...")
        if not pre_process.validate_preprocessing_parameters(args.downsample, raw.info['sfreq'], subject_id):
            logger.error("Invalid preprocessing parameters for subject '%s'", subject_id)
            return False
        
        raw, data = pre_process.apply_preprocessing(raw, data, subject_id, args.downsample)
        if raw is None or data is None:
            logger.error("Preprocessing failed for subject '%s'", subject_id)
            return False

        # 8. STEP 3: Pre-detection analysis
        logger.info("STEP 3) Pre-detection analysis...")
        pre_detection_results = pre_detection.run_pre_detection_analysis(raw, subject_id, output_dirs)

        # 9. STEP 4: Spindle detection
        logger.info("STEP 4) Detecting spindles...")
        spindle_results = spindle_detect.detect_spindles(
            data, raw.info['sfreq'], raw.info['ch_names'], 
            subject_id, tuple(args.freq_range)
        )
        
        if spindle_results is None:
            logger.warning("Spindle detection returned no results for subject '%s'", subject_id)

        # 10. STEP 5: Analyze spindle types
        logger.info("STEP 5) Analyzing spindle types...")
        analysis_results = spindle_detect.analyze_spindle_types(
            spindle_results, subject_id, 
            tuple(args.slow_range), tuple(args.fast_range)
        )
        
        if analysis_results is None:
            logger.error("Spindle analysis failed for subject '%s'", subject_id)
            return False

        # 11. STEP 6: Create dataframes
        logger.info("STEP 6) Creating dataframes...")
        dataframes = df_create.create_spindle_dataframes(analysis_results, subject_id)
        saved_dfs = df_create.save_dataframes(dataframes, output_dirs['results'], subject_id, args.overwrite)

        # 12. STEP 7: Post-detection visualizations
        logger.info("STEP 7) Creating visualizations...")
        viz_results = post_detection.save_all_visualizations(
            analysis_results, raw, output_dirs['visualizations'], subject_id
        )

        # 13. STEP 8: Generate individual report
        if not getattr(args, 'no_individual_reports', False):
            logger.info("STEP 8) Generating individual report...")
            report.generate_subject_report(project_dir, derivatives_dir, subject_id)

        # 14. Log completion
        stats = analysis_results.get('statistics', {})
        logger.info(
            "Processing of subject '%s' completed. %d spindles detected.",
            subject_id, stats.get('total_spindles', 0)
        )

        return True

    except Exception as e:
        logger.error(
            "Error processing subject '%s': %s",
            subject_id, e,
            exc_info=True  # Stack trace in the log
        )
        return False


def main() -> int:
    """Main entry point following SW-detect pattern with spindle-detect structure."""
    # Parse arguments using existing interface
    argument_parser = interface.create_argument_parser()
    args = argument_parser.parse_args()
    
    # Validate arguments using existing interface
    validation_errors = interface.validate_arguments(args)
    if validation_errors:
        print("Argument validation failed:")
        for error in validation_errors:
            print(f"  - {error}")
        return 1

    # Setup basic logging for main process (before per-subject logging)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    main_logger = logging.getLogger(__name__)
    
    try:
        # Validate and setup project directory using existing parser
        project_dir = parser.validate_project_directory(args.project_dir)
        derivatives_dir = parser.create_output_directories(project_dir, args.output_dir)
        
        main_logger.info("Starting spindle analysis pipeline")
        main_logger.info(f"Project directory: {args.project_dir}")
        
        # Find available subjects using existing parser
        available_subjects = parser.find_subjects(project_dir)
        if not available_subjects:
            main_logger.error("No subjects with EEG data found")
            return 1
        
        # Handle list-subjects mode
        if getattr(args, 'list_subjects', False):
            interface.list_subjects(available_subjects, args.project_dir)
            return 0
        
        # Determine subjects to process using existing interface logic
        if args.all:
            subjects_to_process = available_subjects
        elif args.subjects:
            valid_subjects, invalid_subjects = interface.validate_requested_subjects(
                args.subjects, available_subjects
            )
            if invalid_subjects:
                print(f"Warning: Invalid subjects ignored: {', '.join(invalid_subjects)}")
            subjects_to_process = valid_subjects
        elif args.interactive:
            subjects_to_process = interface.interactive_subject_selection(available_subjects)
        else:
            main_logger.error("No processing mode specified")
            return 1
        
        if not subjects_to_process:
            main_logger.error("No valid subjects selected for processing")
            return 1

        # Process all subjects using SW-detect sequential pattern
        main_logger.info(f"Processing {len(subjects_to_process)} subjects")
        
        successful = []
        failed = []
        
        for i, subject_id in enumerate(subjects_to_process, 1):
            main_logger.info(f"Processing subject {i}/{len(subjects_to_process)}: {subject_id}")
            
            try:
                success = process_single_subject(subject_id, project_dir, derivatives_dir, args)
                if success:
                    successful.append(subject_id)
                else:
                    failed.append(subject_id)
            except KeyboardInterrupt:
                main_logger.info("Processing interrupted by user")
                break
            except Exception as e:
                main_logger.error(f"Unexpected error processing {subject_id}: {str(e)}")
                failed.append(subject_id)

        # Generate batch reports if requested
        if args.generate_report and successful:
            main_logger.info("Generating batch report...")
            import report
            batch_report = report.generate_batch_report(
                project_dir, derivatives_dir, successful, args.report_name
            )
            
            if batch_report:
                main_logger.info(f"✓ Batch report generated: {batch_report}")
            else:
                main_logger.error("Failed to generate batch report")

        # Print summary following SW-detect pattern
        main_logger.info("=== PIPELINE EXECUTION SUMMARY ===")
        main_logger.info(f"Total subjects requested: {len(subjects_to_process)}")
        main_logger.info(f"Successfully processed: {len(successful)}")
        main_logger.info(f"Failed: {len(failed)}")
        
        if successful:
            main_logger.info(f"Successful subjects: {', '.join(successful)}")
        
        if failed:
            main_logger.error(f"Failed subjects: {', '.join(failed)}")
        
        main_logger.info("=" * 35)
        
        # Return appropriate exit code
        if failed:
            main_logger.error("Pipeline completed with errors")
            return 1
        else:
            main_logger.info("Pipeline completed successfully")
            return 0
        
    except KeyboardInterrupt:
        main_logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        main_logger.error(f"Unexpected pipeline error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 