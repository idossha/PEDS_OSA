#!/usr/bin/env python3
"""
Parser Module
=============

Handles directory structure parsing, path management, and subject discovery.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict

def validate_project_directory(project_dir: str) -> Path:
    """
    Validate that project directory exists.
    
    Parameters
    ----------
    project_dir : str
        Path to the main project directory
    
    Returns
    -------
    Path
        Validated project directory path
    
    Raises
    ------
    ValueError
        If project directory does not exist
    """
    project_path = Path(project_dir).resolve()
    if not project_path.exists():
        raise ValueError(f"Project directory does not exist: {project_path}")
    return project_path

def find_subjects(project_dir: Path) -> List[str]:
    """
    Find all subjects with .set files in the project directory.
    
    Parameters
    ----------
    project_dir : Path
        Path to the main project directory
    
    Returns
    -------
    List[str]
        List of subject IDs (e.g., ['sub-001', 'sub-002'])
    """
    logger = logging.getLogger(__name__)
    subjects = []
    
    # Look for subject directories matching pattern sub-*
    for item in project_dir.iterdir():
        if item.is_dir() and item.name.startswith('sub-'):
            eeg_dir = item / 'eeg'
            if eeg_dir.exists():
                # Check for .set files
                set_files = list(eeg_dir.glob('*.set'))
                if set_files:
                    subjects.append(item.name)
                    logger.debug(f"Found subject {item.name} with {len(set_files)} .set file(s)")
    
    subjects.sort()
    logger.info(f"Found {len(subjects)} subjects with EEG data")
    return subjects

def get_subject_eeg_file(project_dir: Path, subject_id: str) -> Optional[Path]:
    """
    Get the .set file path for a specific subject.
    
    Parameters
    ----------
    project_dir : Path
        Path to the main project directory
    subject_id : str
        Subject identifier (e.g., 'sub-001')
    
    Returns
    -------
    Optional[Path]
        Path to the .set file, or None if not found
    """
    logger = logging.getLogger(__name__)
    eeg_dir = project_dir / subject_id / 'eeg'
    
    if not eeg_dir.exists():
        logger.warning(f"EEG directory not found for {subject_id}: {eeg_dir}")
        return None
    
    set_files = list(eeg_dir.glob('*.set'))
    
    if not set_files:
        logger.warning(f"No .set files found for {subject_id}")
        return None
    
    if len(set_files) > 1:
        logger.warning(f"Multiple .set files found for {subject_id}, using first: {set_files[0]}")
    
    return set_files[0]

def create_output_directories(project_dir: Path, output_dir: str = "derivatives") -> Path:
    """
    Create and return derivatives directory.
    
    Parameters
    ----------
    project_dir : Path
        Path to the main project directory
    output_dir : str
        Name of the output directory within project (default: "derivatives")
    
    Returns
    -------
    Path
        Path to the derivatives directory
    """
    derivatives_dir = project_dir / output_dir
    derivatives_dir.mkdir(exist_ok=True)
    return derivatives_dir

def get_subject_output_dir(derivatives_dir: Path, subject_id: str) -> Path:
    """
    Get the output directory for a specific subject.
    
    Parameters
    ----------
    derivatives_dir : Path
        Path to the derivatives directory
    subject_id : str
        Subject identifier (e.g., 'sub-001')
    
    Returns
    -------
    Path
        Path to the subject's output directory
    """
    return derivatives_dir / subject_id / 'spindles'

def create_subject_output_structure(derivatives_dir: Path, subject_id: str) -> Dict[str, Path]:
    """
    Create the complete output directory structure for a subject.
    
    Parameters
    ----------
    derivatives_dir : Path
        Path to the derivatives directory
    subject_id : str
        Subject identifier (e.g., 'sub-001')
    
    Returns
    -------
    Dict[str, Path]
        Dictionary with paths to different output subdirectories
    """
    logger = logging.getLogger(__name__)
    base_dir = get_subject_output_dir(derivatives_dir, subject_id)
    
    # Create subdirectories
    subdirs = {
        'base': base_dir,
        'results': base_dir / 'results',
        'visualizations': base_dir / 'visualizations',
        'logs': base_dir / 'logs',
        'data': base_dir / 'data',
        'report': base_dir / 'report'
    }
    
    # Create all directories
    for dir_path in subdirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Created output structure for {subject_id}")
    return subdirs

def subject_results_exist(derivatives_dir: Path, subject_id: str) -> bool:
    """
    Check if results already exist for a subject.
    
    Parameters
    ----------
    derivatives_dir : Path
        Path to the derivatives directory
    subject_id : str
        Subject identifier (e.g., 'sub-001')
    
    Returns
    -------
    bool
        True if results exist, False otherwise
    """
    output_dir = get_subject_output_dir(derivatives_dir, subject_id)
    
    # Check for key result files
    key_files = [
        output_dir / 'results' / 'spindle_summary.csv',
        output_dir / 'visualizations' / 'comprehensive_analysis.png'
    ]
    
    return all(f.exists() for f in key_files)

def validate_subject(project_dir: Path, subject_id: str) -> bool:
    """
    Validate that a subject exists and has the required data.
    
    Parameters
    ----------
    project_dir : Path
        Path to the main project directory
    subject_id : str
        Subject identifier (e.g., 'sub-001')
    
    Returns
    -------
    bool
        True if subject is valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Check if subject directory exists
    subject_dir = project_dir / subject_id
    if not subject_dir.exists():
        logger.error(f"Subject directory not found: {subject_dir}")
        return False
    
    # Check if EEG file exists
    eeg_file = get_subject_eeg_file(project_dir, subject_id)
    if eeg_file is None:
        logger.error(f"No EEG .set file found for {subject_id}")
        return False
    
    # Check file size (basic validation)
    if eeg_file.stat().st_size == 0:
        logger.error(f"EEG file is empty for {subject_id}: {eeg_file}")
        return False
    
    logger.debug(f"Subject {subject_id} validated successfully")
    return True

def get_project_summary(project_dir: Path, derivatives_dir: Path) -> Dict:
    """
    Get a summary of the project structure.
    
    Parameters
    ----------
    project_dir : Path
        Path to the main project directory
    derivatives_dir : Path
        Path to the derivatives directory
    
    Returns
    -------
    Dict
        Summary information about the project
    """
    subjects = find_subjects(project_dir)
    processed_subjects = []
    
    for subject in subjects:
        if subject_results_exist(derivatives_dir, subject):
            processed_subjects.append(subject)
    
    return {
        'project_dir': str(project_dir),
        'derivatives_dir': str(derivatives_dir),
        'total_subjects': len(subjects),
        'available_subjects': subjects,
        'processed_subjects': processed_subjects,
        'unprocessed_subjects': [s for s in subjects if s not in processed_subjects]
    } 