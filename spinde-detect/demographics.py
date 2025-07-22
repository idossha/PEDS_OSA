#!/usr/bin/env python3
"""
Demographics Module
==================

Handles loading and parsing demographic information from demographics.csv file.
Includes age calculation logic where 6+ months rounds age up.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union

def load_demographics(project_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load demographics.csv from the project root directory.
    
    Parameters
    ----------
    project_dir : Path
        Path to the main project directory
    
    Returns
    -------
    Optional[pd.DataFrame]
        Demographics dataframe, or None if file not found/invalid
    """
    logger = logging.getLogger(__name__)
    
    demographics_file = project_dir / 'demographics.csv'
    
    if not demographics_file.exists():
        logger.warning(f"Demographics file not found: {demographics_file}")
        return None
    
    try:
        # Load the CSV file
        df = pd.read_csv(demographics_file)
        
        # Validate required columns
        required_columns = ['pptid', 'group', 'age_years', 'age_months', 'nrem_oai', 'nrem_ahi']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Demographics file missing required columns: {missing_columns}")
            return None
        
        logger.info(f"Successfully loaded demographics for {len(df)} subjects")
        return df
        
    except Exception as e:
        logger.error(f"Error loading demographics file: {str(e)}")
        return None

def calculate_rounded_age(age_years: Union[int, float], age_months: Union[int, float]) -> int:
    """
    Calculate rounded age where 6+ months rounds up to next year.
    
    Parameters
    ----------
    age_years : Union[int, float]
        Age in years
    age_months : Union[int, float]
        Additional months
    
    Returns
    -------
    int
        Rounded age in years
    
    Examples
    --------
    >>> calculate_rounded_age(6, 6)  # 6 years, 6 months -> 7
    7
    >>> calculate_rounded_age(6, 4)  # 6 years, 4 months -> 6
    6
    >>> calculate_rounded_age(6, 0)  # 6 years, 0 months -> 6
    6
    """
    try:
        years = int(age_years)
        months = int(age_months)
        
        # If 6 or more months, round up
        if months >= 6:
            return years + 1
        else:
            return years
    except (ValueError, TypeError):
        # If we can't parse the values, return the years component
        try:
            return int(age_years)
        except (ValueError, TypeError):
            return 0

def get_subject_demographics(demographics_df: pd.DataFrame, subject_id: str) -> Optional[Dict]:
    """
    Get demographic information for a specific subject.
    
    Parameters
    ----------
    demographics_df : pd.DataFrame
        Demographics dataframe
    subject_id : str
        Subject identifier (e.g., 'sub-001')
    
    Returns
    -------
    Optional[Dict]
        Dictionary with demographic information, or None if subject not found
    """
    logger = logging.getLogger(__name__)
    
    if demographics_df is None:
        return None
    
    # Extract the participant ID from subject_id (remove 'sub-' prefix if present)
    if subject_id.startswith('sub-'):
        pptid = subject_id[4:]  # Remove 'sub-' prefix
    else:
        pptid = subject_id
    
    # Try multiple formats to find the subject
    search_ids = [
        pptid,                          # Original extracted ID
        subject_id,                     # Full subject ID
        pptid.zfill(3),                # Zero-padded to 3 digits (001, 002, etc.)
        pptid.lstrip('0'),             # Remove leading zeros
        str(int(pptid)) if pptid.isdigit() else pptid  # Convert to int and back to remove leading zeros
    ]
    
    # Remove duplicates while preserving order
    search_ids = list(dict.fromkeys(search_ids))
    
    subject_row = pd.DataFrame()
    for search_id in search_ids:
        subject_row = demographics_df[demographics_df['pptid'].astype(str) == search_id]
        if not subject_row.empty:
            break
    
    if subject_row.empty:
        logger.warning(f"Subject {subject_id} not found in demographics (searched: {search_ids})")
        return None
    
    # Get the first matching row
    row = subject_row.iloc[0]
    
    # Calculate rounded age
    rounded_age = calculate_rounded_age(row['age_years'], row['age_months'])
    
    # Extract demographic information
    demographics = {
        'subject_id': subject_id,
        'pptid': str(row['pptid']),
        'group': str(row['group']),
        'age_years_raw': row['age_years'],
        'age_months_raw': row['age_months'],
        'age_rounded': rounded_age,
        'nrem_oai': row['nrem_oai'] if pd.notna(row['nrem_oai']) else 'N/A',
        'nrem_ahi': row['nrem_ahi'] if pd.notna(row['nrem_ahi']) else 'N/A'
    }
    
    # Add gender if available
    if 'gender' in row:
        demographics['gender'] = str(row['gender']) if pd.notna(row['gender']) else 'N/A'
    
    logger.debug(f"Retrieved demographics for {subject_id}: age {rounded_age}, group {demographics['group']}")
    
    return demographics

def format_demographics_for_report(demographics: Optional[Dict]) -> str:
    """
    Format demographic information for inclusion in reports.
    
    Parameters
    ----------
    demographics : Optional[Dict]
        Dictionary with demographic information
    
    Returns
    -------
    str
        Formatted markdown string for report inclusion
    """
    if demographics is None:
        return "## Subject Demographics\n*Demographics information not available*\n\n"
    
    # Format age display
    age_display = f"{demographics['age_rounded']} years"
    if demographics.get('age_years_raw') and demographics.get('age_months_raw'):
        age_display += f" (from {demographics['age_years_raw']} years, {demographics['age_months_raw']} months)"
    
    content = f"""## Subject Demographics
- **Age**: {age_display}
- **Group**: {demographics['group']}
- **NREM OAI**: {demographics['nrem_oai']}
- **NREM AHI**: {demographics['nrem_ahi']}

"""
    
    # Add gender if available
    if demographics.get('gender') and demographics['gender'] != 'N/A':
        content = content.replace('- **NREM AHI**:', f"- **Gender**: {demographics['gender']}\n- **NREM AHI**:")
    
    return content

def load_subject_demographics(project_dir: Path, subject_id: str) -> Optional[Dict]:
    """
    Convenience function to load demographics for a specific subject.
    
    Parameters
    ----------
    project_dir : Path
        Path to the main project directory
    subject_id : str
        Subject identifier
    
    Returns
    -------
    Optional[Dict]
        Dictionary with demographic information, or None if not found
    """
    demographics_df = load_demographics(project_dir)
    if demographics_df is None:
        return None
    
    return get_subject_demographics(demographics_df, subject_id) 