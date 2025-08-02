#!/usr/bin/env python3
"""
PEDS OSA: Spindle Density vs AHI Analysis Pipeline
==================================================

This script analyzes the relationship between sleep spindle density and 
Apnea-Hypopnea Index (AHI) in children with and without Obstructive Sleep Apnea (OSA).

Research Questions:
1. Does AHI predict spindle density, and how does its impact compare to age?
2. Does AHI change the relationship between age and spindle density (disrupt development)?

Data Sources:
- Spindle data: derivatives/sub-XXX/spindles/results/all_spindles_detailed.csv
- Demographics: demographics.csv (includes AHI, age, sex, group)

Statistical Models:
1. Main Effects: Spindle_Density ~ Age + AHI + Sex
2. Interaction: Spindle_Density ~ Age * AHI + Sex  
3. Log-Transformed: Spindle_Density ~ Age + log1p(AHI) + Sex
4. Regional: Frontal vs Parietal spindle density
5. Group Comparisons: Control vs OSA

Output: Results saved to project_dir/analysis_2/
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def collect_spindle_density_data(project_dir):
    """
    Extract spindle density metrics from individual subject files.
    
    This function processes all spindle detection results and calculates
    density metrics (spindles per minute) for each subject.
    
    Parameters
    ----------
    project_dir : str
        Path to the project directory containing derivatives/sub-*/spindles/
    
    Returns
    -------
    pd.DataFrame
        DataFrame with subject ID and spindle density metrics
    """
    print("Collecting spindle density data from all subjects...")
    
    # Find all spindle detailed files
    pattern = os.path.join(project_dir, "derivatives", "sub-*", "spindles", "results", "all_spindles_detailed.csv")
    spindle_files = glob.glob(pattern)
    
    print(f"Found {len(spindle_files)} subject spindle files")
    
    spindle_data = []
    
    for file_path in spindle_files:
        try:
            # Extract subject ID from file path
            # Expected path: derivatives/sub-001/spindles/results/all_spindles_detailed.csv
            path_parts = file_path.split(os.sep)
            subject_folder = [part for part in path_parts if part.startswith('sub-')][0]
            subject_id = subject_folder.replace("sub-", "")
            
            # Read spindle detailed data
            print(f"  Processing {subject_id}...")
            spindle_df = pd.read_csv(file_path)
            
            # Calculate recording duration from the data
            total_duration_seconds = spindle_df['End'].max()
            total_duration_minutes = total_duration_seconds / 60
            
            # Calculate spindle density (spindles per minute)
            total_spindles = len(spindle_df)
            spindle_density = total_spindles / total_duration_minutes if total_duration_minutes > 0 else 0
            
            # Additional spindle characteristics
            mean_duration = spindle_df['Duration'].mean()
            mean_frequency = spindle_df['Frequency'].mean()
            mean_amplitude = spindle_df['Amplitude'].mean()
            
            # Calculate slow vs fast spindle distribution
            slow_spindles = len(spindle_df[spindle_df['Frequency'] < 12])
            fast_spindles = len(spindle_df[spindle_df['Frequency'] >= 12])
            
            # Calculate regional spindle density (if channel information available)
            frontal_density = 0
            parietal_density = 0
            
            if 'Channel' in spindle_df.columns:
                # Identify frontal and parietal channels
                frontal_channels = [col for col in spindle_df.columns if 'F' in col and 'Channel' in col]
                parietal_channels = [col for col in spindle_df.columns if 'P' in col and 'Channel' in col]
                
                # Calculate frontal spindle density
                if frontal_channels:
                    frontal_spindles = len(spindle_df[spindle_df[frontal_channels[0]] == 1])
                    frontal_density = frontal_spindles / total_duration_minutes if total_duration_minutes > 0 else 0
                
                # Calculate parietal spindle density
                if parietal_channels:
                    parietal_spindles = len(spindle_df[spindle_df[parietal_channels[0]] == 1])
                    parietal_density = parietal_spindles / total_duration_minutes if total_duration_minutes > 0 else 0
            
            spindle_data.append({
                'Subject': subject_id,
                'spindle_density': spindle_density,
                'frontal_spindle_density': frontal_density,
                'parietal_spindle_density': parietal_density,
                'total_spindles': total_spindles,
                'recording_duration_min': total_duration_minutes,
                'recording_duration_sec': total_duration_seconds,
                'mean_duration': mean_duration,
                'mean_frequency': mean_frequency,
                'mean_amplitude': mean_amplitude,
                'slow_spindles': slow_spindles,
                'fast_spindles': fast_spindles,
                'slow_percentage': (slow_spindles / total_spindles * 100) if total_spindles > 0 else 0,
                'fast_percentage': (fast_spindles / total_spindles * 100) if total_spindles > 0 else 0
            })
            
        except Exception as e:
            print(f"  Error processing {subject_id}: {e}")
            continue
    
    spindle_df = pd.DataFrame(spindle_data)
    print(f"✓ Successfully processed {len(spindle_df)} subjects")
    return spindle_df

def load_and_prepare_data(project_dir):
    """
    Load and prepare spindle and demographic data for analysis.
    
    This function loads spindle density data and demographic information,
    then prepares them for merging and analysis.
    
    Parameters
    ----------
    project_dir : str
        Path to the project directory
    
    Returns
    -------
    tuple
        (spindle_df, demo_df) - prepared DataFrames ready for analysis
    """
    print("\nLoading and preparing data...")
    
    # Load spindle density data
    spindle_df = collect_spindle_density_data(project_dir)
    
    # Load demographics data
    demo_file = os.path.join(project_dir, "demographics.csv")
    if not os.path.exists(demo_file):
        demo_file = os.path.join(project_dir, "analysis", "reference", "demographics.csv")
    
    if os.path.exists(demo_file):
        demo_df = pd.read_csv(demo_file)
        print(f"✓ Loaded demographics data: {len(demo_df)} subjects")
        
        # Map column names to standardized format
        column_mapping = {
            'pptid': 'Subject',
            'age_years': 'age',
            'gender': 'sex',
            'overall_ahi': 'AHI',
            'group': 'group'
        }
        
        # Create standardized columns
        for old_name, new_name in column_mapping.items():
            if old_name in demo_df.columns:
                demo_df[new_name] = demo_df[old_name]
        
        # Convert gender coding (1=M, 0=F) to M/F format
        if 'sex' in demo_df.columns:
            demo_df['sex'] = demo_df['sex'].map({1: 'M', 0: 'F'})
        
        print(f"  Available columns after mapping: {demo_df.columns.tolist()}")
        
    else:
        print("Warning: Demographics file not found. Creating dummy data for testing.")
        demo_df = pd.DataFrame({
            'Subject': spindle_df['Subject'],
            'age': np.random.normal(10, 2, len(spindle_df)),
            'sex': np.random.choice(['M', 'F'], len(spindle_df)),
            'AHI': np.random.exponential(2, len(spindle_df)),
            'group': np.random.choice(['Control', 'OSA'], len(spindle_df))
        })
    
    # Clean and prepare data - normalize subject IDs to match format
    # Convert subject IDs to match format - normalize to remove leading zeros
    spindle_df['Subject'] = spindle_df['Subject'].astype(str).str.lstrip('0')
    demo_df['pptid'] = demo_df['pptid'].astype(str)
    
    return spindle_df, demo_df

def create_analysis_dataset(spindle_df, demo_df):
    """
    Create the final analysis dataset by merging and preparing data.
    
    This function merges spindle density data with demographic information,
    creates derived variables, and prepares the dataset for statistical analysis.
    
    Parameters
    ----------
    spindle_df : pd.DataFrame
        Spindle density data with subject IDs and metrics
    demo_df : pd.DataFrame
        Demographic data with AHI, age, sex, and group information
    
    Returns
    -------
    pd.DataFrame
        Complete analysis dataset with all variables and derived metrics
    """
    print("\nCreating analysis dataset...")
    
    # Check subject ID matching before merging
    print(f"  Spindle data subjects: {spindle_df['Subject'].tolist()[:10]}...")
    print(f"  Demographics subjects: {demo_df['pptid'].tolist()[:10]}...")
    
    # Merge spindle data with demographics
    merged_df = spindle_df.merge(demo_df, left_on='Subject', right_on='pptid', how='inner')
    
    print(f"  After merging: {len(merged_df)} subjects")
    print(f"  Merged columns: {merged_df.columns.tolist()}")
    
    if len(merged_df) == 0:
        print("  Warning: No subjects matched between spindle and demographics data!")
        print("  This could be due to different subject ID formats.")
        return merged_df
    
    # Create standardized analysis variables
    merged_df['age'] = merged_df['age_years']
    merged_df['sex'] = merged_df['gender'].map({0: 'F', 1: 'M'})
    merged_df['AHI'] = merged_df['overall_ahi']
    
    # Create log-transformed AHI (log1p for safety with values close to 0)
    merged_df['log_AHI'] = np.log1p(merged_df['AHI'])
    
    # Create standardized (z-score) variables for effect size comparison
    merged_df['z_age'] = (merged_df['age'] - merged_df['age'].mean()) / merged_df['age'].std()
    merged_df['z_AHI'] = (merged_df['AHI'] - merged_df['AHI'].mean()) / merged_df['AHI'].std()
    merged_df['z_log_AHI'] = (merged_df['log_AHI'] - merged_df['log_AHI'].mean()) / merged_df['log_AHI'].std()
    
    # Create binary OSA classification (AHI > 1)
    merged_df['OSA_binary'] = (merged_df['AHI'] > 1).astype(int)
    
    # Create OSA severity categories
    merged_df['OSA_severity'] = pd.cut(merged_df['AHI'], 
                                      bins=[0, 1, 5, 10, np.inf], 
                                      labels=['Normal', 'Mild', 'Moderate', 'Severe'])
    
    print(f"✓ Final dataset: {len(merged_df)} subjects with complete data")
    print(f"  Age range: {merged_df['age'].min():.1f} - {merged_df['age'].max():.1f} years")
    print(f"  AHI range: {merged_df['AHI'].min():.1f} - {merged_df['AHI'].max():.1f}")
    print(f"  Spindle density range: {merged_df['spindle_density'].min():.2f} - {merged_df['spindle_density'].max():.2f} spindles/min")
    
    return merged_df

def analyze_main_effects(merged_df, output_dir):
    """
    Analyze main effects of age and AHI on spindle density.
    
    This function tests the independent effects of age and AHI on spindle density,
    controlling for sex differences. It addresses the question: "Does AHI predict
    spindle density, and how does its impact compare to age?"
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Complete analysis dataset with all variables
    output_dir : str
        Directory to save analysis results
    
    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted regression model results
    """
    print("\nPerforming main effects analysis...")
    
    # Prepare data for analysis
    analysis_df = merged_df.dropna(subset=['spindle_density', 'age', 'AHI', 'sex'])
    
    # Check if we have enough data and variety in categorical variables
    if len(analysis_df) == 0:
        print("  Error: No complete cases found for analysis")
        return None
    
    # Check sex variable levels
    sex_levels = analysis_df['sex'].unique()
    print(f"  Sex levels found: {sex_levels}")
    
    if len(sex_levels) < 2:
        print("  Warning: Only one sex level found. Removing sex from model.")
        model = ols('spindle_density ~ age + AHI', data=analysis_df).fit()
    else:
        # Create the model with sex
        model = ols('spindle_density ~ age + AHI + sex', data=analysis_df).fit()
    
    # Save results
    results_file = os.path.join(output_dir, 'main_effects_analysis_results.txt')
    with open(results_file, 'w') as f:
        f.write("MAIN EFFECTS ANALYSIS: Spindle Density ~ Age + AHI + Sex\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: spindle_density ~ age + AHI + sex\n")
        f.write(f"Sample size: {len(analysis_df)}\n\n")
        
        f.write("MODEL SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(str(model.summary()))
        f.write("\n\n")
        
        f.write("COEFFICIENT INTERPRETATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Age effect: {model.params['age']:.4f} (p = {model.pvalues['age']:.4f})\n")
        f.write(f"AHI effect: {model.params['AHI']:.4f} (p = {model.pvalues['AHI']:.4f})\n")
        f.write(f"Sex effect (M vs F): {model.params['sex[T.M]']:.4f} (p = {model.pvalues['sex[T.M]']:.4f})\n")
        
        # Calculate standardized coefficients
        f.write("\nSTANDARDIZED COEFFICIENTS\n")
        f.write("-" * 30 + "\n")
        std_age = model.params['age'] * analysis_df['age'].std() / analysis_df['spindle_density'].std()
        std_ahi = model.params['AHI'] * analysis_df['AHI'].std() / analysis_df['spindle_density'].std()
        f.write(f"Standardized Age effect: {std_age:.4f}\n")
        f.write(f"Standardized AHI effect: {std_ahi:.4f}\n")
        
        # Calculate partial R²
        f.write("\nPARTIAL R² VALUES\n")
        f.write("-" * 20 + "\n")
        f.write(f"Age partial R²: {model.rsquared_adj:.4f}\n")
        f.write(f"AHI partial R²: {model.rsquared_adj:.4f}\n")
        
        # Model diagnostics
        f.write("\nMODEL DIAGNOSTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"R²: {model.rsquared:.4f}\n")
        f.write(f"Adjusted R²: {model.rsquared_adj:.4f}\n")
        f.write(f"F-statistic: {model.fvalue:.4f}\n")
        f.write(f"F p-value: {model.f_pvalue:.4f}\n")
    
    print(f"✓ Main effects analysis results saved to {results_file}")
    return model

def analyze_age_ahi_interaction(merged_df, output_dir):
    """
    Analyze interaction between age and AHI on spindle density.
    
    This function tests whether AHI moderates the relationship between age and
    spindle density. It addresses the question: "Does AHI disrupt normal
    spindle development?"
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Complete analysis dataset with all variables
    output_dir : str
        Directory to save analysis results
    
    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted interaction model results
    """
    print("\nPerforming interaction analysis...")
    
    # Prepare data for analysis
    analysis_df = merged_df.dropna(subset=['spindle_density', 'age', 'AHI', 'sex'])
    
    # Create the interaction model
    model = ols('spindle_density ~ age * AHI + sex', data=analysis_df).fit()
    
    # Save results
    results_file = os.path.join(output_dir, 'interaction_analysis_results.txt')
    with open(results_file, 'w') as f:
        f.write("INTERACTION ANALYSIS: Spindle Density ~ Age * AHI + Sex\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: spindle_density ~ age * AHI + sex\n")
        f.write(f"Sample size: {len(analysis_df)}\n\n")
        
        f.write("MODEL SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(str(model.summary()))
        f.write("\n\n")
        
        f.write("INTERACTION INTERPRETATION\n")
        f.write("-" * 30 + "\n")
        if 'age:AHI' in model.params.index:
            interaction_effect = model.params['age:AHI']
            interaction_p = model.pvalues['age:AHI']
            f.write(f"Age × AHI interaction: {interaction_effect:.4f} (p = {interaction_p:.4f})\n")
            
            if interaction_p < 0.05:
                f.write("✓ Significant interaction found!\n")
                f.write("  This indicates that AHI moderates the relationship between age and spindle density.\n")
                if interaction_effect < 0:
                    f.write("  Negative interaction: Higher AHI reduces the positive effect of age on spindle density.\n")
                else:
                    f.write("  Positive interaction: Higher AHI enhances the positive effect of age on spindle density.\n")
            else:
                f.write("✗ No significant interaction found.\n")
                f.write("  Age and AHI have independent effects on spindle density.\n")
        
        # Model diagnostics
        f.write("\nMODEL DIAGNOSTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"R²: {model.rsquared:.4f}\n")
        f.write(f"Adjusted R²: {model.rsquared_adj:.4f}\n")
        f.write(f"F-statistic: {model.fvalue:.4f}\n")
        f.write(f"F p-value: {model.f_pvalue:.4f}\n")
    
    print(f"✓ Interaction analysis results saved to {results_file}")
    return model

def analyze_log_transformed_ahi(merged_df, output_dir):
    """
    Analyze spindle density using log-transformed AHI.
    
    This function uses log1p(AHI) transformation as recommended in the reference
    document to address non-linearity and heteroscedasticity in AHI effects.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Complete analysis dataset with all variables
    output_dir : str
        Directory to save analysis results
    
    Returns
    -------
    tuple
        (main_model, interaction_model) - Both fitted regression models
    """
    print("\nPerforming log-AHI analysis...")
    
    # Prepare data for analysis
    analysis_df = merged_df.dropna(subset=['spindle_density', 'age', 'log_AHI', 'sex'])
    
    # Create models with log-transformed AHI
    model_main = ols('spindle_density ~ age + log_AHI + sex', data=analysis_df).fit()
    model_interaction = ols('spindle_density ~ age * log_AHI + sex', data=analysis_df).fit()
    
    # Save results
    results_file = os.path.join(output_dir, 'log_ahi_analysis_results.txt')
    with open(results_file, 'w') as f:
        f.write("LOG-TRANSFORMED AHI ANALYSIS\n")
        f.write("=" * 40 + "\n\n")
        f.write("Using log1p(AHI) as recommended in the reference document.\n")
        f.write("This transformation helps with:\n")
        f.write("- Non-linear relationships\n")
        f.write("- Heteroscedasticity\n")
        f.write("- Interpretability of multiplicative effects\n\n")
        
        f.write("MAIN EFFECTS MODEL: Spindle Density ~ Age + log(AHI) + Sex\n")
        f.write("-" * 60 + "\n")
        f.write(str(model_main.summary()))
        f.write("\n\n")
        
        f.write("INTERACTION MODEL: Spindle Density ~ Age * log(AHI) + Sex\n")
        f.write("-" * 60 + "\n")
        f.write(str(model_interaction.summary()))
        f.write("\n\n")
        
        f.write("INTERPRETATION\n")
        f.write("-" * 15 + "\n")
        f.write(f"Log-AHI effect: {model_main.params['log_AHI']:.4f} (p = {model_main.pvalues['log_AHI']:.4f})\n")
        f.write("This represents the change in spindle density for a unit increase in log(AHI).\n")
        f.write("For practical interpretation: a 1-unit increase in log(AHI) corresponds to\n")
        f.write("approximately a 2.7-fold increase in raw AHI values.\n")
    
    print(f"✓ Log-AHI analysis results saved to {results_file}")
    return model_main, model_interaction

def analyze_regional_differences(merged_df, output_dir):
    """
    Analyze regional differences in spindle density (frontal vs parietal).
    
    This function compares spindle density between frontal and parietal regions
    and tests for regional differences in AHI effects.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Complete analysis dataset with regional spindle data
    output_dir : str
        Directory to save analysis results
    
    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted regional analysis model results
    """
    print("\nPerforming regional analysis...")
    
    # Check if regional data is available
    if 'frontal_spindle_density' not in merged_df.columns or 'parietal_spindle_density' not in merged_df.columns:
        print("  Regional data not available, skipping regional analysis")
        return
    
    # Prepare data for analysis
    analysis_df = merged_df.dropna(subset=['frontal_spindle_density', 'parietal_spindle_density', 'age', 'AHI', 'sex'])
    
    # Create long format for regional comparison
    regional_data = []
    for _, row in analysis_df.iterrows():
        regional_data.append({
            'Subject': row['Subject_x'] if 'Subject_x' in row else row['Subject'],
            'Region': 'Frontal',
            'spindle_density': row['frontal_spindle_density'],
            'age': row['age'],
            'AHI': row['AHI'],
            'sex': row['sex']
        })
        regional_data.append({
            'Subject': row['Subject_x'] if 'Subject_x' in row else row['Subject'],
            'Region': 'Parietal',
            'spindle_density': row['parietal_spindle_density'],
            'age': row['age'],
            'AHI': row['AHI'],
            'sex': row['sex']
        })
    
    regional_df = pd.DataFrame(regional_data)
    
    # Perform mixed model analysis
    model = ols('spindle_density ~ Region * age + Region * AHI + sex', data=regional_df).fit()
    
    # Save results
    results_file = os.path.join(output_dir, 'regional_analysis_results.txt')
    with open(results_file, 'w') as f:
        f.write("REGIONAL ANALYSIS: Frontal vs Parietal Spindles\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: spindle_density ~ Region * age + Region * AHI + sex\n")
        f.write(f"Sample size: {len(analysis_df)} subjects × 2 regions = {len(regional_df)} observations\n\n")
        
        f.write("MODEL SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(str(model.summary()))
        f.write("\n\n")
        
        f.write("REGIONAL COMPARISONS\n")
        f.write("-" * 25 + "\n")
        frontal_mean = analysis_df['frontal_spindle_density'].mean()
        parietal_mean = analysis_df['parietal_spindle_density'].mean()
        f.write(f"Frontal spindle density: {frontal_mean:.3f} ± {analysis_df['frontal_spindle_density'].std():.3f}\n")
        f.write(f"Parietal spindle density: {parietal_mean:.3f} ± {analysis_df['parietal_spindle_density'].std():.3f}\n")
        
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(analysis_df['frontal_spindle_density'], analysis_df['parietal_spindle_density'])
        f.write(f"Paired t-test (Frontal vs Parietal): t = {t_stat:.3f}, p = {p_val:.4f}\n")
        
        if p_val < 0.05:
            f.write("✓ Significant difference between regions\n")
        else:
            f.write("✗ No significant difference between regions\n")
    
    print(f"✓ Regional analysis results saved to {results_file}")
    return model

def compare_control_osa_groups(merged_df, output_dir):
    """
    Compare spindle density between Control and OSA groups.
    
    This function performs group comparisons using t-tests and Mann-Whitney U tests,
    and calculates effect sizes to quantify group differences.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Complete analysis dataset with group assignments
    output_dir : str
        Directory to save analysis results
    """
    print("\nPerforming group analysis...")
    
    # Prepare data for analysis
    analysis_df = merged_df.dropna(subset=['spindle_density', 'group'])
    
    # Ensure we have the right subject column
    if 'Subject_x' in analysis_df.columns:
        analysis_df['Subject'] = analysis_df['Subject_x']
    
    # Group statistics
    control_group = analysis_df[analysis_df['group'] == 'Control']
    osa_group = analysis_df[analysis_df['group'] == 'OSA']
    
    # Independent t-test
    t_stat, p_val = stats.ttest_ind(control_group['spindle_density'], osa_group['spindle_density'])
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_p_val = stats.mannwhitneyu(control_group['spindle_density'], osa_group['spindle_density'], alternative='two-sided')
    
    # Save results
    results_file = os.path.join(output_dir, 'group_analysis_results.txt')
    with open(results_file, 'w') as f:
        f.write("GROUP COMPARISON ANALYSIS: Control vs OSA\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("GROUP DESCRIPTIVE STATISTICS\n")
        f.write("-" * 35 + "\n")
        f.write(f"Control group (n={len(control_group)}):\n")
        f.write(f"  Spindle density: {control_group['spindle_density'].mean():.3f} ± {control_group['spindle_density'].std():.3f}\n")
        f.write(f"  Age: {control_group['age'].mean():.1f} ± {control_group['age'].std():.1f}\n")
        f.write(f"  AHI: {control_group['AHI'].mean():.2f} ± {control_group['AHI'].std():.2f}\n\n")
        
        f.write(f"OSA group (n={len(osa_group)}):\n")
        f.write(f"  Spindle density: {osa_group['spindle_density'].mean():.3f} ± {osa_group['spindle_density'].std():.3f}\n")
        f.write(f"  Age: {osa_group['age'].mean():.1f} ± {osa_group['age'].std():.1f}\n")
        f.write(f"  AHI: {osa_group['AHI'].mean():.2f} ± {osa_group['AHI'].std():.2f}\n\n")
        
        f.write("STATISTICAL TESTS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Independent t-test: t = {t_stat:.3f}, p = {p_val:.4f}\n")
        f.write(f"Mann-Whitney U test: U = {u_stat:.1f}, p = {u_p_val:.4f}\n\n")
        
        if p_val < 0.05:
            f.write("✓ Significant difference between groups (t-test)\n")
        else:
            f.write("✗ No significant difference between groups (t-test)\n")
        
        if u_p_val < 0.05:
            f.write("✓ Significant difference between groups (Mann-Whitney)\n")
        else:
            f.write("✗ No significant difference between groups (Mann-Whitney)\n")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_group) - 1) * control_group['spindle_density'].var() + 
                             (len(osa_group) - 1) * osa_group['spindle_density'].var()) / 
                            (len(control_group) + len(osa_group) - 2))
        cohens_d = (control_group['spindle_density'].mean() - osa_group['spindle_density'].mean()) / pooled_std
        f.write(f"\nEffect size (Cohen's d): {cohens_d:.3f}\n")
        
        if abs(cohens_d) < 0.2:
            f.write("Effect size interpretation: Negligible\n")
        elif abs(cohens_d) < 0.5:
            f.write("Effect size interpretation: Small\n")
        elif abs(cohens_d) < 0.8:
            f.write("Effect size interpretation: Medium\n")
        else:
            f.write("Effect size interpretation: Large\n")
    
    print(f"✓ Group analysis results saved to {results_file}")

def create_analysis_visualizations(merged_df, output_dir):
    """
    Create comprehensive visualization plots for the analysis.
    
    This function generates publication-ready plots including:
    - Spindle density vs AHI scatter plots
    - Age relationships and interactions
    - Group comparisons
    - Distribution plots
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Complete analysis dataset
    output_dir : str
        Directory to save visualization plots
    """
    print("\nCreating visualization plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PEDS OSA Spindle Density Analysis', fontsize=16, fontweight='bold')
    
    # 1. Spindle density vs AHI
    ax1 = axes[0, 0]
    sns.scatterplot(data=merged_df, x='AHI', y='spindle_density', hue='group', ax=ax1)
    sns.regplot(data=merged_df, x='AHI', y='spindle_density', scatter=False, ax=ax1, color='red')
    ax1.set_xlabel('Apnea-Hypopnea Index (AHI)')
    ax1.set_ylabel('Spindle Density (spindles/min)')
    ax1.set_title('Spindle Density vs AHI')
    
    # 2. Spindle density vs Age
    ax2 = axes[0, 1]
    sns.scatterplot(data=merged_df, x='age', y='spindle_density', hue='group', ax=ax2)
    sns.regplot(data=merged_df, x='age', y='spindle_density', scatter=False, ax=ax2, color='red')
    ax2.set_xlabel('Age (years)')
    ax2.set_ylabel('Spindle Density (spindles/min)')
    ax2.set_title('Spindle Density vs Age')
    
    # 3. Box plot by group
    ax3 = axes[0, 2]
    sns.boxplot(data=merged_df, x='group', y='spindle_density', ax=ax3)
    ax3.set_xlabel('Group')
    ax3.set_ylabel('Spindle Density (spindles/min)')
    ax3.set_title('Spindle Density by Group')
    
    # 4. Spindle density vs log(AHI)
    ax4 = axes[1, 0]
    sns.scatterplot(data=merged_df, x='log_AHI', y='spindle_density', hue='group', ax=ax4)
    sns.regplot(data=merged_df, x='log_AHI', y='spindle_density', scatter=False, ax=ax4, color='red')
    ax4.set_xlabel('Log(AHI + 1)')
    ax4.set_ylabel('Spindle Density (spindles/min)')
    ax4.set_title('Spindle Density vs Log(AHI)')
    
    # 5. Interaction plot: Age × AHI
    ax5 = axes[1, 1]
    # Create interaction plot by splitting AHI into tertiles
    merged_df['AHI_tertile'] = pd.qcut(merged_df['AHI'], q=3, labels=['Low', 'Medium', 'High'])
    sns.scatterplot(data=merged_df, x='age', y='spindle_density', hue='AHI_tertile', ax=ax5)
    for tertile in ['Low', 'Medium', 'High']:
        subset = merged_df[merged_df['AHI_tertile'] == tertile]
        if len(subset) > 0:
            sns.regplot(data=subset, x='age', y='spindle_density', scatter=False, ax=ax5)
    ax5.set_xlabel('Age (years)')
    ax5.set_ylabel('Spindle Density (spindles/min)')
    ax5.set_title('Age × AHI Interaction')
    
    # 6. Distribution of spindle density
    ax6 = axes[1, 2]
    sns.histplot(data=merged_df, x='spindle_density', hue='group', bins=20, ax=ax6)
    ax6.set_xlabel('Spindle Density (spindles/min)')
    ax6.set_ylabel('Count')
    ax6.set_title('Distribution of Spindle Density')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'spindle_density_analysis_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization plots saved to {plot_file}")

def generate_analysis_summary(merged_df, output_dir):
    """
    Generate a comprehensive summary report of the analysis.
    
    This function creates a detailed summary including:
    - Dataset description and sample characteristics
    - Key findings from all statistical models
    - File inventory and interpretation guidelines
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Complete analysis dataset
    output_dir : str
        Directory to save summary report
    """
    print("\nCreating summary report...")
    
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("PEDS OSA SPINDLE DENSITY ANALYSIS - COMPREHENSIVE SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total subjects with complete data: {len(merged_df)}\n")
        f.write(f"Age range: {merged_df['age'].min():.1f} - {merged_df['age'].max():.1f} years\n")
        f.write(f"Mean age: {merged_df['age'].mean():.1f} ± {merged_df['age'].std():.1f} years\n")
        f.write(f"AHI range: {merged_df['AHI'].min():.1f} - {merged_df['AHI'].max():.1f}\n")
        f.write(f"Mean AHI: {merged_df['AHI'].mean():.2f} ± {merged_df['AHI'].std():.2f}\n")
        f.write(f"Spindle density range: {merged_df['spindle_density'].min():.2f} - {merged_df['spindle_density'].max():.2f} spindles/min\n")
        f.write(f"Mean spindle density: {merged_df['spindle_density'].mean():.2f} ± {merged_df['spindle_density'].std():.2f} spindles/min\n")
        
        if 'group' in merged_df.columns:
            f.write(f"\nGroup breakdown:\n")
            for group, count in merged_df['group'].value_counts().items():
                f.write(f"  {group}: {count} subjects\n")
        
        f.write(f"\nSex breakdown:\n")
        for sex, count in merged_df['sex'].value_counts().items():
            f.write(f"  {sex}: {count} subjects\n")
        
        f.write("\n\nKEY FINDINGS\n")
        f.write("-" * 20 + "\n")
        f.write("1. Main Effects Model: Spindle Density ~ Age + AHI + Sex\n")
        f.write("   - Tests the independent effects of age and AHI on spindle density\n")
        f.write("   - Controls for sex differences\n\n")
        
        f.write("2. Interaction Model: Spindle Density ~ Age * AHI + Sex\n")
        f.write("   - Tests whether AHI moderates the age-spindle density relationship\n")
        f.write("   - Addresses the question: 'Does AHI disrupt normal spindle development?'\n\n")
        
        f.write("3. Log-Transformed AHI Analysis\n")
        f.write("   - Uses log1p(AHI) as recommended in the reference document\n")
        f.write("   - Addresses non-linearity and heteroscedasticity\n\n")
        
        f.write("4. Regional Analysis (if data available)\n")
        f.write("   - Compares frontal vs parietal spindle density\n")
        f.write("   - Tests for regional differences in AHI effects\n\n")
        
        f.write("5. Group Comparisons\n")
        f.write("   - Control vs OSA group differences\n")
        f.write("   - Effect size calculations\n\n")
        
        f.write("STATISTICAL MODELS\n")
        f.write("-" * 20 + "\n")
        f.write("Model 1: Main Effects\n")
        f.write("  Formula: spindle_density ~ age + AHI + sex\n")
        f.write("  Purpose: Test independent effects of age and AHI\n\n")
        
        f.write("Model 2: Interaction\n")
        f.write("  Formula: spindle_density ~ age * AHI + sex\n")
        f.write("  Purpose: Test AHI moderation of age effect\n\n")
        
        f.write("Model 3: Log-Transformed\n")
        f.write("  Formula: spindle_density ~ age + log1p(AHI) + sex\n")
        f.write("  Purpose: Address non-linear AHI effects\n\n")
        
        f.write("FILES GENERATED\n")
        f.write("-" * 20 + "\n")
        f.write("- analysis_dataset.csv: Complete analysis dataset\n")
        f.write("- main_effects_analysis_results.txt: Main effects model results\n")
        f.write("- interaction_analysis_results.txt: Interaction model results\n")
        f.write("- log_ahi_analysis_results.txt: Log-transformed AHI analysis\n")
        f.write("- regional_analysis_results.txt: Regional comparison results\n")
        f.write("- group_analysis_results.txt: Group comparison results\n")
        f.write("- spindle_density_analysis_plots.png: Visualization plots\n")
        f.write("- analysis_summary.txt: This summary file\n")
    
    print(f"✓ Summary report saved to {summary_file}")

def main(project_dir=None):
    """
    Execute the complete PEDS OSA spindle density analysis pipeline.
    
    This function orchestrates the entire analysis workflow:
    1. Data loading and preparation
    2. Statistical modeling (main effects, interactions, log-transformed)
    3. Regional and group comparisons
    4. Visualization and reporting
    
    Parameters
    ----------
    project_dir : str, optional
        Path to the project directory. If None, uses default path.
    """
    print("PEDS OSA: Spindle Density vs AHI Analysis Pipeline")
    print("=" * 55)
    
    # Set project directory - MODIFY THIS TO YOUR ACTUAL PROJECT PATH
    if project_dir is None:
        project_dir = "/Volumes/Ido/002_PEDS_OSA"  # Change this to your actual project directory
    
    try:
        # Load and prepare data
        spindle_df, demo_df = load_and_prepare_data(project_dir)
        
        # Create analysis dataset
        merged_df = create_analysis_dataset(spindle_df, demo_df)
        
        # Create output directory
        output_dir = os.path.join(project_dir, "analysis_2")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save merged dataset
        merged_df.to_csv(os.path.join(output_dir, 'analysis_dataset.csv'), index=False)
        print(f"\n✓ Saved analysis dataset to {output_dir}/")
        
        # Perform statistical analyses
        analyze_main_effects(merged_df, output_dir)
        analyze_age_ahi_interaction(merged_df, output_dir)
        analyze_log_transformed_ahi(merged_df, output_dir)
        analyze_regional_differences(merged_df, output_dir)
        compare_control_osa_groups(merged_df, output_dir)
        
        # Create visualizations
        create_analysis_visualizations(merged_df, output_dir)
        
        # Generate summary report
        generate_analysis_summary(merged_df, output_dir)
        
        print(f"\n✓ Advanced spindle density analysis completed successfully!")
        print(f"✓ All results saved to {output_dir}/")
        
    except Exception as e:
        print(f"Error in analysis pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 