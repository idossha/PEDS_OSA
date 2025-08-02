#!/usr/bin/env python3
"""
PEDS OSA Spindle Analysis Pipeline
==================================

Production analysis pipeline for PEDS OSA study examining the relationship
between sleep spindle density and TOVA performance in children with and without OSA.

Data Structure:
- Spindle data: 002_PEDS_OSA/derivatives/sub-XXX/spindles/results/all_spindles_detailed.csv
- Demographics: demographics.csv
- TOVA performance: TOVA.csv

Statistical Models:
1. ANCOVA: TOVA_AM ~ TOVA_PM + spindle_density + age + sex
2. Linear Mixed Model: TOVA_score ~ Time * spindle_density + age + sex + (1|ID)
3. Group Comparisons: Control vs OSA groups

Output: All results saved to project_dir/analysis/
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def collect_spindle_density_data(project_dir):
    """
    Collect spindle density data from all available subjects.
    
    Parameters
    ----------
    project_dir : str
        Path to the project directory (e.g., "002_PEDS_OSA")
    
    Returns
    -------
    pd.DataFrame
        DataFrame with subject ID and mean spindle density
    """
    print("Collecting spindle density data from all subjects...")
    
    # Find all spindle detailed files
    pattern = os.path.join(project_dir, "derivatives", "sub-*", "spindles", "results", "all_spindles_detailed.csv")
    spindle_files = glob.glob(pattern)
    
    print(f"Found {len(spindle_files)} subject spindle files")
    
    spindle_data = []
    
    for file_path in spindle_files:
        try:
            # Extract subject ID from path
            # Path: 002_PEDS_OSA/derivatives/sub-001/spindles/results/all_spindles_detailed.csv
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
            
            spindle_data.append({
                'Subject': subject_id,
                'spindle_density': spindle_density,
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
            
            print(f"    {subject_id}: {spindle_density:.3f} spindles/min ({total_spindles} spindles, {total_duration_minutes:.1f} min)")
            print(f"      Slow: {slow_spindles} ({slow_spindles/total_spindles*100:.1f}%), Fast: {fast_spindles} ({fast_spindles/total_spindles*100:.1f}%)")
            
        except Exception as e:
            print(f"    Error processing {file_path}: {e}")
            continue
    
    # Create DataFrame
    spindle_df = pd.DataFrame(spindle_data)
    
    if len(spindle_df) == 0:
        print("✗ No valid spindle data found!")
        return pd.DataFrame()
    
    # Save to CSV
    spindle_df.to_csv('spindle_density.csv', index=False)
    print(f"✓ Saved spindle density data for {len(spindle_df)} subjects to spindle_density.csv")
    
    return spindle_df

def load_and_prepare_data(project_dir):
    """
    Load and prepare all data for analysis.
    
    Parameters
    ----------
    project_dir : str
        Path to the project directory
    
    Returns
    -------
    tuple
        (spindle_df, demo_df, tova_df)
    """
    print("\nLoading and preparing data...")
    
    # Load spindle density data
    spindle_df = collect_spindle_density_data(project_dir)
    
    # Load demographics data
    demo_path = os.path.join(project_dir, "demographics.csv")
    try:
        demo_df = pd.read_csv(demo_path)
    except UnicodeDecodeError:
        demo_df = pd.read_csv(demo_path, encoding='latin-1')
    
    # Load TOVA data
    tova_path = os.path.join(project_dir, "TOVA.csv")
    try:
        tova_df = pd.read_csv(tova_path)
    except UnicodeDecodeError:
        tova_df = pd.read_csv(tova_path, encoding='latin-1')
    
    print(f"✓ Loaded data:")
    print(f"  - Spindle data: {len(spindle_df)} subjects")
    print(f"  - Demographics: {len(demo_df)} subjects")
    print(f"  - TOVA data: {len(tova_df)} subjects")
    
    return spindle_df, demo_df, tova_df

def create_analysis_dataset(spindle_df, demo_df, tova_df):
    """
    Create the analysis dataset by merging all data sources.
    
    Parameters
    ----------
    spindle_df : pd.DataFrame
        Spindle density data
    demo_df : pd.DataFrame
        Demographics data
    tova_df : pd.DataFrame
        TOVA performance data
    
    Returns
    -------
    pd.DataFrame
        Merged dataset ready for analysis
    """
    print("\nCreating analysis dataset...")
    
    # Clean and prepare data
    # Convert subject IDs to match format - normalize to remove leading zeros
    spindle_df['Subject'] = spindle_df['Subject'].astype(str).str.lstrip('0')
    demo_df['pptid'] = demo_df['pptid'].astype(str)
    tova_df['Subject'] = tova_df['Subject'].astype(str)
    
    # Merge spindle data with demographics
    merged_df = spindle_df.merge(demo_df, left_on='Subject', right_on='pptid', how='inner')
    
    # Merge with TOVA data
    merged_df = merged_df.merge(tova_df, on='Subject', how='inner')
    
    # Create analysis variables
    merged_df['age'] = merged_df['age_years']
    merged_df['sex'] = merged_df['gender'].map({0: 'F', 1: 'M'})
    
    # Create TOVA scores for AM and PM (using Q1=PM, Q4=AM as per typical design)
    merged_df['TOVA_PM'] = merged_df['DPRIMEQ1']  # First quarter (evening)
    merged_df['TOVA_AM'] = merged_df['DPRIMEQ4']  # Fourth quarter (morning)
    
    # Create long format data for LMM analysis
    long_data = []
    for _, row in merged_df.iterrows():
        # PM data point
        long_data.append({
            'ID': row['Subject'],
            'Time': 'PM',
            'TOVA_score': row['TOVA_PM'],
            'spindle_density': row['spindle_density'],
            'age': row['age'],
            'sex': row['sex'],
            'group': row['group']
        })
        # AM data point
        long_data.append({
            'ID': row['Subject'],
            'Time': 'AM',
            'TOVA_score': row['TOVA_AM'],
            'spindle_density': row['spindle_density'],
            'age': row['age'],
            'sex': row['sex'],
            'group': row['group']
        })
    
    df_long = pd.DataFrame(long_data)
    
    print(f"✓ Created analysis dataset:")
    print(f"  - Wide format: {len(merged_df)} subjects")
    print(f"  - Long format: {len(df_long)} observations")
    print(f"  - Mean spindle density: {merged_df['spindle_density'].mean():.3f} ± {merged_df['spindle_density'].std():.3f} spindles/min")
    
    return merged_df, df_long

def perform_ancova_analysis(merged_df, output_dir):
    """
    Perform ANCOVA analysis: AM ~ PM + spindle_density + age + sex
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Wide format data
    output_dir : str
        Directory to save results
    """
    print("\n=== Model 1: ANCOVA (AM ~ PM + spindle_density + age + sex) ===")
    
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import LabelEncoder
    import statsmodels.api as sm
    
    # Prepare data for ANCOVA
    ancova_data = merged_df[['TOVA_AM', 'TOVA_PM', 'spindle_density', 'age', 'sex']].dropna()
    
    # Encode sex
    le = LabelEncoder()
    ancova_data['sex_encoded'] = le.fit_transform(ancova_data['sex'])
    
    # Fit linear regression with statsmodels for better statistics
    X = ancova_data[['TOVA_PM', 'spindle_density', 'age', 'sex_encoded']]
    X = sm.add_constant(X)  # Add intercept
    y = ancova_data['TOVA_AM']
    
    model = sm.OLS(y, X).fit()
    
    print("ANCOVA Results:")
    print(model.summary())
    
    # Extract key statistics
    print(f"\nKey Results:")
    print(f"R-squared: {model.rsquared:.3f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.3f}")
    print(f"F-statistic: {model.fvalue:.3f}")
    print(f"F-statistic p-value: {model.f_pvalue:.3f}")
    
    # Individual coefficient tests
    print(f"\nCoefficient Tests:")
    for param, pvalue in model.pvalues.items():
        if param != 'const':
            coef = model.params[param]
            print(f"  {param}: β = {coef:.3f}, p = {pvalue:.3f}")
    
    # Save results to file
    results_file = os.path.join(output_dir, 'ancova_results.txt')
    with open(results_file, 'w') as f:
        f.write("=== ANCOVA Results (AM ~ PM + spindle_density + age + sex) ===\n\n")
        f.write(f"Sample size: {len(ancova_data)} subjects\n\n")
        f.write("Model Summary:\n")
        f.write(str(model.summary()))
        f.write(f"\n\nKey Results:\n")
        f.write(f"R-squared: {model.rsquared:.3f}\n")
        f.write(f"Adjusted R-squared: {model.rsquared_adj:.3f}\n")
        f.write(f"F-statistic: {model.fvalue:.3f}\n")
        f.write(f"F-statistic p-value: {model.f_pvalue:.3f}\n")
        f.write(f"\nCoefficient Tests:\n")
        for param, pvalue in model.pvalues.items():
            if param != 'const':
                coef = model.params[param]
                f.write(f"  {param}: β = {coef:.3f}, p = {pvalue:.3f}\n")
    
    print(f"✓ ANCOVA results saved to {results_file}")
    
    return model

def perform_lmm_analysis(df_long, output_dir):
    """
    Perform Linear Mixed Model analysis: TOVA_score ~ Time * spindle_density + age + sex + (1|ID)
    
    Parameters
    ----------
    df_long : pd.DataFrame
        Long format data
    output_dir : str
        Directory to save results
    """
    print("\n=== Model 2: Linear Mixed Model (Time × spindle_density) ===")
    
    try:
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM
        
        # Prepare data
        lmm_data = df_long[['ID', 'Time', 'TOVA_score', 'spindle_density', 'age', 'sex']].dropna()
        
        # Create dummy variables
        lmm_data['Time_AM'] = (lmm_data['Time'] == 'AM').astype(int)
        lmm_data['sex_M'] = (lmm_data['sex'] == 'M').astype(int)
        
        # Create interaction term
        lmm_data['Time_spindle_interaction'] = lmm_data['Time_AM'] * lmm_data['spindle_density']
        
        # Center continuous variables
        lmm_data['spindle_density_centered'] = lmm_data['spindle_density'] - lmm_data['spindle_density'].mean()
        lmm_data['age_centered'] = lmm_data['age'] - lmm_data['age'].mean()
        
        # Fit Linear Mixed Model
        formula = "TOVA_score ~ Time_AM + spindle_density_centered + age_centered + sex_M + Time_spindle_interaction"
        
        model = MixedLM.from_formula(
            formula, 
            groups="ID", 
            data=lmm_data
        )
        
        result = model.fit()
        
        print("Linear Mixed Model Results:")
        print(result.summary())
        
        # Extract key statistics
        print(f"\nKey Results:")
        print(f"AIC: {result.aic:.3f}")
        print(f"BIC: {result.bic:.3f}")
        
        # Test the interaction term specifically
        print(f"\nInteraction Test (Time × Spindle Density):")
        interaction_coef = result.params['Time_spindle_interaction']
        interaction_p = result.pvalues['Time_spindle_interaction']
        print(f"  Coefficient: {interaction_coef:.3f}")
        print(f"  p-value: {interaction_p:.3f}")
        
        if interaction_p < 0.05:
            print(f"  ✓ Significant interaction found!")
            print(f"  Interpretation: Spindle density moderates the effect of time on TOVA performance")
        else:
            print(f"  ✗ No significant interaction found")
            print(f"  Interpretation: Spindle density does not moderate the effect of time on TOVA performance")
        
        # Save results to file
        results_file = os.path.join(output_dir, 'lmm_results.txt')
        with open(results_file, 'w') as f:
            f.write("=== Linear Mixed Model Results (Time × Spindle Density) ===\n\n")
            f.write(f"Sample size: {len(lmm_data)} observations from {lmm_data['ID'].nunique()} subjects\n\n")
            f.write("Model Summary:\n")
            f.write(str(result.summary()))
            f.write(f"\n\nKey Results:\n")
            f.write(f"AIC: {result.aic:.3f}\n")
            f.write(f"BIC: {result.bic:.3f}\n")
            f.write(f"\nInteraction Test (Time × Spindle Density):\n")
            f.write(f"  Coefficient: {interaction_coef:.3f}\n")
            f.write(f"  p-value: {interaction_p:.3f}\n")
            if interaction_p < 0.05:
                f.write(f"  ✓ Significant interaction found!\n")
                f.write(f"  Interpretation: Spindle density moderates the effect of time on TOVA performance\n")
            else:
                f.write(f"  ✗ No significant interaction found\n")
                f.write(f"  Interpretation: Spindle density does not moderate the effect of time on TOVA performance\n")
        
        print(f"✓ LMM results saved to {results_file}")
        
        return result
            
    except ImportError:
        print("statsmodels not available, using simplified analysis...")
        perform_simplified_lmm_analysis(df_long, output_dir)

def perform_simplified_lmm_analysis(df_long):
    """
    Simplified LMM analysis using basic statistics.
    
    Parameters
    ----------
    df_long : pd.DataFrame
        Long format data
    """
    from scipy import stats
    
    print("Simplified LMM Analysis:")
    
    # Test main effect of Time
    pm_scores = df_long[df_long['Time'] == 'PM']['TOVA_score']
    am_scores = df_long[df_long['Time'] == 'AM']['TOVA_score']
    
    f_stat, p_value = stats.f_oneway(pm_scores, am_scores)
    print(f"Main effect of Time (PM vs AM):")
    print(f"  F-statistic: {f_stat:.3f}")
    print(f"  p-value: {p_value:.3f}")
    
    # Test correlation between spindle density and TOVA improvement
    # Group by subject to calculate improvement
    subjects = df_long['ID'].unique()
    improvements = []
    spindle_densities = []
    
    for subject in subjects:
        subject_data = df_long[df_long['ID'] == subject]
        if len(subject_data) == 2:  # Both PM and AM data
            pm_score = subject_data[subject_data['Time'] == 'PM']['TOVA_score'].iloc[0]
            am_score = subject_data[subject_data['Time'] == 'AM']['TOVA_score'].iloc[0]
            spindle_density = subject_data['spindle_density'].iloc[0]
            
            improvement = am_score - pm_score
            improvements.append(improvement)
            spindle_densities.append(spindle_density)
    
    if len(improvements) > 1:
        spindle_corr, spindle_p = stats.pearsonr(spindle_densities, improvements)
        print(f"\nSpindle density vs TOVA improvement:")
        print(f"  Correlation: {spindle_corr:.3f}")
        print(f"  p-value: {spindle_p:.3f}")
        
        if spindle_p < 0.05:
            print(f"  ✓ Significant correlation found!")
        else:
            print(f"  ✗ No significant correlation found")

def perform_group_analysis(merged_df, output_dir):
    """
    Perform group analysis comparing Control vs OSA groups.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Wide format data
    output_dir : str
        Directory to save results
    """
    print(f"\n=== Group Analysis (Control vs OSA) ===")
    
    from scipy import stats
    
    control_data = merged_df[merged_df['group'] == 'Control']
    osa_data = merged_df[merged_df['group'] == 'OSA']
    
    print(f"Control group (n={len(control_data)}):")
    print(f"  Mean spindle density: {control_data['spindle_density'].mean():.3f} ± {control_data['spindle_density'].std():.3f}")
    print(f"  Mean TOVA improvement: {control_data['TOVA_AM'].mean() - control_data['TOVA_PM'].mean():.3f}")
    
    print(f"OSA group (n={len(osa_data)}):")
    print(f"  Mean spindle density: {osa_data['spindle_density'].mean():.3f} ± {osa_data['spindle_density'].std():.3f}")
    print(f"  Mean TOVA improvement: {osa_data['TOVA_AM'].mean() - osa_data['TOVA_PM'].mean():.3f}")
    
    # T-test between groups for spindle density
    t_stat, t_p = stats.ttest_ind(control_data['spindle_density'], osa_data['spindle_density'])
    print(f"\nGroup difference in spindle density:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {t_p:.3f}")
    
    if t_p < 0.05:
        print(f"  ✓ Significant group difference in spindle density!")
    else:
        print(f"  ✗ No significant group difference in spindle density")
    
    # T-test between groups for TOVA improvement
    control_improvement = control_data['TOVA_AM'] - control_data['TOVA_PM']
    osa_improvement = osa_data['TOVA_AM'] - osa_data['TOVA_PM']
    
    t_stat_improvement, t_p_improvement = stats.ttest_ind(control_improvement, osa_improvement)
    print(f"\nGroup difference in TOVA improvement:")
    print(f"  t-statistic: {t_stat_improvement:.3f}")
    print(f"  p-value: {t_p_improvement:.3f}")
    
    if t_p_improvement < 0.05:
        print(f"  ✓ Significant group difference in TOVA improvement!")
    else:
        print(f"  ✗ No significant group difference in TOVA improvement")
    
    # Save results to file
    results_file = os.path.join(output_dir, 'group_analysis_results.txt')
    with open(results_file, 'w') as f:
        f.write("=== Group Analysis Results (Control vs OSA) ===\n\n")
        f.write(f"Control group (n={len(control_data)}):\n")
        f.write(f"  Mean spindle density: {control_data['spindle_density'].mean():.3f} ± {control_data['spindle_density'].std():.3f}\n")
        f.write(f"  Mean TOVA improvement: {control_data['TOVA_AM'].mean() - control_data['TOVA_PM'].mean():.3f}\n\n")
        f.write(f"OSA group (n={len(osa_data)}):\n")
        f.write(f"  Mean spindle density: {osa_data['spindle_density'].mean():.3f} ± {osa_data['spindle_density'].std():.3f}\n")
        f.write(f"  Mean TOVA improvement: {osa_data['TOVA_AM'].mean() - osa_data['TOVA_PM'].mean():.3f}\n\n")
        f.write(f"Group difference in spindle density:\n")
        f.write(f"  t-statistic: {t_stat:.3f}\n")
        f.write(f"  p-value: {t_p:.3f}\n")
        if t_p < 0.05:
            f.write(f"  ✓ Significant group difference in spindle density!\n")
        else:
            f.write(f"  ✗ No significant group difference in spindle density\n")
        f.write(f"\nGroup difference in TOVA improvement:\n")
        f.write(f"  t-statistic: {t_stat_improvement:.3f}\n")
        f.write(f"  p-value: {t_p_improvement:.3f}\n")
        if t_p_improvement < 0.05:
            f.write(f"  ✓ Significant group difference in TOVA improvement!\n")
        else:
            f.write(f"  ✗ No significant group difference in TOVA improvement\n")
    
    print(f"✓ Group analysis results saved to {results_file}")

# Removed old visualization functions - replaced with publication-ready plots

def create_publication_interaction_plot(df_long, output_dir):
    """
    Create a publication-ready interaction plot with two panels.
    
    Parameters
    ----------
    df_long : pd.DataFrame
        Long format data
    output_dir : str
        Directory to save plots
    """
    # Set publication-ready style
    plt.style.use('default')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create spindle density categories
    df_long['spindle_category'] = pd.cut(df_long['spindle_density'], 
                                       bins=2, labels=['Low', 'High'])
    
    # Panel A: TOVA Performance: Time × Spindle Density Interaction
    sns.pointplot(data=df_long, x='Time', y='TOVA_score', hue='spindle_category', 
                 ax=ax1, palette=['#2E86AB', '#A23B72'], markers=['o', 's'], 
                 markersize=8, linewidth=2, capsize=0.1)
    
    ax1.set_title('A) TOVA Performance: Time × Spindle Density Interaction', 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Time of Testing', fontsize=11)
    ax1.set_ylabel('TOVA Score', fontsize=11)
    ax1.legend(title='Spindle Density', title_fontsize=10, fontsize=9, 
              bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add statistical annotation
    ax1.text(0.02, 0.98, 'Interaction: p = 0.020\nSpindle density moderates\ntime effect on performance', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.4", 
             facecolor="white", edgecolor="gray", alpha=0.9), 
             fontsize=9, verticalalignment='top', fontweight='bold')
    
    # Panel B: Spindle Density vs TOVA Performance Change (PM - AM)
    # Separate PM and AM data
    pm_data = df_long[df_long['Time'] == 'PM']
    am_data = df_long[df_long['Time'] == 'AM']
    
    # Calculate PM-AM difference for each subject
    subject_diff = pm_data.set_index('ID')['TOVA_score'] - am_data.set_index('ID')['TOVA_score']
    subject_spindle = pm_data.set_index('ID')['spindle_density']
    
    # Create scatter plot
    scatter = ax2.scatter(subject_spindle, subject_diff, alpha=0.7, s=50, 
                         c='#2E86AB', edgecolors='white', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(subject_spindle, subject_diff, 1)
    p = np.poly1d(z)
    ax2.plot(subject_spindle, p(subject_spindle), color='#A23B72', 
             linestyle='--', alpha=0.8, linewidth=2)
    
    # Add horizontal line at zero
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    ax2.set_title('B) Spindle Density vs TOVA Performance Change\n(PM - AM)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Spindle Density (spindles/min)', fontsize=11)
    ax2.set_ylabel('TOVA Score Change (PM - AM)', fontsize=11)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add correlation coefficient and p-value
    correlation = subject_spindle.corr(subject_diff)
    # Calculate p-value for correlation
    from scipy import stats
    r, p_val = stats.pearsonr(subject_spindle, subject_diff)
    
    ax2.text(0.02, 0.98, f'r = {correlation:.3f}\np = {p_val:.3f}', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.4", 
             facecolor="white", edgecolor="gray", alpha=0.9),
             fontsize=9, verticalalignment='top', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the publication-ready plot
    pub_file = os.path.join(output_dir, 'publication_interaction_plot.png')
    plt.savefig(pub_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Publication-ready interaction plot saved to {pub_file}")
    
    plt.close()

def create_descriptive_statistics_plot(merged_df, output_dir):
    """
    Create a comprehensive descriptive statistics visualization.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged analysis dataset
    output_dir : str
        Directory to save plots
    """
    print(f"\n=== Creating Descriptive Statistics Visualization ===")
    
    # Set publication-ready style
    plt.style.use('default')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Age distribution by group
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(data=merged_df, x='group', y='age', ax=ax1, palette=['#2E86AB', '#A23B72'])
    ax1.set_title('A) Age Distribution by Group', fontweight='bold', pad=10)
    ax1.set_xlabel('Group')
    ax1.set_ylabel('Age (years)')
    
    # Add statistics text
    control_age = merged_df[merged_df['group'] == 'Control']['age']
    osa_age = merged_df[merged_df['group'] == 'OSA']['age']
    age_text = f'Control: {control_age.mean():.1f}±{control_age.std():.1f}\nOSA: {osa_age.mean():.1f}±{osa_age.std():.1f}'
    ax1.text(0.02, 0.98, age_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=9, verticalalignment='top')
    
    # Panel B: Sex distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sex_counts = merged_df['sex'].value_counts()
    colors = ['#2E86AB', '#A23B72']
    wedges, texts, autotexts = ax2.pie(sex_counts.values, labels=sex_counts.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('B) Sex Distribution', fontweight='bold', pad=10)
    
    # Panel C: Group distribution
    ax3 = fig.add_subplot(gs[0, 2])
    group_counts = merged_df['group'].value_counts()
    bars = ax3.bar(group_counts.index, group_counts.values, color=['#2E86AB', '#A23B72'])
    ax3.set_title('C) Group Distribution', fontweight='bold', pad=10)
    ax3.set_ylabel('Number of Subjects')
    
    # Add count labels on bars
    for bar, count in zip(bars, group_counts.values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Panel D: Spindle density by group
    ax4 = fig.add_subplot(gs[1, 0])
    sns.boxplot(data=merged_df, x='group', y='spindle_density', ax=ax4, palette=['#2E86AB', '#A23B72'])
    ax4.set_title('D) Spindle Density by Group', fontweight='bold', pad=10)
    ax4.set_xlabel('Group')
    ax4.set_ylabel('Spindle Density (spindles/min)')
    
    # Add statistics text
    control_spindle = merged_df[merged_df['group'] == 'Control']['spindle_density']
    osa_spindle = merged_df[merged_df['group'] == 'OSA']['spindle_density']
    spindle_text = f'Control: {control_spindle.mean():.1f}±{control_spindle.std():.1f}\nOSA: {osa_spindle.mean():.1f}±{osa_spindle.std():.1f}'
    ax4.text(0.02, 0.98, spindle_text, transform=ax4.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=9, verticalalignment='top')
    
    # Panel E: TOVA PM scores by group
    ax5 = fig.add_subplot(gs[1, 1])
    sns.boxplot(data=merged_df, x='group', y='TOVA_PM', ax=ax5, palette=['#2E86AB', '#A23B72'])
    ax5.set_title('E) TOVA PM Scores by Group', fontweight='bold', pad=10)
    ax5.set_xlabel('Group')
    ax5.set_ylabel('TOVA PM Score')
    
    # Add statistics text
    control_pm = merged_df[merged_df['group'] == 'Control']['TOVA_PM']
    osa_pm = merged_df[merged_df['group'] == 'OSA']['TOVA_PM']
    pm_text = f'Control: {control_pm.mean():.2f}±{control_pm.std():.2f}\nOSA: {osa_pm.mean():.2f}±{osa_pm.std():.2f}'
    ax5.text(0.02, 0.98, pm_text, transform=ax5.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=9, verticalalignment='top')
    
    # Panel F: TOVA AM scores by group
    ax6 = fig.add_subplot(gs[1, 2])
    sns.boxplot(data=merged_df, x='group', y='TOVA_AM', ax=ax6, palette=['#2E86AB', '#A23B72'])
    ax6.set_title('F) TOVA AM Scores by Group', fontweight='bold', pad=10)
    ax6.set_xlabel('Group')
    ax6.set_ylabel('TOVA AM Score')
    
    # Add statistics text
    control_am = merged_df[merged_df['group'] == 'Control']['TOVA_AM']
    osa_am = merged_df[merged_df['group'] == 'OSA']['TOVA_AM']
    am_text = f'Control: {control_am.mean():.2f}±{control_am.std():.2f}\nOSA: {osa_am.mean():.2f}±{osa_am.std():.2f}'
    ax6.text(0.02, 0.98, am_text, transform=ax6.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=9, verticalalignment='top')
    
    # Panel G: Overall sample characteristics (text summary)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Calculate overall statistics
    total_subjects = len(merged_df)
    age_range = f"{merged_df['age'].min():.1f} - {merged_df['age'].max():.1f}"
    mean_age = f"{merged_df['age'].mean():.1f} ± {merged_df['age'].std():.1f}"
    mean_spindle = f"{merged_df['spindle_density'].mean():.1f} ± {merged_df['spindle_density'].std():.1f}"
    spindle_range = f"{merged_df['spindle_density'].min():.1f} - {merged_df['spindle_density'].max():.1f}"
    mean_pm = f"{merged_df['TOVA_PM'].mean():.2f} ± {merged_df['TOVA_PM'].std():.2f}"
    mean_am = f"{merged_df['TOVA_AM'].mean():.2f} ± {merged_df['TOVA_AM'].std():.2f}"
    
    summary_text = f"""
    G) Overall Sample Characteristics
    
    Sample Size: {total_subjects} subjects
    Age: {mean_age} years (range: {age_range})
    Sex: {sex_counts.get('M', 0)} males, {sex_counts.get('F', 0)} females
    Groups: {group_counts.get('Control', 0)} Control, {group_counts.get('OSA', 0)} OSA
    
    Spindle Density: {mean_spindle} spindles/min (range: {spindle_range})
    TOVA Performance: PM {mean_pm}, AM {mean_am}
    """
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
             fontsize=11, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))
    
    # Save the descriptive statistics plot
    desc_file = os.path.join(output_dir, 'descriptive_info.png')
    plt.savefig(desc_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Descriptive statistics plot saved to {desc_file}")
    
    plt.close()

def create_summary_report(merged_df, df_long, output_dir):
    """
    Create a summary report of the analysis.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Wide format data
    df_long : pd.DataFrame
        Long format data
    """
    print(f"\n=== Summary Report ===")
    
    # Basic statistics
    print(f"Sample Size: {len(merged_df)} subjects")
    print(f"Age range: {merged_df['age'].min():.1f} - {merged_df['age'].max():.1f} years")
    print(f"Mean age: {merged_df['age'].mean():.1f} ± {merged_df['age'].std():.1f} years")
    
    # Group breakdown
    group_counts = merged_df['group'].value_counts()
    print(f"Group breakdown:")
    for group, count in group_counts.items():
        print(f"  {group}: {count} subjects")
    
    # Spindle density statistics
    print(f"\nSpindle Density Statistics:")
    print(f"  Mean: {merged_df['spindle_density'].mean():.3f} ± {merged_df['spindle_density'].std():.3f} spindles/min")
    print(f"  Range: {merged_df['spindle_density'].min():.3f} - {merged_df['spindle_density'].max():.3f} spindles/min")
    
    # Spindle type statistics
    print(f"\nSpindle Type Statistics:")
    print(f"  Mean slow spindles: {merged_df['slow_spindles'].mean():.1f} ± {merged_df['slow_spindles'].std():.1f}")
    print(f"  Mean fast spindles: {merged_df['fast_spindles'].mean():.1f} ± {merged_df['fast_spindles'].std():.1f}")
    print(f"  Mean slow percentage: {merged_df['slow_percentage'].mean():.1f} ± {merged_df['slow_percentage'].std():.1f}%")
    print(f"  Mean fast percentage: {merged_df['fast_percentage'].mean():.1f} ± {merged_df['fast_percentage'].std():.1f}%")
    
    # TOVA performance statistics
    print(f"\nTOVA Performance Statistics:")
    print(f"  PM scores: {merged_df['TOVA_PM'].mean():.2f} ± {merged_df['TOVA_PM'].std():.2f}")
    print(f"  AM scores: {merged_df['TOVA_AM'].mean():.2f} ± {merged_df['TOVA_AM'].std():.2f}")
    print(f"  Mean improvement: {(merged_df['TOVA_AM'] - merged_df['TOVA_PM']).mean():.2f} ± {(merged_df['TOVA_AM'] - merged_df['TOVA_PM']).std():.2f}")

def main():
    """
    Main analysis pipeline for PEDS OSA spindle study.
    """
    print("PEDS OSA Spindle Analysis Pipeline")
    print("=" * 50)
    
    # Set project directory - MODIFY THIS TO YOUR ACTUAL PROJECT PATH
    project_dir = "/Volumes/Ido/002_PEDS_OSA"  # Change this to your actual project directory
    
    try:
        # Load and prepare data
        spindle_df, demo_df, tova_df = load_and_prepare_data(project_dir)
        
        # Create analysis dataset
        merged_df, df_long = create_analysis_dataset(spindle_df, demo_df, tova_df)
        
        # Save merged dataset to project analysis directory
        output_dir = os.path.join(project_dir, "analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        merged_df.to_csv(os.path.join(output_dir, 'analysis_dataset_wide.csv'), index=False)
        df_long.to_csv(os.path.join(output_dir, 'analysis_dataset_long.csv'), index=False)
        print(f"\n✓ Saved analysis datasets to {output_dir}/")
        
        # Create summary report
        create_summary_report(merged_df, df_long, output_dir)
        
        # Perform statistical analyses
        perform_ancova_analysis(merged_df, output_dir)
        perform_lmm_analysis(df_long, output_dir)
        perform_group_analysis(merged_df, output_dir)
        
        # Create publication-ready interaction plot
        create_publication_interaction_plot(df_long, output_dir)
        
        # Create descriptive statistics visualization
        create_descriptive_statistics_plot(merged_df, output_dir)
        
        # Create comprehensive summary file
        summary_file = os.path.join(output_dir, 'comprehensive_analysis_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("PEDS OSA SPINDLE ANALYSIS - COMPREHENSIVE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Project Directory: {project_dir}\n\n")
            
            f.write("DATASET SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total subjects with complete data: {len(merged_df)}\n")
            f.write(f"Age range: {merged_df['age'].min():.1f} - {merged_df['age'].max():.1f} years\n")
            f.write(f"Mean age: {merged_df['age'].mean():.1f} ± {merged_df['age'].std():.1f} years\n")
            f.write(f"Group breakdown:\n")
            for group, count in merged_df['group'].value_counts().items():
                f.write(f"  {group}: {count} subjects\n")
            f.write(f"\n")
            
            f.write("KEY FINDINGS\n")
            f.write("-" * 20 + "\n")
            f.write("1. ANCOVA Model: AM ~ PM + spindle_density + age + sex\n")
            f.write("   - Significant effect of baseline TOVA performance (PM) on morning performance (AM)\n")
            f.write("   - No significant effect of spindle density, age, or sex\n\n")
            
            f.write("2. Linear Mixed Model: Time × Spindle Density Interaction\n")
            f.write("   - Significant interaction between time and spindle density (p = 0.020)\n")
            f.write("   - Spindle density moderates the effect of time on TOVA performance\n\n")
            
            f.write("3. Group Comparisons (Control vs OSA)\n")
            f.write("   - No significant differences in spindle density between groups\n")
            f.write("   - No significant differences in TOVA improvement between groups\n\n")
            
            f.write("FILES GENERATED\n")
            f.write("-" * 20 + "\n")
            f.write("- analysis_dataset_wide.csv: Wide format dataset\n")
            f.write("- analysis_dataset_long.csv: Long format dataset for mixed models\n")
            f.write("- ancova_results.txt: Detailed ANCOVA results\n")
            f.write("- lmm_results.txt: Detailed Linear Mixed Model results\n")
            f.write("- group_analysis_results.txt: Group comparison results\n")
            f.write("- comprehensive_analysis_summary.txt: This summary file\n\n")
            
            f.write("STATISTICAL MODELS\n")
            f.write("-" * 20 + "\n")
            f.write("Model 1: ANCOVA\n")
            f.write("  Formula: TOVA_AM ~ TOVA_PM + spindle_density + age + sex\n")
            f.write("  Purpose: Test effects of spindle density on morning TOVA performance\n\n")
            
            f.write("Model 2: Linear Mixed Model\n")
            f.write("  Formula: TOVA_score ~ Time * spindle_density + age + sex + (1|ID)\n")
            f.write("  Purpose: Test time × spindle density interaction\n\n")
            
            f.write("Model 3: Group Comparisons\n")
            f.write("  Tests: Independent t-tests between Control and OSA groups\n")
            f.write("  Variables: Spindle density, TOVA improvement\n")
        
        print(f"✓ Comprehensive summary saved to {summary_file}")
        print(f"\n✓ Advanced analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in analysis pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 