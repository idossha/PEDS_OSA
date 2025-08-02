#!/usr/bin/env python3
"""
PEDS OSA: Regional AHI vs Spindle Density Analysis
=================================================

Complete regional analysis examining the relationship between log(AHI) and sleep spindle density
in anterior vs posterior brain regions.

Research Questions:
1. Does log(AHI) predict anterior spindle density, controlling for age and sex?
2. Does log(AHI) predict posterior spindle density, controlling for age and sex?
3. Which brain region shows stronger associations with sleep apnea severity?

Output: Two publication figures:
- main_results_figure.png (3 panels)
- demographics_dashboard.png (6 panels)
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class RegionalAHIAnalysis:
    """Regional AHI analysis pipeline."""
    
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.output_dir = os.path.join(project_dir, "regional_ahi_results")
        self.anterior_channels = None
        self.posterior_channels = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_channel_definitions(self):
        """Load anterior and posterior channel definitions."""
        segment_file = os.path.join(os.path.dirname(__file__), "reference", "segment_net.txt")
        
        if not os.path.exists(segment_file):
            print(f"Warning: Channel definition file not found: {segment_file}")
            # Use default channel indices
            self.anterior_channels = [str(i) for i in range(1, 41)]
            self.posterior_channels = [str(i) for i in range(70, 100)]
            return
        
        try:
            with open(segment_file, 'r') as f:
                content = f.read()
            
            # Extract anterior channels section
            anterior_section = content.split('Anterior_channels:')[1].split('Posterior_channels:')[0]
            
            # Parse anterior channel names
            anterior_channels = []
            for line in anterior_section.strip().split('\n'):
                if line.strip():
                    line_channels = [ch.strip() for ch in line.split(',') if ch.strip()]
                    anterior_channels.extend(line_channels)
            
            # Convert to numeric format
            anterior_numeric = []
            for channel in anterior_channels:
                if channel.startswith('E'):
                    num = int(channel[1:])
                    anterior_numeric.append(str(num))
                elif channel.isdigit():
                    anterior_numeric.append(channel)
                else:
                    import re
                    digits = re.findall(r'\d+', channel)
                    if digits:
                        anterior_numeric.append(str(int(digits[0])))
            
            # Extract posterior channels section
            posterior_section = content.split('Posterior_channels:')[1]
            
            # Parse posterior channel names
            posterior_channels = []
            for line in posterior_section.strip().split('\n'):
                if line.strip():
                    line_channels = [ch.strip() for ch in line.split(',') if ch.strip()]
                    posterior_channels.extend(line_channels)
            
            # Convert to numeric format
            posterior_numeric = []
            for channel in posterior_channels:
                if channel.startswith('E'):
                    num = int(channel[1:])
                    posterior_numeric.append(str(num))
                elif channel.isdigit():
                    posterior_numeric.append(channel)
                else:
                    import re
                    digits = re.findall(r'\d+', channel)
                    if digits:
                        posterior_numeric.append(str(int(digits[0])))
            
            self.anterior_channels = anterior_numeric
            self.posterior_channels = posterior_numeric
            
            print(f"Loaded {len(self.anterior_channels)} anterior channels")
            print(f"Loaded {len(self.posterior_channels)} posterior channels")
            
        except Exception as e:
            print(f"Error loading channel definitions: {e}")
            # Fallback to default ranges
            self.anterior_channels = [str(i) for i in range(1, 41)]
            self.posterior_channels = [str(i) for i in range(70, 100)]
    
    def collect_regional_spindle_data(self, region='anterior'):
        """Collect spindle density data for specified region."""
        channels = self.anterior_channels if region == 'anterior' else self.posterior_channels
        region_name = region.capitalize()
        
        print(f"Collecting {region_name} spindle density data...")
        
        # Find all spindle detailed files
        pattern = os.path.join(self.project_dir, "derivatives", "sub-*", "spindles", "results", "all_spindles_detailed.csv")
        spindle_files = glob.glob(pattern)
        
        print(f"Found {len(spindle_files)} subject spindle files")
        
        spindle_data = []
        
        for file_path in spindle_files:
            try:
                # Extract subject ID
                path_parts = file_path.split(os.sep)
                subject_folder = [part for part in path_parts if part.startswith('sub-')][0]
                subject_id = subject_folder.replace("sub-", "")
                
                # Read spindle data
                try:
                    spindle_df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        spindle_df = pd.read_csv(file_path, encoding='latin-1')
                    except:
                        spindle_df = pd.read_csv(file_path, encoding='cp1252')
                
                # Filter for specified channels
                if 'Channel' not in spindle_df.columns:
                    continue
                
                # Convert channel list to integers for filtering
                channels_int = [int(ch) for ch in channels]
                region_spindles = spindle_df[spindle_df['Channel'].isin(channels_int)]
                
                if len(region_spindles) == 0:
                    continue
                
                # Calculate recording duration from filtered data
                total_duration_seconds = region_spindles['End'].max()
                total_duration_minutes = total_duration_seconds / 60
                
                # Calculate regional spindle density
                total_spindles = len(region_spindles)
                spindle_density = total_spindles / total_duration_minutes if total_duration_minutes > 0 else 0
                
                # Additional characteristics
                mean_duration = region_spindles['Duration'].mean() if len(region_spindles) > 0 else 0
                mean_frequency = region_spindles['Frequency'].mean() if len(region_spindles) > 0 else 0
                mean_amplitude = region_spindles['Amplitude'].mean() if len(region_spindles) > 0 else 0
                
                spindle_data.append({
                    'Subject': subject_id,
                    f'{region}_spindle_density': spindle_density,
                    f'{region}_total_spindles': total_spindles,
                    f'{region}_mean_duration': mean_duration,
                    f'{region}_mean_frequency': mean_frequency,
                    f'{region}_mean_amplitude': mean_amplitude,
                    'recording_duration_min': total_duration_minutes
                })
                
            except Exception as e:
                print(f"  Error processing {subject_id}: {e}")
                continue
        
        df = pd.DataFrame(spindle_data)
        print(f"✓ Collected {region} spindle data for {len(df)} subjects")
        return df
    
    def load_and_prepare_data(self):
        """Load and prepare all data for analysis."""
        print("\nLoading and preparing data...")
        
        # Load regional spindle data
        anterior_df = self.collect_regional_spindle_data('anterior')
        posterior_df = self.collect_regional_spindle_data('posterior')
        
        # Load demographics
        demo_file = os.path.join(self.project_dir, "demographics.csv")
        if not os.path.exists(demo_file):
            raise FileNotFoundError(f"Demographics file not found: {demo_file}")
        
        if os.path.exists(demo_file):
            demo_df = pd.read_csv(demo_file)
            print(f"✓ Loaded demographics: {len(demo_df)} subjects")
        else:
            raise FileNotFoundError(f"Demographics file not found: {demo_file}")
        
        return anterior_df, posterior_df, demo_df
    
    def create_analysis_dataset(self, anterior_df, posterior_df, demo_df):
        """Create complete analysis dataset."""
        print("\nCreating complete analysis dataset...")
        
        # Merge anterior and posterior data on Subject only 
        # (not recording duration which may have slight differences due to filtering)
        merged_spindle = anterior_df.merge(posterior_df, on='Subject', how='inner', suffixes=('_ant', '_post'))
        
        # Use the anterior recording duration (they should be very similar)
        merged_spindle['recording_duration_min'] = merged_spindle['recording_duration_min_ant']
        
        # Normalize subject IDs for matching
        merged_spindle['Subject'] = merged_spindle['Subject'].astype(str).str.lstrip('0')
        demo_df['pptid'] = demo_df['pptid'].astype(str)
        
        # Merge with demographics
        final_df = merged_spindle.merge(demo_df, left_on='Subject', right_on='pptid', how='inner')
        
        # Create standardized variables
        final_df['age'] = final_df['age_years'] + (final_df['age_months'] >= 6).astype(int)
        final_df['sex'] = final_df['gender'].map({0: 'F', 1: 'M'})
        final_df['AHI'] = final_df['nrem_ahi']
        final_df['group'] = final_df['group']
        
        # Create log-transformed AHI
        final_df['log_AHI'] = np.log1p(final_df['AHI'])
        
        # Remove subjects with missing critical data
        final_df = final_df.dropna(subset=['anterior_spindle_density', 'posterior_spindle_density', 'age', 'sex', 'AHI'])
        
        print(f"✓ Complete dataset: {len(final_df)} subjects with all data")
        print(f"  - Mean age: {final_df['age'].mean():.1f} ± {final_df['age'].std():.1f} years")
        print(f"  - Sex: {final_df['sex'].value_counts().to_dict()}")
        print(f"  - Groups: {final_df['group'].value_counts().to_dict()}")
        print(f"  - Mean AHI: {final_df['AHI'].mean():.2f} ± {final_df['AHI'].std():.2f}")
        
        return final_df
    
    def perform_statistical_analysis(self, df):
        """Perform comprehensive statistical analysis by group."""
        print("\nPerforming statistical analysis by group...")
        
        results = {}
        
        # Split data by group
        control_data = df[df['group'] == 'Control']
        osa_data = df[df['group'] == 'OSA']
        
        print(f"Control group: n = {len(control_data)}")
        print(f"OSA group: n = {len(osa_data)}")
        
        # Helper function for Steiger's Z-test
        def steiger_z_test(r1, r2, n):
            """Test difference between two dependent correlations."""
            if n <= 3:  # Not enough data for test
                return np.nan, np.nan
            z1 = 0.5 * np.log((1 + r1) / (1 - r1))
            z2 = 0.5 * np.log((1 + r2) / (1 - r2))
            se_diff = np.sqrt(2 / (n - 3))
            z = (z1 - z2) / se_diff
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            return z, p
        
        # Analyze each group separately
        for group_name, group_data in [('control', control_data), ('osa', osa_data)]:
            print(f"\n{group_name.upper()} GROUP ANALYSIS:")
            
            if len(group_data) < 5:  # Need minimum data for analysis
                print(f"  Insufficient data for {group_name} group (n={len(group_data)})")
                results[group_name] = {'insufficient_data': True}
                continue
            
            # 1. Anterior region analysis
            print(f"  1. Anterior region - {group_name}...")
            try:
                anterior_model = ols('anterior_spindle_density ~ log_AHI + age + sex', data=group_data).fit()
                anterior_corr, anterior_corr_p = stats.pearsonr(group_data['log_AHI'], group_data['anterior_spindle_density'])
            except:
                # Fallback without covariates if too few subjects
                anterior_model = ols('anterior_spindle_density ~ log_AHI', data=group_data).fit()
                anterior_corr, anterior_corr_p = stats.pearsonr(group_data['log_AHI'], group_data['anterior_spindle_density'])
            
            # 2. Posterior region analysis
            print(f"  2. Posterior region - {group_name}...")
            try:
                posterior_model = ols('posterior_spindle_density ~ log_AHI + age + sex', data=group_data).fit()
                posterior_corr, posterior_corr_p = stats.pearsonr(group_data['log_AHI'], group_data['posterior_spindle_density'])
            except:
                # Fallback without covariates if too few subjects
                posterior_model = ols('posterior_spindle_density ~ log_AHI', data=group_data).fit()
                posterior_corr, posterior_corr_p = stats.pearsonr(group_data['log_AHI'], group_data['posterior_spindle_density'])
            
            # 3. Combined regional model
            print(f"  3. Combined regional model - {group_name}...")
            group_data_copy = group_data.copy()
            group_data_copy['avg_spindle_density'] = (group_data_copy['anterior_spindle_density'] + group_data_copy['posterior_spindle_density']) / 2
            try:
                combined_model = ols('avg_spindle_density ~ log_AHI + age + sex', data=group_data_copy).fit()
            except:
                combined_model = ols('avg_spindle_density ~ log_AHI', data=group_data_copy).fit()
            
            # 4. Regional comparison (Steiger's Z-test)
            print(f"  4. Regional comparison - {group_name}...")
            steiger_z, steiger_p = steiger_z_test(anterior_corr, posterior_corr, len(group_data))
            
            # Store results for this group
            results[group_name] = {
                'n': len(group_data),
                'anterior': {
                    'model': anterior_model,
                    'correlation': anterior_corr,
                    'correlation_p': anterior_corr_p,
                    'r_squared': anterior_model.rsquared,
                    'f_statistic': anterior_model.fvalue,
                    'f_p_value': anterior_model.f_pvalue
                },
                'posterior': {
                    'model': posterior_model,
                    'correlation': posterior_corr,
                    'correlation_p': posterior_corr_p,
                    'r_squared': posterior_model.rsquared,
                    'f_statistic': posterior_model.fvalue,
                    'f_p_value': posterior_model.f_pvalue
                },
                'combined': {
                    'model': combined_model,
                    'r_squared': combined_model.rsquared,
                    'f_statistic': combined_model.fvalue,
                    'f_p_value': combined_model.f_pvalue
                },
                'comparison': {
                    'steiger_z': steiger_z,
                    'steiger_p': steiger_p
                }
            }
        
        # Overall combined analysis for reference
        print("\nOVERALL COMBINED ANALYSIS (for reference):")
        df['avg_spindle_density'] = (df['anterior_spindle_density'] + df['posterior_spindle_density']) / 2
        overall_anterior_corr, overall_anterior_p = stats.pearsonr(df['log_AHI'], df['anterior_spindle_density'])
        overall_posterior_corr, overall_posterior_p = stats.pearsonr(df['log_AHI'], df['posterior_spindle_density'])
        
        results['overall'] = {
            'anterior_correlation': overall_anterior_corr,
            'anterior_correlation_p': overall_anterior_p,
            'posterior_correlation': overall_posterior_corr,
            'posterior_correlation_p': overall_posterior_p
        }
        
        # 5. Group comparison analysis
        print("\nGROUP COMPARISON ANALYSIS:")
        
        # Group t-tests
        ahi_t, ahi_p = stats.ttest_ind(control_data['AHI'], osa_data['AHI'])
        anterior_t, anterior_p = stats.ttest_ind(control_data['anterior_spindle_density'], osa_data['anterior_spindle_density'])
        posterior_t, posterior_p = stats.ttest_ind(control_data['posterior_spindle_density'], osa_data['posterior_spindle_density'])
        
        results['group_comparison'] = {
            'control_n': len(control_data),
            'osa_n': len(osa_data),
            'ahi_ttest': (ahi_t, ahi_p),
            'anterior_ttest': (anterior_t, anterior_p),
            'posterior_ttest': (posterior_t, posterior_p)
        }
        
        print("✓ Statistical analysis completed")
        return results
    
    def create_main_results_figure(self, df, results):
        """Create main results figure with group-specific analyses."""
        print("Creating main results figure with group-specific analyses...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Regional AHI Analysis Results by Group', fontsize=16, fontweight='bold')
        
        # Define colors for groups
        group_colors = {'Control': 'blue', 'OSA': 'red'}
        
        # Panel 1: Anterior region - Control
        ax1 = axes[0, 0]
        control_data = df[df['group'] == 'Control']
        if len(control_data) > 0 and 'control' in results and 'insufficient_data' not in results['control']:
            sns.scatterplot(data=control_data, x='log_AHI', y='anterior_spindle_density', 
                          color=group_colors['Control'], ax=ax1, alpha=0.7, s=60)
            sns.regplot(data=control_data, x='log_AHI', y='anterior_spindle_density', 
                       scatter=False, ax=ax1, color=group_colors['Control'])
            r_val = results['control']['anterior']['correlation']
            p_val = results['control']['anterior']['correlation_p']
            ax1.set_title(f'Anterior - Control (n={results["control"]["n"]})\nr = {r_val:.3f}, p = {p_val:.4f}')
        else:
            ax1.set_title('Anterior - Control\n(Insufficient data)')
        ax1.set_xlabel('Log(AHI + 1)')
        ax1.set_ylabel('Anterior Spindle Density\n(spindles/min)')
        
        # Panel 2: Anterior region - OSA
        ax2 = axes[0, 1]
        osa_data = df[df['group'] == 'OSA']
        if len(osa_data) > 0 and 'osa' in results and 'insufficient_data' not in results['osa']:
            sns.scatterplot(data=osa_data, x='log_AHI', y='anterior_spindle_density', 
                          color=group_colors['OSA'], ax=ax2, alpha=0.7, s=60)
            sns.regplot(data=osa_data, x='log_AHI', y='anterior_spindle_density', 
                       scatter=False, ax=ax2, color=group_colors['OSA'])
            r_val = results['osa']['anterior']['correlation']
            p_val = results['osa']['anterior']['correlation_p']
            ax2.set_title(f'Anterior - OSA (n={results["osa"]["n"]})\nr = {r_val:.3f}, p = {p_val:.4f}')
        else:
            ax2.set_title('Anterior - OSA\n(Insufficient data)')
        ax2.set_xlabel('Log(AHI + 1)')
        ax2.set_ylabel('Anterior Spindle Density\n(spindles/min)')
        
        # Panel 3: Anterior region - Combined for comparison
        ax3 = axes[0, 2]
        sns.scatterplot(data=df, x='log_AHI', y='anterior_spindle_density', hue='group', ax=ax3, alpha=0.7, s=50)
        # Plot separate regression lines for each group
        if len(control_data) > 0:
            sns.regplot(data=control_data, x='log_AHI', y='anterior_spindle_density', 
                       scatter=False, ax=ax3, color=group_colors['Control'], label='Control')
        if len(osa_data) > 0:
            sns.regplot(data=osa_data, x='log_AHI', y='anterior_spindle_density', 
                       scatter=False, ax=ax3, color=group_colors['OSA'], label='OSA')
        
        if 'overall' in results:
            overall_r = results['overall']['anterior_correlation']
            overall_p = results['overall']['anterior_correlation_p']
            ax3.set_title(f'Anterior - Combined\nr = {overall_r:.3f}, p = {overall_p:.4f}')
        else:
            ax3.set_title('Anterior - Combined')
        ax3.set_xlabel('Log(AHI + 1)')
        ax3.set_ylabel('Anterior Spindle Density\n(spindles/min)')
        ax3.legend(title='Group', loc='upper right')
        
        # Panel 4: Posterior region - Control
        ax4 = axes[1, 0]
        if len(control_data) > 0 and 'control' in results and 'insufficient_data' not in results['control']:
            sns.scatterplot(data=control_data, x='log_AHI', y='posterior_spindle_density', 
                          color=group_colors['Control'], ax=ax4, alpha=0.7, s=60)
            sns.regplot(data=control_data, x='log_AHI', y='posterior_spindle_density', 
                       scatter=False, ax=ax4, color=group_colors['Control'])
            r_val = results['control']['posterior']['correlation']
            p_val = results['control']['posterior']['correlation_p']
            ax4.set_title(f'Posterior - Control (n={results["control"]["n"]})\nr = {r_val:.3f}, p = {p_val:.4f}')
        else:
            ax4.set_title('Posterior - Control\n(Insufficient data)')
        ax4.set_xlabel('Log(AHI + 1)')
        ax4.set_ylabel('Posterior Spindle Density\n(spindles/min)')
        
        # Panel 5: Posterior region - OSA
        ax5 = axes[1, 1]
        if len(osa_data) > 0 and 'osa' in results and 'insufficient_data' not in results['osa']:
            sns.scatterplot(data=osa_data, x='log_AHI', y='posterior_spindle_density', 
                          color=group_colors['OSA'], ax=ax5, alpha=0.7, s=60)
            sns.regplot(data=osa_data, x='log_AHI', y='posterior_spindle_density', 
                       scatter=False, ax=ax5, color=group_colors['OSA'])
            r_val = results['osa']['posterior']['correlation']
            p_val = results['osa']['posterior']['correlation_p']
            ax5.set_title(f'Posterior - OSA (n={results["osa"]["n"]})\nr = {r_val:.3f}, p = {p_val:.4f}')
        else:
            ax5.set_title('Posterior - OSA\n(Insufficient data)')
        ax5.set_xlabel('Log(AHI + 1)')
        ax5.set_ylabel('Posterior Spindle Density\n(spindles/min)')
        
        # Panel 6: Posterior region - Combined for comparison
        ax6 = axes[1, 2]
        sns.scatterplot(data=df, x='log_AHI', y='posterior_spindle_density', hue='group', ax=ax6, alpha=0.7, s=50)
        # Plot separate regression lines for each group
        if len(control_data) > 0:
            sns.regplot(data=control_data, x='log_AHI', y='posterior_spindle_density', 
                       scatter=False, ax=ax6, color=group_colors['Control'], label='Control')
        if len(osa_data) > 0:
            sns.regplot(data=osa_data, x='log_AHI', y='posterior_spindle_density', 
                       scatter=False, ax=ax6, color=group_colors['OSA'], label='OSA')
        
        if 'overall' in results:
            overall_r = results['overall']['posterior_correlation']
            overall_p = results['overall']['posterior_correlation_p']
            ax6.set_title(f'Posterior - Combined\nr = {overall_r:.3f}, p = {overall_p:.4f}')
        else:
            ax6.set_title('Posterior - Combined')
        ax6.set_xlabel('Log(AHI + 1)')
        ax6.set_ylabel('Posterior Spindle Density\n(spindles/min)')
        ax6.legend(title='Group', loc='upper right')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'main_results_figure.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Main results figure saved to {output_file}")
    
    def create_demographics_dashboard(self, df):
        """Create demographics dashboard (6 panels)."""
        print("Creating demographics dashboard...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Demographics and Sample Characteristics', fontsize=16, fontweight='bold')
        
        # Panel 1: Age distribution by group
        ax1 = axes[0, 0]
        sns.boxplot(data=df, x='group', y='age', ax=ax1)
        ax1.set_title('Age Distribution by Group')
        ax1.set_ylabel('Age (years)')
        
        # Panel 2: AHI distribution by group
        ax2 = axes[0, 1]
        sns.boxplot(data=df, x='group', y='AHI', ax=ax2)
        ax2.set_title('AHI Distribution by Group')
        ax2.set_ylabel('AHI')
        ax2.set_yscale('log')
        
        # Panel 3: Sex distribution
        ax3 = axes[0, 2]
        sex_counts = df.groupby(['group', 'sex']).size().unstack(fill_value=0)
        sex_counts.plot(kind='bar', ax=ax3, color=['pink', 'lightblue'])
        ax3.set_title('Sex Distribution by Group')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=0)
        ax3.legend(title='Sex')
        
        # Panel 4: Anterior spindle density by group
        ax4 = axes[1, 0]
        sns.boxplot(data=df, x='group', y='anterior_spindle_density', ax=ax4)
        ax4.set_title('Anterior Spindle Density')
        ax4.set_ylabel('Spindles/min')
        
        # Panel 5: Posterior spindle density by group
        ax5 = axes[1, 1]
        sns.boxplot(data=df, x='group', y='posterior_spindle_density', ax=ax5)
        ax5.set_title('Posterior Spindle Density')
        ax5.set_ylabel('Spindles/min')
        
        # Panel 6: Age vs AHI scatter
        ax6 = axes[1, 2]
        sns.scatterplot(data=df, x='age', y='AHI', hue='group', ax=ax6, alpha=0.7)
        ax6.set_title('Age vs AHI by Group')
        ax6.set_xlabel('Age (years)')
        ax6.set_ylabel('AHI')
        ax6.set_yscale('log')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'demographics_dashboard.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Demographics dashboard saved to {output_file}")
    
    def save_results_summary(self, df, results):
        """Save comprehensive technical summary with group-specific analyses."""
        summary_file = os.path.join(self.output_dir, 'analysis_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("PEDS OSA REGIONAL AHI ANALYSIS - COMPREHENSIVE TECHNICAL SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("ANALYSIS PIPELINE DETAILS\n")
            f.write("-" * 25 + "\n")
            f.write("Script: regional_ahi_analysis.py\n")
            f.write("Analysis Date: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("Python Environment: pandas, numpy, scipy, statsmodels, matplotlib, seaborn\n")
            f.write("Statistical Framework: OLS regression, Pearson correlation, Steiger's Z-test\n\n")
            
            f.write("SAMPLE CHARACTERISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total subjects: {len(df)}\n")
            f.write(f"Mean age: {df['age'].mean():.1f} ± {df['age'].std():.1f} years (range: {df['age'].min():.0f}-{df['age'].max():.0f})\n")
            f.write(f"Sex distribution: {df['sex'].value_counts().to_dict()}\n")
            f.write(f"Group distribution: {df['group'].value_counts().to_dict()}\n\n")
            
            f.write("SLEEP APNEA AND SPINDLE DENSITY STATISTICS\n")
            f.write("-" * 42 + "\n")
            f.write(f"AHI: {df['AHI'].mean():.2f} ± {df['AHI'].std():.2f} (range: {df['AHI'].min():.1f}-{df['AHI'].max():.1f})\n")
            f.write(f"Log(AHI+1): {df['log_AHI'].mean():.3f} ± {df['log_AHI'].std():.3f}\n")
            f.write(f"Anterior spindle density: {df['anterior_spindle_density'].mean():.3f} ± {df['anterior_spindle_density'].std():.3f} spindles/min\n")
            f.write(f"  Range: {df['anterior_spindle_density'].min():.3f} - {df['anterior_spindle_density'].max():.3f} spindles/min\n")
            f.write(f"Posterior spindle density: {df['posterior_spindle_density'].mean():.3f} ± {df['posterior_spindle_density'].std():.3f} spindles/min\n")
            f.write(f"  Range: {df['posterior_spindle_density'].min():.3f} - {df['posterior_spindle_density'].max():.3f} spindles/min\n\n")
            
            f.write("STATISTICAL ANALYSIS METHODS\n")
            f.write("-" * 28 + "\n\n")
            
            f.write("1. CORRELATION ANALYSIS\n")
            f.write("   Library: scipy.stats\n")
            f.write("   Function: stats.pearsonr(x, y)\n")
            f.write("   Method: Pearson product-moment correlation coefficient\n")
            f.write("   Variables: log(NREM_AHI+1) vs regional spindle density\n")
            f.write("   Test statistic: r (correlation coefficient)\n")
            f.write("   Significance test: Two-tailed t-test with df = n-2\n\n")
            
            f.write("2. LINEAR REGRESSION MODELS\n")
            f.write("   Library: statsmodels.formula.api\n")
            f.write("   Function: ols(formula, data).fit()\n")
            f.write("   Method: Ordinary Least Squares (OLS) regression\n")
            f.write("   Estimation: Maximum likelihood estimation\n")
            f.write("   Standard errors: Robust (HC1) covariance matrix\n\n")
            
            f.write("   Group-Specific Models:\n")
            f.write("   Formula: 'spindle_density ~ log(NREM_AHI+1) + age + sex' (per group and region)\n")
            f.write("   Note: Analysis performed separately for Control and OSA groups\n")
            f.write("   Fallback: Simple regression without covariates if insufficient sample size\n\n")
            
            f.write("3. REGIONAL COMPARISON TEST\n")
            f.write("   Library: Custom implementation (based on Steiger, 1980)\n")
            f.write("   Function: steiger_z_test(r1, r2, n)\n")
            f.write("   Method: Test for difference between two dependent correlation coefficients\n")
            f.write("   Formula: Z = (z1 - z2) / sqrt(2/(n-3))\n")
            f.write("   Where: z1 = 0.5*ln((1+r1)/(1-r1)), z2 = 0.5*ln((1+r2)/(1-r2))\n")
            f.write("   Distribution: Standard normal (Z-distribution)\n")
            f.write("   Test type: Two-tailed\n\n")
            
            f.write("4. GROUP COMPARISON TESTS\n")
            f.write("   Library: scipy.stats\n")
            f.write("   Function: stats.ttest_ind(group1, group2)\n")
            f.write("   Method: Independent samples t-test (Welch's t-test)\n")
            f.write("   Assumptions: Independent samples, normality (robust to violations with n>30)\n")
            f.write("   Equal variances: Not assumed (Welch's correction applied)\n")
            f.write("   Test type: Two-tailed\n\n")
            
            f.write("STATISTICAL RESULTS SUMMARY - GROUP-SPECIFIC ANALYSES\n")
            f.write("-" * 50 + "\n\n")
            
            # Control Group Results
            if 'control' in results and 'insufficient_data' not in results['control']:
                f.write("CONTROL GROUP RESULTS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Sample size: n = {results['control']['n']}\n\n")
                
                f.write("Primary Correlations:\n")
                anterior_r = results['control']['anterior']['correlation']
                anterior_p = results['control']['anterior']['correlation_p']
                posterior_r = results['control']['posterior']['correlation']
                posterior_p = results['control']['posterior']['correlation_p']
                
                f.write(f"  Anterior region (log AHI vs spindle density): r = {anterior_r:.4f}, p = {anterior_p:.6f}\n")
                f.write(f"  Posterior region (log AHI vs spindle density): r = {posterior_r:.4f}, p = {posterior_p:.6f}\n\n")
                
                f.write("Regression Model Performance:\n")
                f.write(f"  Anterior model R-squared: {results['control']['anterior']['r_squared']:.4f}\n")
                f.write(f"  Posterior model R-squared: {results['control']['posterior']['r_squared']:.4f}\n")
                f.write(f"  Combined model R-squared: {results['control']['combined']['r_squared']:.4f}\n\n")
                
                # Regional comparison for control group
                steiger_z = results['control']['comparison']['steiger_z']
                steiger_p = results['control']['comparison']['steiger_p']
                if not np.isnan(steiger_z):
                    f.write(f"Regional comparison (Steiger's Z): Z = {steiger_z:.4f}, p = {steiger_p:.6f}\n")
                    interpretation = "Significant" if steiger_p < 0.05 else "Not significant"
                    f.write(f"  Interpretation: {interpretation} difference between regions in Control group\n\n")
                else:
                    f.write("Regional comparison: Insufficient data for Steiger's Z-test\n\n")
            else:
                f.write("CONTROL GROUP RESULTS:\n")
                f.write("-" * 20 + "\n")
                f.write("Insufficient data for analysis\n\n")
            
            # OSA Group Results
            if 'osa' in results and 'insufficient_data' not in results['osa']:
                f.write("OSA GROUP RESULTS:\n")
                f.write("-" * 16 + "\n")
                f.write(f"Sample size: n = {results['osa']['n']}\n\n")
                
                f.write("Primary Correlations:\n")
                anterior_r = results['osa']['anterior']['correlation']
                anterior_p = results['osa']['anterior']['correlation_p']
                posterior_r = results['osa']['posterior']['correlation']
                posterior_p = results['osa']['posterior']['correlation_p']
                
                f.write(f"  Anterior region (log AHI vs spindle density): r = {anterior_r:.4f}, p = {anterior_p:.6f}\n")
                f.write(f"  Posterior region (log AHI vs spindle density): r = {posterior_r:.4f}, p = {posterior_p:.6f}\n\n")
                
                f.write("Regression Model Performance:\n")
                f.write(f"  Anterior model R-squared: {results['osa']['anterior']['r_squared']:.4f}\n")
                f.write(f"  Posterior model R-squared: {results['osa']['posterior']['r_squared']:.4f}\n")
                f.write(f"  Combined model R-squared: {results['osa']['combined']['r_squared']:.4f}\n\n")
                
                # Regional comparison for OSA group
                steiger_z = results['osa']['comparison']['steiger_z']
                steiger_p = results['osa']['comparison']['steiger_p']
                if not np.isnan(steiger_z):
                    f.write(f"Regional comparison (Steiger's Z): Z = {steiger_z:.4f}, p = {steiger_p:.6f}\n")
                    interpretation = "Significant" if steiger_p < 0.05 else "Not significant"
                    f.write(f"  Interpretation: {interpretation} difference between regions in OSA group\n\n")
                else:
                    f.write("Regional comparison: Insufficient data for Steiger's Z-test\n\n")
            else:
                f.write("OSA GROUP RESULTS:\n")
                f.write("-" * 16 + "\n")
                f.write("Insufficient data for analysis\n\n")
            
            # Overall comparison for reference
            if 'overall' in results:
                f.write("OVERALL CORRELATIONS (for reference):\n")
                f.write("-" * 36 + "\n")
                f.write(f"  Anterior region (all subjects): r = {results['overall']['anterior_correlation']:.4f}, p = {results['overall']['anterior_correlation_p']:.6f}\n")
                f.write(f"  Posterior region (all subjects): r = {results['overall']['posterior_correlation']:.4f}, p = {results['overall']['posterior_correlation_p']:.6f}\n\n")
            
            # Group comparison
            if 'group_comparison' in results:
                f.write("GROUP COMPARISONS (Control vs OSA):\n")
                f.write("-" * 32 + "\n")
                ahi_t, ahi_p = results['group_comparison']['ahi_ttest']
                anterior_t, anterior_p = results['group_comparison']['anterior_ttest']
                posterior_t, posterior_p = results['group_comparison']['posterior_ttest']
                
                f.write(f"  NREM AHI: t = {ahi_t:.4f}, p = {ahi_p:.6f}\n")
                f.write(f"  Anterior spindle density: t = {anterior_t:.4f}, p = {anterior_p:.6f}\n")
                f.write(f"  Posterior spindle density: t = {posterior_t:.4f}, p = {posterior_p:.6f}\n\n")
            
            f.write("KEY FINDINGS:\n")
            f.write("-" * 12 + "\n")
            
            # Summarize key findings from group-specific analyses
            control_findings = []
            osa_findings = []
            
            if 'control' in results and 'insufficient_data' not in results['control']:
                if results['control']['anterior']['correlation_p'] < 0.05:
                    control_findings.append("anterior")
                if results['control']['posterior']['correlation_p'] < 0.05:
                    control_findings.append("posterior")
            
            if 'osa' in results and 'insufficient_data' not in results['osa']:
                if results['osa']['anterior']['correlation_p'] < 0.05:
                    osa_findings.append("anterior")
                if results['osa']['posterior']['correlation_p'] < 0.05:
                    osa_findings.append("posterior")
            
            f.write("** GROUP-SPECIFIC ANALYSIS RESULTS **\n\n")
            
            if control_findings:
                f.write(f"CONTROL GROUP: Significant associations in {', '.join(control_findings)} region(s)\n")
            else:
                f.write("CONTROL GROUP: No significant associations found\n")
                
            if osa_findings:
                f.write(f"OSA GROUP: Significant associations in {', '.join(osa_findings)} region(s)\n")
            else:
                f.write("OSA GROUP: No significant associations found\n")
            
            f.write(f"\nNegative correlations indicate that higher NREM AHI is associated with\n")
            f.write(f"lower spindle density in that brain region.\n\n")
            
            f.write("SOFTWARE VERSIONS AND TECHNICAL NOTES:\n")
            f.write("-" * 38 + "\n")
            f.write("Analysis performed using:\n")
            f.write(f"  - Python: {sys.version.split()[0]}\n")
            f.write(f"  - pandas: {pd.__version__}\n")
            f.write(f"  - numpy: {np.__version__}\n")
            try:
                import scipy
                f.write(f"  - scipy: {scipy.__version__}\n")
            except:
                f.write("  - scipy: version not available\n")
            try:
                import statsmodels
                f.write(f"  - statsmodels: {statsmodels.__version__}\n")
            except:
                f.write("  - statsmodels: version not available\n")
            
            f.write(f"\nData processing notes:\n")
            f.write(f"  - AHI source: NREM AHI (nrem_ahi) used instead of overall AHI\n")
            f.write(f"  - AHI transformation: log(NREM_AHI + 1) to handle zero values and non-linearity\n")
            f.write(f"  - Age calculation: age_years + 1 if age_months >= 6, otherwise age_years\n")
            f.write(f"  - Group-specific analysis: Separate correlations/regressions for Control vs OSA groups\n")
            f.write(f"  - Spindle detection: All spindles 0.5-3.0 seconds duration\n")
            f.write(f"  - Channel classification: Anterior vs posterior based on reference/segment_net.txt\n")
            f.write(f"  - Missing data handling: Complete case analysis (listwise deletion)\n")
            f.write(f"  - Multiple comparisons: Not corrected (exploratory analysis)\n")
            f.write(f"  - Significance level: α = 0.05 (two-tailed tests)\n")
        
        print(f"✓ Comprehensive technical summary saved to {summary_file}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("PEDS OSA Regional AHI Analysis - Complete Pipeline")
        print("=" * 60)
        
        # Load channel definitions
        self.load_channel_definitions()
        
        # Load and prepare data
        anterior_df, posterior_df, demo_df = self.load_and_prepare_data()
        
        # Create analysis dataset
        df = self.create_analysis_dataset(anterior_df, posterior_df, demo_df)
        
        # Perform statistical analysis
        results = self.perform_statistical_analysis(df)
        
        # Create figures
        self.create_main_results_figure(df, results)
        self.create_demographics_dashboard(df)
        
        # Save comprehensive summary
        self.save_results_summary(df, results)
        
        print(f"\n✓ COMPLETE ANALYSIS FINISHED!")
        print(f"✓ Results saved to: {self.output_dir}")
        print(f"✓ Generated files:")
        print(f"  - main_results_figure.png")
        print(f"  - demographics_dashboard.png")
        print(f"  - analysis_summary.txt")

def main(project_dir):
    """Main function to run the regional AHI analysis."""
    try:
        analysis = RegionalAHIAnalysis(project_dir)
        analysis.run_complete_analysis()
        return True
    except Exception as e:
        print(f"Error in regional AHI analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # For testing
    import sys
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
    else:
        project_dir = "/Volumes/Ido/002_PEDS_OSA"
    
    main(project_dir)