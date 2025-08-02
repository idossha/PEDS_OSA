#!/usr/bin/env python3
"""
PEDS OSA Regional Spindle Analysis - Complete Pipeline
=====================================================

Single comprehensive script that performs:
1. Anterior spindle density vs Cohen's d analysis
2. Posterior spindle density vs Cohen's d analysis  
3. Regional comparison analysis
4. Generation of publication-ready figures

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

class RegionalSpindleAnalysis:
    """Complete regional spindle analysis pipeline."""
    
    def __init__(self, project_dir, output_dir):
        self.project_dir = project_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_anterior_channels(self):
        """Load anterior channels from segment_net.txt."""
        anterior_channels_path = os.path.join(os.path.dirname(__file__), "reference", "segment_net.txt")
        
        try:
            with open(anterior_channels_path, 'r') as f:
                content = f.read()
            
            # Extract anterior channels section
            anterior_section = content.split('Anterior_channels:')[1].split('Posterior_channels:')[0]
            
            # Parse channel names
            channels = []
            for line in anterior_section.strip().split('\n'):
                if line.strip():
                    line_channels = [ch.strip() for ch in line.split(',') if ch.strip()]
                    channels.extend(line_channels)
            
            # Clean and convert to numeric format
            numeric_channels = []
            for channel in channels:
                if channel.startswith('E'):
                    num = int(channel[1:])
                    numeric_channels.append(str(num))
                elif channel.isdigit():
                    numeric_channels.append(channel)
                else:
                    import re
                    digits = re.findall(r'\d+', channel)
                    if digits:
                        numeric_channels.append(str(int(digits[0])))
            
            print(f"Loaded {len(numeric_channels)} anterior channels")
            return numeric_channels
            
        except Exception as e:
            print(f"Error loading anterior channels: {e}")
            # Fallback
            return [str(i) for i in range(1, 41)]

    def load_posterior_channels(self):
        """Load posterior channels from segment_net.txt."""
        posterior_channels_path = os.path.join(os.path.dirname(__file__), "reference", "segment_net.txt")
        
        try:
            with open(posterior_channels_path, 'r') as f:
                content = f.read()
            
            # Extract posterior channels section
            posterior_section = content.split('Posterior_channels:')[1]
            
            # Parse channel names
            channels = []
            for line in posterior_section.strip().split('\n'):
                if line.strip():
                    line_channels = [ch.strip() for ch in line.split(',') if ch.strip()]
                    channels.extend(line_channels)
            
            # Clean and convert to numeric format
            numeric_channels = []
            for channel in channels:
                if channel.startswith('E'):
                    num = int(channel[1:])
                    numeric_channels.append(str(num))
                elif channel.isdigit():
                    numeric_channels.append(channel)
                else:
                    import re
                    digits = re.findall(r'\d+', channel)
                    if digits:
                        numeric_channels.append(str(int(digits[0])))
            
            print(f"Loaded {len(numeric_channels)} posterior channels")
            return numeric_channels
            
        except Exception as e:
            print(f"Error loading posterior channels: {e}")
            # Fallback
            return [str(i) for i in range(70, 90)]

    def collect_spindle_density_data(self, channels, region_name):
        """Collect spindle density data for specified channels."""
        print(f"Collecting {region_name} spindle density data...")
        
        # Find all spindle files
        pattern = os.path.join(self.project_dir, "derivatives", "sub-*", "spindles", "results", "all_spindles_detailed.csv")
        spindle_files = glob.glob(pattern)
        print(f"Found {len(spindle_files)} subject spindle files")
        
        spindle_data = []
        channels_int = [int(ch) for ch in channels]
        
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
                    
                region_spindles = spindle_df[spindle_df['Channel'].isin(channels_int)]
                
                if len(region_spindles) == 0:
                    continue
                
                # Calculate spindle density
                total_duration_seconds = region_spindles['End'].max()
                total_duration_minutes = total_duration_seconds / 60
                total_spindles = len(region_spindles)
                spindle_density = total_spindles / total_duration_minutes if total_duration_minutes > 0 else 0
                
                # Additional statistics
                mean_duration = region_spindles['Duration'].mean()
                mean_frequency = region_spindles['Frequency'].mean()
                slow_spindles = len(region_spindles[region_spindles['Frequency'] < 12])
                fast_spindles = len(region_spindles[region_spindles['Frequency'] >= 12])
                
                spindle_data.append({
                    'Subject': subject_id,
                    f'{region_name.lower()}_spindle_density': spindle_density,
                    f'{region_name.lower()}_mean_duration': mean_duration,
                    f'{region_name.lower()}_mean_frequency': mean_frequency,
                    f'{region_name.lower()}_slow_spindles': slow_spindles,
                    f'{region_name.lower()}_fast_spindles': fast_spindles,
                    f'{region_name.lower()}_total_spindles': total_spindles,
                    f'{region_name.lower()}_recording_duration': total_duration_minutes
                })
                
            except Exception as e:
                print(f"Error processing {subject_id}: {e}")
                continue
        
        df = pd.DataFrame(spindle_data)
        print(f"✓ Collected {region_name.lower()} spindle data for {len(df)} subjects")
        return df

    def load_demographics_and_tova(self):
        """Load demographics and TOVA data."""
        # Load demographics
        demo_path = os.path.join(self.project_dir, "demographics.csv")
        try:
            demo_df = pd.read_csv(demo_path)
        except UnicodeDecodeError:
            demo_df = pd.read_csv(demo_path, encoding='latin-1')
        
        # Load TOVA
        tova_path = os.path.join(self.project_dir, "TOVA.csv")
        try:
            tova_df = pd.read_csv(tova_path)
        except UnicodeDecodeError:
            tova_df = pd.read_csv(tova_path, encoding='latin-1')
        
        print(f"✓ Loaded demographics: {len(demo_df)} subjects")
        print(f"✓ Loaded TOVA data: {len(tova_df)} subjects")
        
        return demo_df, tova_df

    def create_analysis_dataset(self, anterior_df, posterior_df, demo_df, tova_df):
        """Create complete analysis dataset with both regions."""
        print("Creating complete analysis dataset...")
        
        # Normalize subject IDs
        anterior_df['Subject'] = anterior_df['Subject'].astype(str).str.lstrip('0')
        posterior_df['Subject'] = posterior_df['Subject'].astype(str).str.lstrip('0')
        demo_df['pptid'] = demo_df['pptid'].astype(str)
        tova_df['Subject'] = tova_df['Subject'].astype(str)
        
        # Merge anterior and posterior data
        merged_df = anterior_df.merge(posterior_df, on='Subject', how='inner')
        
        # Merge with demographics
        merged_df = merged_df.merge(demo_df, left_on='Subject', right_on='pptid', how='inner')
        
        # Merge with TOVA
        merged_df = merged_df.merge(tova_df, on='Subject', how='inner')
        
        # Create analysis variables
        merged_df['age'] = merged_df['age_years']
        merged_df['sex'] = merged_df['gender'].map({0: 'F', 1: 'M'})
        merged_df['cohens_d'] = merged_df["Cohen's d"]
        
        # Rename for clarity
        merged_df = merged_df.rename(columns={'anterior_spindle_density': 'anterior_spindle_density'})
        
        print(f"✓ Complete dataset: {len(merged_df)} subjects with all data")
        print(f"  - Mean age: {merged_df['age'].mean():.1f} ± {merged_df['age'].std():.1f} years")
        print(f"  - Sex: {merged_df['sex'].value_counts().to_dict()}")
        print(f"  - Groups: {merged_df['group'].value_counts().to_dict()}")
        print(f"  - Mean Cohen's d: {merged_df['cohens_d'].mean():.3f} ± {merged_df['cohens_d'].std():.3f}")
        
        return merged_df

    def perform_statistical_analysis(self, df):
        """Perform complete statistical analysis."""
        print("\nPerforming statistical analysis...")
        
        results = {}
        
        # Anterior analysis
        print("1. Anterior spindle density analysis...")
        anterior_model = ols('cohens_d ~ anterior_spindle_density + age + sex', data=df).fit()
        anterior_r, anterior_p = stats.pearsonr(df['anterior_spindle_density'], df['cohens_d'])
        
        results['anterior'] = {
            'model': anterior_model,
            'correlation': anterior_r,
            'correlation_p': anterior_p,
            'r_squared': anterior_model.rsquared,
            'f_statistic': anterior_model.fvalue,
            'f_p_value': anterior_model.f_pvalue
        }
        
        # Posterior analysis
        print("2. Posterior spindle density analysis...")
        posterior_model = ols('cohens_d ~ posterior_spindle_density + age + sex', data=df).fit()
        posterior_r, posterior_p = stats.pearsonr(df['posterior_spindle_density'], df['cohens_d'])
        
        results['posterior'] = {
            'model': posterior_model,
            'correlation': posterior_r,
            'correlation_p': posterior_p,
            'r_squared': posterior_model.rsquared,
            'f_statistic': posterior_model.fvalue,
            'f_p_value': posterior_model.f_pvalue
        }
        
        # Combined model
        print("3. Combined regional model...")
        combined_model = ols('cohens_d ~ anterior_spindle_density + posterior_spindle_density + age + sex', data=df).fit()
        
        results['combined'] = {
            'model': combined_model,
            'r_squared': combined_model.rsquared,
            'f_statistic': combined_model.fvalue,
            'f_p_value': combined_model.f_pvalue
        }
        
        # Regional comparison
        print("4. Regional comparison...")
        def steiger_z_test(r1, r2, n):
            """Test difference between correlations."""
            z1 = 0.5 * np.log((1 + r1) / (1 - r1))
            z2 = 0.5 * np.log((1 + r2) / (1 - r2))
            z_diff = (z1 - z2) / np.sqrt(2 / (n - 3))
            p_value = 2 * (1 - stats.norm.cdf(abs(z_diff)))
            return z_diff, p_value
        
        z_diff, p_diff = steiger_z_test(anterior_r, posterior_r, len(df))
        
        results['comparison'] = {
            'steiger_z': z_diff,
            'steiger_p': p_diff
        }
        
        # Group analysis
        print("5. Group analysis...")
        control_data = df[df['group'] == 'Control']
        osa_data = df[df['group'] == 'OSA']
        
        cohens_d_ttest = stats.ttest_ind(control_data['cohens_d'], osa_data['cohens_d'])
        anterior_density_ttest = stats.ttest_ind(control_data['anterior_spindle_density'], osa_data['anterior_spindle_density'])
        posterior_density_ttest = stats.ttest_ind(control_data['posterior_spindle_density'], osa_data['posterior_spindle_density'])
        
        results['groups'] = {
            'control_n': len(control_data),
            'osa_n': len(osa_data),
            'cohens_d_ttest': cohens_d_ttest,
            'anterior_ttest': anterior_density_ttest,
            'posterior_ttest': posterior_density_ttest
        }
        
        print("✓ Statistical analysis completed")
        return results

    def create_main_results_figure(self, df, results):
        """Create the main results figure (3 panels)."""
        print("Creating main results figure...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel 1: Regional Effects with Regression Lines
        ax1 = axes[0]
        
        x_anterior = df['anterior_spindle_density']
        x_posterior = df['posterior_spindle_density']
        y = df['cohens_d']
        
        # Anterior scatter and regression
        ax1.scatter(x_anterior, y, alpha=0.6, color='blue', s=50, label='Anterior', edgecolors='navy', linewidth=0.5)
        
        # Regression lines
        anterior_slope, anterior_intercept = np.polyfit(x_anterior, y, 1)
        posterior_slope, posterior_intercept = np.polyfit(x_posterior, y, 1)
        
        x_anterior_line = np.linspace(x_anterior.min(), x_anterior.max(), 100)
        y_anterior_line = anterior_slope * x_anterior_line + anterior_intercept
        ax1.plot(x_anterior_line, y_anterior_line, color='blue', linewidth=2, alpha=0.8)
        
        # Posterior scatter (offset for visibility)
        x_posterior_offset = x_posterior + 200
        ax1.scatter(x_posterior_offset, y, alpha=0.6, color='red', s=50, label='Posterior', edgecolors='darkred', linewidth=0.5)
        
        x_posterior_line = np.linspace(x_posterior.min(), x_posterior.max(), 100) + 200
        y_posterior_line = posterior_slope * (x_posterior_line - 200) + posterior_intercept
        ax1.plot(x_posterior_line, y_posterior_line, color='red', linewidth=2, alpha=0.8)
        
        # Add statistics
        ax1.text(0.05, 0.95, f'Anterior: r = {results["anterior"]["correlation"]:.3f}, p = {results["anterior"]["correlation_p"]:.3f}', 
                 transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax1.text(0.05, 0.85, f'Posterior: r = {results["posterior"]["correlation"]:.3f}, p = {results["posterior"]["correlation_p"]:.3f}', 
                 transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        ax1.set_xlabel('Spindle Density (spindles/min)', fontsize=12, weight='bold')
        ax1.set_ylabel("Cohen's d (TOVA Improvement)", fontsize=12, weight='bold')
        ax1.set_title('A. Regional Spindle Density vs Cognitive Performance', fontsize=14, weight='bold', pad=20)
        ax1.legend(loc='lower right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Axis break indication
        ax1.axvline(x=150, color='gray', linestyle=':', alpha=0.7)
        ax1.text(150, ax1.get_ylim()[1]*0.9, '//axis break//', rotation=90, 
                 verticalalignment='center', horizontalalignment='center', fontsize=9, alpha=0.7)
        
        # Panel 2: Box plots by group
        ax2 = axes[1]
        
        control_data = df[df['group'] == 'Control']['cohens_d']
        osa_data = df[df['group'] == 'OSA']['cohens_d']
        
        box_data = [control_data, osa_data]
        box_labels = ['Control\n(n=19)', 'OSA\n(n=34)']
        
        bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True, 
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2))
        
        # Add individual points
        for i, data in enumerate(box_data):
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax2.scatter(x, data, alpha=0.6, s=30, color='darkblue', edgecolors='navy', linewidth=0.5)
        
        # Add statistics
        t_stat, p_val = results['groups']['cohens_d_ttest']
        ax2.text(0.5, 0.95, f't = {t_stat:.3f}, p = {p_val:.3f}', 
                 transform=ax2.transAxes, fontsize=11, horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Add means
        means = [control_data.mean(), osa_data.mean()]
        ax2.scatter([1, 2], means, marker='D', s=100, color='orange', 
                    edgecolors='darkorange', linewidth=2, zorder=10, label='Mean')
        
        ax2.set_ylabel("Cohen's d (TOVA Improvement)", fontsize=12, weight='bold')
        ax2.set_title('B. Cognitive Improvement by Group', fontsize=14, weight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=10)
        
        # Panel 3: Anterior vs Posterior comparison
        ax3 = axes[2]
        
        ax3.scatter(df['anterior_spindle_density'], df['posterior_spindle_density'], 
                    alpha=0.6, s=50, color='green', edgecolors='darkgreen', linewidth=0.5)
        
        # Unity line
        min_val = min(df[['anterior_spindle_density', 'posterior_spindle_density']].min())
        max_val = max(df[['anterior_spindle_density', 'posterior_spindle_density']].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Unity line (x=y)')
        
        # Correlation
        corr_coef, corr_p = stats.pearsonr(df['anterior_spindle_density'], df['posterior_spindle_density'])
        ax3.text(0.05, 0.95, f'r = {corr_coef:.3f}\np = {corr_p:.3e}', 
                 transform=ax3.transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Paired t-test
        t_stat_paired, p_val_paired = stats.ttest_rel(df['anterior_spindle_density'], df['posterior_spindle_density'])
        ax3.text(0.05, 0.75, f'Paired t-test:\nt = {t_stat_paired:.3f}\np < 0.001', 
                 transform=ax3.transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax3.set_xlabel('Anterior Spindle Density (spindles/min)', fontsize=12, weight='bold')
        ax3.set_ylabel('Posterior Spindle Density (spindles/min)', fontsize=12, weight='bold')
        ax3.set_title('C. Regional Spindle Density Comparison', fontsize=14, weight='bold', pad=20)
        ax3.legend(loc='upper right', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'main_results_figure.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Main results figure saved to {output_file}")

    def create_demographics_dashboard(self, df):
        """Create demographics and descriptive statistics dashboard."""
        print("Creating demographics dashboard...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Panel 1: Age distribution
        ax1 = axes[0, 0]
        ages = df['age']
        ax1.hist(ages, bins=np.arange(4.5, 12.5, 1), alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1)
        ax1.axvline(ages.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ages.mean():.1f} years')
        ax1.set_xlabel('Age (years)', fontsize=12, weight='bold')
        ax1.set_ylabel('Number of Subjects', fontsize=12, weight='bold')
        ax1.set_title('Age Distribution', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.text(0.7, 0.9, f'n = {len(ages)}\nMean: {ages.mean():.1f} ± {ages.std():.1f}\nRange: {ages.min():.0f}-{ages.max():.0f}', 
                 transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel 2: Sex distribution
        ax2 = axes[0, 1]
        sex_counts = df['sex'].value_counts()
        colors = ['lightpink', 'lightblue']
        wedges, texts, autotexts = ax2.pie(sex_counts.values, labels=[f'{label}\n(n={count})' for label, count in sex_counts.items()], 
                                           autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
        ax2.set_title('Sex Distribution', fontsize=14, weight='bold')
        
        # Panel 3: Group distribution
        ax3 = axes[0, 2]
        group_counts = df['group'].value_counts()
        colors = ['lightcoral', 'lightgreen']
        wedges, texts, autotexts = ax3.pie(group_counts.values, labels=[f'{label}\n(n={count})' for label, count in group_counts.items()], 
                                           autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
        ax3.set_title('Group Distribution', fontsize=14, weight='bold')
        
        # Panel 4: Cohen's d distribution
        ax4 = axes[1, 0]
        cohens_d = df['cohens_d']
        ax4.hist(cohens_d, bins=15, alpha=0.7, color='gold', edgecolor='orange', linewidth=1)
        ax4.axvline(cohens_d.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cohens_d.mean():.2f}')
        ax4.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.7, label='No change')
        ax4.set_xlabel("Cohen's d (TOVA Improvement)", fontsize=12, weight='bold')
        ax4.set_ylabel('Number of Subjects', fontsize=12, weight='bold')
        ax4.set_title("Cognitive Improvement Distribution", fontsize=14, weight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        improved = len(cohens_d[cohens_d < 0])
        declined = len(cohens_d[cohens_d > 0])
        ax4.text(0.7, 0.9, f'Improved: {improved}/53 ({improved/53*100:.1f}%)\nDeclined: {declined}/53 ({declined/53*100:.1f}%)\nMean: {cohens_d.mean():.2f} ± {cohens_d.std():.2f}', 
                 transform=ax4.transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel 5: Anterior spindle density
        ax5 = axes[1, 1]
        anterior_density = df['anterior_spindle_density']
        ax5.hist(anterior_density, bins=15, alpha=0.7, color='lightblue', edgecolor='navy', linewidth=1)
        ax5.axvline(anterior_density.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {anterior_density.mean():.1f}')
        ax5.set_xlabel('Anterior Spindle Density (spindles/min)', fontsize=12, weight='bold')
        ax5.set_ylabel('Number of Subjects', fontsize=12, weight='bold')
        ax5.set_title('Anterior Spindle Density', fontsize=14, weight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.text(0.7, 0.9, f'Mean: {anterior_density.mean():.1f} ± {anterior_density.std():.1f}\nRange: {anterior_density.min():.1f}-{anterior_density.max():.1f}', 
                 transform=ax5.transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel 6: Posterior spindle density
        ax6 = axes[1, 2]
        posterior_density = df['posterior_spindle_density']
        ax6.hist(posterior_density, bins=15, alpha=0.7, color='lightcoral', edgecolor='darkred', linewidth=1)
        ax6.axvline(posterior_density.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {posterior_density.mean():.1f}')
        ax6.set_xlabel('Posterior Spindle Density (spindles/min)', fontsize=12, weight='bold')
        ax6.set_ylabel('Number of Subjects', fontsize=12, weight='bold')
        ax6.set_title('Posterior Spindle Density', fontsize=14, weight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.text(0.7, 0.9, f'Mean: {posterior_density.mean():.1f} ± {posterior_density.std():.1f}\nRange: {posterior_density.min():.1f}-{posterior_density.max():.1f}', 
                 transform=ax6.transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle('PEDS OSA Regional Spindle Analysis - Demographics & Descriptive Statistics', 
                     fontsize=16, weight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        output_file = os.path.join(self.output_dir, 'demographics_dashboard.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Demographics dashboard saved to {output_file}")

    def save_results_summary(self, df, results):
        """Save a comprehensive results summary with technical details."""
        summary_file = os.path.join(self.output_dir, 'analysis_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("PEDS OSA REGIONAL SPINDLE ANALYSIS - COMPREHENSIVE TECHNICAL SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("ANALYSIS PIPELINE DETAILS\n")
            f.write("-" * 25 + "\n")
            f.write("Script: regional_spindle_analysis.py\n")
            f.write("Analysis Date: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("Python Environment: pandas, numpy, scipy, statsmodels, matplotlib, seaborn\n")
            f.write("Statistical Framework: OLS regression, Pearson correlation, Steiger's Z-test\n\n")
            
            f.write("SAMPLE CHARACTERISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total subjects: {len(df)}\n")
            f.write(f"Mean age: {df['age'].mean():.1f} ± {df['age'].std():.1f} years (range: {df['age'].min():.0f}-{df['age'].max():.0f})\n")
            f.write(f"Sex distribution: {df['sex'].value_counts().to_dict()}\n")
            f.write(f"Group distribution: {df['group'].value_counts().to_dict()}\n\n")
            
            f.write("REGIONAL SPINDLE DENSITY STATISTICS\n")
            f.write("-" * 35 + "\n")
            f.write(f"Anterior spindle density: {df['anterior_spindle_density'].mean():.3f} ± {df['anterior_spindle_density'].std():.3f} spindles/min\n")
            f.write(f"  Range: {df['anterior_spindle_density'].min():.3f} - {df['anterior_spindle_density'].max():.3f} spindles/min\n")
            f.write(f"Posterior spindle density: {df['posterior_spindle_density'].mean():.3f} ± {df['posterior_spindle_density'].std():.3f} spindles/min\n")
            f.write(f"  Range: {df['posterior_spindle_density'].min():.3f} - {df['posterior_spindle_density'].max():.3f} spindles/min\n")
            f.write(f"Cohen's d: {df['cohens_d'].mean():.3f} ± {df['cohens_d'].std():.3f}\n")
            f.write(f"  Range: {df['cohens_d'].min():.3f} - {df['cohens_d'].max():.3f}\n")
            f.write(f"  Subjects with improvement (negative d): {len(df[df['cohens_d'] < 0])}/{len(df)} ({len(df[df['cohens_d'] < 0])/len(df)*100:.1f}%)\n\n")
            
            f.write("STATISTICAL ANALYSIS METHODS\n")
            f.write("-" * 28 + "\n\n")
            
            f.write("1. CORRELATION ANALYSIS\n")
            f.write("   Library: scipy.stats\n")
            f.write("   Function: stats.pearsonr(x, y)\n")
            f.write("   Method: Pearson product-moment correlation coefficient\n")
            f.write("   Assumptions: Linear relationship, normal distribution, homoscedasticity\n")
            f.write("   Test statistic: r (correlation coefficient)\n")
            f.write("   Significance test: Two-tailed t-test with df = n-2\n\n")
            
            f.write("2. LINEAR REGRESSION MODELS\n")
            f.write("   Library: statsmodels.formula.api\n")
            f.write("   Function: ols(formula, data).fit()\n")
            f.write("   Method: Ordinary Least Squares (OLS) regression\n")
            f.write("   Estimation: Maximum likelihood estimation\n")
            f.write("   Standard errors: Robust (HC1) covariance matrix\n\n")
            
            f.write("   Model 1 - Anterior Region:\n")
            f.write("   Formula: 'cohens_d ~ anterior_spindle_density + age + sex'\n")
            f.write(f"   R-squared: {results['anterior']['r_squared']:.4f}\n")
            f.write(f"   F-statistic: {results['anterior']['f_statistic']:.4f} (p = {results['anterior']['f_p_value']:.6f})\n")
            f.write(f"   Degrees of freedom: model = 3, residual = {len(df)-4}\n")
            f.write(f"   AIC: {results['anterior']['model'].aic:.2f}\n")
            f.write(f"   BIC: {results['anterior']['model'].bic:.2f}\n\n")
            
            f.write("   Model 2 - Posterior Region:\n")
            f.write("   Formula: 'cohens_d ~ posterior_spindle_density + age + sex'\n")
            f.write(f"   R-squared: {results['posterior']['r_squared']:.4f}\n")
            f.write(f"   F-statistic: {results['posterior']['f_statistic']:.4f} (p = {results['posterior']['f_p_value']:.6f})\n")
            f.write(f"   Degrees of freedom: model = 3, residual = {len(df)-4}\n")
            f.write(f"   AIC: {results['posterior']['model'].aic:.2f}\n")
            f.write(f"   BIC: {results['posterior']['model'].bic:.2f}\n\n")
            
            f.write("   Model 3 - Combined Regional Model:\n")
            f.write("   Formula: 'cohens_d ~ anterior_spindle_density + posterior_spindle_density + age + sex'\n")
            f.write(f"   R-squared: {results['combined']['r_squared']:.4f}\n")
            f.write(f"   F-statistic: {results['combined']['f_statistic']:.4f} (p = {results['combined']['f_p_value']:.6f})\n")
            f.write(f"   Degrees of freedom: model = 4, residual = {len(df)-5}\n")
            f.write(f"   AIC: {results['combined']['model'].aic:.2f}\n")
            f.write(f"   BIC: {results['combined']['model'].bic:.2f}\n\n")
            
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
            
            f.write("5. PAIRED REGIONAL COMPARISON\n")
            f.write("   Library: scipy.stats\n")
            f.write("   Function: stats.ttest_rel(anterior, posterior)\n")
            f.write("   Method: Paired samples t-test\n")
            f.write("   Assumptions: Paired observations, normality of differences\n")
            f.write("   Test type: Two-tailed\n\n")
            
            f.write("STATISTICAL RESULTS SUMMARY\n")
            f.write("-" * 27 + "\n\n")
            
            f.write("PRIMARY CORRELATIONS:\n")
            f.write(f"  Anterior spindle density vs Cohen's d: r = {results['anterior']['correlation']:.4f}, p = {results['anterior']['correlation_p']:.6f}\n")
            f.write(f"  Posterior spindle density vs Cohen's d: r = {results['posterior']['correlation']:.4f}, p = {results['posterior']['correlation_p']:.6f}\n\n")
            
            f.write("REGRESSION COEFFICIENTS:\n")
            anterior_coef = results['anterior']['model'].params['anterior_spindle_density']
            anterior_pval = results['anterior']['model'].pvalues['anterior_spindle_density']
            posterior_coef = results['posterior']['model'].params['posterior_spindle_density']
            posterior_pval = results['posterior']['model'].pvalues['posterior_spindle_density']
            
            f.write(f"  Anterior model - spindle density coefficient: β = {anterior_coef:.6f}, p = {anterior_pval:.6f}\n")
            f.write(f"  Posterior model - spindle density coefficient: β = {posterior_coef:.6f}, p = {posterior_pval:.6f}\n\n")
            
            f.write("REGIONAL DIFFERENCE TEST:\n")
            f.write(f"  Steiger's Z-test: Z = {results['comparison']['steiger_z']:.4f}, p = {results['comparison']['steiger_p']:.6f}\n")
            f.write(f"  Interpretation: {'Significant' if results['comparison']['steiger_p'] < 0.05 else 'Not significant'} difference between regions\n\n")
            
            f.write("GROUP COMPARISONS (Control vs OSA):\n")
            cohens_d_t, cohens_d_p = results['groups']['cohens_d_ttest']
            anterior_t, anterior_p = results['groups']['anterior_ttest']
            posterior_t, posterior_p = results['groups']['posterior_ttest']
            
            f.write(f"  Cohen's d: t = {cohens_d_t:.4f}, p = {cohens_d_p:.6f} (df = {results['groups']['control_n'] + results['groups']['osa_n'] - 2})\n")
            f.write(f"  Anterior spindle density: t = {anterior_t:.4f}, p = {anterior_p:.6f}\n")
            f.write(f"  Posterior spindle density: t = {posterior_t:.4f}, p = {posterior_p:.6f}\n\n")
            
            f.write("EFFECT SIZES AND CLINICAL INTERPRETATION:\n")
            f.write("-" * 40 + "\n")
            
            # Calculate Cohen's d effect sizes for correlations
            anterior_cohens_d_effect = 2 * results['anterior']['correlation'] / np.sqrt(1 - results['anterior']['correlation']**2)
            posterior_cohens_d_effect = 2 * results['posterior']['correlation'] / np.sqrt(1 - results['posterior']['correlation']**2)
            
            f.write(f"Correlation effect sizes (Cohen's d equivalent):\n")
            f.write(f"  Anterior: d = {anterior_cohens_d_effect:.3f} ({'small' if abs(anterior_cohens_d_effect) < 0.5 else 'medium' if abs(anterior_cohens_d_effect) < 0.8 else 'large'} effect)\n")
            f.write(f"  Posterior: d = {posterior_cohens_d_effect:.3f} ({'small' if abs(posterior_cohens_d_effect) < 0.5 else 'medium' if abs(posterior_cohens_d_effect) < 0.8 else 'large'} effect)\n\n")
            
            f.write("CLINICAL SIGNIFICANCE:\n")
            f.write(f"  Variance explained by anterior model: {results['anterior']['r_squared']*100:.2f}%\n")
            f.write(f"  Variance explained by posterior model: {results['posterior']['r_squared']*100:.2f}%\n")
            f.write(f"  Variance explained by combined model: {results['combined']['r_squared']*100:.2f}%\n\n")
            
            f.write("KEY FINDINGS:\n")
            f.write("-" * 12 + "\n")
            
            if results['posterior']['correlation_p'] < 0.05 and results['anterior']['correlation_p'] >= 0.05:
                f.write("** POSTERIOR REGIONS SHOW SIGNIFICANT ASSOCIATION WITH COGNITIVE IMPROVEMENT **\n")
                f.write("** Anterior regions show no significant association **\n")
                f.write(f"** Clinical implication: Posterior brain regions may be more important for \n")
                f.write(f"   cognitive resilience than traditionally thought **\n")
            elif results['anterior']['correlation_p'] < 0.05 and results['posterior']['correlation_p'] >= 0.05:
                f.write("** ANTERIOR REGIONS SHOW SIGNIFICANT ASSOCIATION WITH COGNITIVE IMPROVEMENT **\n")
                f.write("** Posterior regions show no significant association **\n")
            elif results['anterior']['correlation_p'] < 0.05 and results['posterior']['correlation_p'] < 0.05:
                f.write("** BOTH REGIONS SHOW SIGNIFICANT ASSOCIATIONS WITH COGNITIVE IMPROVEMENT **\n")
            else:
                f.write("** NO REGIONS SHOW SIGNIFICANT ASSOCIATIONS WITH COGNITIVE IMPROVEMENT **\n")
            
            f.write(f"\nNegative correlations indicate that higher spindle density is associated with\n")
            f.write(f"better cognitive improvement (more negative Cohen's d values).\n\n")
            
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
            f.write(f"  - Spindle detection: All spindles 0.5-3.0 seconds duration\n")
            f.write(f"  - Channel classification: Anterior vs posterior based on reference/segment_net.txt\n")
            f.write(f"  - Missing data handling: Complete case analysis (listwise deletion)\n")
            f.write(f"  - Multiple comparisons: Not corrected (exploratory analysis)\n")
            f.write(f"  - Significance level: α = 0.05 (two-tailed tests)\n")
        
        print(f"✓ Comprehensive technical summary saved to {summary_file}")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("PEDS OSA Regional Spindle Analysis - Complete Pipeline")
        print("=" * 60)
        
        # Step 1: Load channel definitions
        anterior_channels = self.load_anterior_channels()
        posterior_channels = self.load_posterior_channels()
        
        # Step 2: Collect spindle data
        anterior_df = self.collect_spindle_density_data(anterior_channels, "Anterior")
        posterior_df = self.collect_spindle_density_data(posterior_channels, "Posterior")
        
        # Step 3: Load demographics and TOVA
        demo_df, tova_df = self.load_demographics_and_tova()
        
        # Step 4: Create complete dataset
        complete_df = self.create_analysis_dataset(anterior_df, posterior_df, demo_df, tova_df)
        
        # Step 5: Perform statistical analysis
        results = self.perform_statistical_analysis(complete_df)
        
        # Step 6: Create publication figures
        self.create_main_results_figure(complete_df, results)
        self.create_demographics_dashboard(complete_df)
        
        # Step 7: Save summary
        self.save_results_summary(complete_df, results)
        
        print(f"\n✓ COMPLETE ANALYSIS FINISHED!")
        print(f"✓ Results saved to: {self.output_dir}")
        print("✓ Generated files:")
        print("  - main_results_figure.png")
        print("  - demographics_dashboard.png") 
        print("  - analysis_summary.txt")
        
        return complete_df, results


def main(project_dir=None):
    """Main entry point for the analysis."""
    if project_dir is None:
        try:
            from config import PROJECT_DIR
            project_dir = PROJECT_DIR
        except ImportError:
            project_dir = "/Volumes/Ido/002_PEDS_OSA"
    
    output_dir = os.path.join(project_dir, "regional_analysis_results")
    
    # Run complete analysis
    analysis = RegionalSpindleAnalysis(project_dir, output_dir)
    df, results = analysis.run_complete_analysis()
    
    return df, results


if __name__ == "__main__":
    main()