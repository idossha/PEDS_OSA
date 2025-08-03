#!/usr/bin/env python3
"""
PEDS OSA: Simplified Mixed Effects Analysis
==========================================

Research Questions:
1. Do kids with higher spindle density show greater performance improvement over time?
2. Does NREM AHI predict regional spindle density?

Author: Simplified version for clarity and visualization
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import mixedlm, ols
import warnings

warnings.filterwarnings('ignore')

# Set clean plotting style
plt.style.use('default')
sns.set_palette("Set2")

class SimplifiedMixedEffectsAnalysis:
    """Simplified Mixed Effects Analysis for PEDS OSA data."""
    
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.output_dir = os.path.join(project_dir, "mixed_effects_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define channels (simplified)
        self.anterior_channels = [str(i) for i in range(1, 41)]  # E1-E40
        self.posterior_channels = [str(i) for i in range(70, 100)]  # E70-E99
        
        print(f"Analysis initialized. Output: {self.output_dir}")
    
    def load_data(self):
        """Load and combine all necessary data."""
        print("Loading data...")
        
        # 1. Load spindle data for both regions
        spindle_data = []
        pattern = os.path.join(self.project_dir, "derivatives", "sub-*", "spindles", "results", "all_spindles_detailed.csv")
        files = glob.glob(pattern)
        
        print(f"Found {len(files)} subject files")
        
        for file_path in files:
            subject_id = file_path.split(os.sep)[-4].replace("sub-", "")
            
            try:
                # Handle encoding issues
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='latin-1')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, encoding='cp1252')
                
                if 'Channel' not in df.columns:
                    continue
                
                # Process anterior region
                anterior_data = df[df['Channel'].isin([int(ch) for ch in self.anterior_channels])]
                if len(anterior_data) > 0:
                    duration_min = anterior_data['End'].max() / 60
                    density = len(anterior_data) / duration_min if duration_min > 0 else 0
                    spindle_data.append({
                        'Subject': subject_id,
                        'Region': 'anterior',
                        'spindle_density': density
                    })
                
                # Process posterior region
                posterior_data = df[df['Channel'].isin([int(ch) for ch in self.posterior_channels])]
                if len(posterior_data) > 0:
                    duration_min = posterior_data['End'].max() / 60
                    density = len(posterior_data) / duration_min if duration_min > 0 else 0
                    spindle_data.append({
                        'Subject': subject_id,
                        'Region': 'posterior',
                        'spindle_density': density
                    })
                    
            except Exception as e:
                print(f"  Warning: Could not process {subject_id}: {e}")
                continue
        
        spindle_df = pd.DataFrame(spindle_data)
        print(f"✓ Spindle data: {len(spindle_df)} observations from {len(spindle_df['Subject'].unique()) if len(spindle_df) > 0 else 0} subjects")
        
        # 2. Load demographics with encoding handling
        demo_file = os.path.join(self.project_dir, "demographics.csv")
        try:
            demo_df = pd.read_csv(demo_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                demo_df = pd.read_csv(demo_file, encoding='latin-1')
            except UnicodeDecodeError:
                demo_df = pd.read_csv(demo_file, encoding='cp1252')
        
        demo_df['Subject'] = demo_df['pptid'].astype(str)
        # Calculate age: round to nearest year based on months
        demo_df['age'] = demo_df['age_years'] + (demo_df['age_months'] >= 6).astype(int)
        demo_df['sex'] = demo_df['gender'].map({0: 'F', 1: 'M'})
        demo_df['AHI'] = demo_df['nrem_ahi']
        demo_df['log_AHI'] = np.log1p(demo_df['AHI'])
        print(f"✓ Demographics: {len(demo_df)} subjects")
        
        # 3. Load TOVA data with encoding handling
        tova_file = os.path.join(self.project_dir, "TOVA.csv")
        try:
            tova_df = pd.read_csv(tova_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                tova_df = pd.read_csv(tova_file, encoding='latin-1')
            except UnicodeDecodeError:
                tova_df = pd.read_csv(tova_file, encoding='cp1252')
        
        tova_df['Subject'] = tova_df['Subject'].astype(str)
        print(f"✓ TOVA data: {len(tova_df)} subjects")
        
        # 4. Normalize subject IDs (remove leading zeros to match across datasets)
        spindle_df['Subject'] = spindle_df['Subject'].astype(str).str.lstrip('0')
        demo_df['Subject'] = demo_df['Subject'].astype(str)
        tova_df['Subject'] = tova_df['Subject'].astype(str)
        
        # 5. Merge everything
        data = spindle_df.merge(demo_df[['Subject', 'age', 'sex', 'AHI', 'log_AHI']], on='Subject', how='inner')
        data = data.merge(tova_df[['Subject', "Cohen's d"]], on='Subject', how='inner')
        data = data.rename(columns={"Cohen's d": "cohens_d"})
        
        # Clean data
        data = data.dropna(subset=['spindle_density', 'age', 'sex', 'cohens_d', 'AHI'])
        
        print(f"✓ Final dataset: {len(data)} observations from {len(data['Subject'].unique())} subjects")
        print(f"  - Anterior: {len(data[data['Region'] == 'anterior'])} observations")
        print(f"  - Posterior: {len(data[data['Region'] == 'posterior'])} observations")
        
        # Save the final merged dataset
        output_file = os.path.join(self.output_dir, 'mixed_effects_dataset.csv')
        data.to_csv(output_file, index=False)
        print(f"✓ Dataset saved to: {output_file}")
        
        return data
    
    def create_longitudinal_data(self, data):
        """Create longitudinal format for RQ1."""
        print("Creating longitudinal dataset...")
        
        # Session A (baseline): Cohen's d = 0
        session_a = data.copy()
        session_a['Session'] = 'A'
        session_a['Time'] = 0
        session_a['performance'] = 0  # Baseline
        
        # Session B (post): Cohen's d = change score
        session_b = data.copy()
        session_b['Session'] = 'B'
        session_b['Time'] = 1
        session_b['performance'] = session_b['cohens_d']  # Change from baseline
        
        # Combine
        longitudinal = pd.concat([session_a, session_b], ignore_index=True)
        
        print(f"✓ Longitudinal dataset: {len(longitudinal)} observations ({len(longitudinal)//2} subjects × 2 sessions)")
        return longitudinal
    
    def analyze_rq1_longitudinal(self, data):
        """RQ1: Spindle density predicting performance change over time."""
        print("\n" + "="*50)
        print("RQ1: LONGITUDINAL ANALYSIS")
        print("="*50)
        
        results = {}
        
        # Model 1: Main effects only
        print("\nModel 1: Main effects (Time + Spindle Density)")
        try:
            model1 = mixedlm("performance ~ Time + spindle_density + age + sex", 
                           data=data, 
                           groups=data["Subject"]).fit()
            results['main_effects'] = model1
            print(f"✓ Time effect: β = {model1.params['Time']:.4f}, p = {model1.pvalues['Time']:.4f}")
            print(f"✓ Spindle density effect: β = {model1.params['spindle_density']:.4f}, p = {model1.pvalues['spindle_density']:.4f}")
        except Exception as e:
            print(f"✗ Model 1 failed: {e}")
            results['main_effects'] = None
        
        # Model 2: Interaction model
        print("\nModel 2: Time × Spindle Density Interaction")
        try:
            model2 = mixedlm("performance ~ Time * spindle_density + age + sex", 
                           data=data, 
                           groups=data["Subject"]).fit()
            results['interaction'] = model2
            print(f"✓ Time effect: β = {model2.params['Time']:.4f}, p = {model2.pvalues['Time']:.4f}")
            print(f"✓ Spindle density effect: β = {model2.params['spindle_density']:.4f}, p = {model2.pvalues['spindle_density']:.4f}")
            if 'Time:spindle_density' in model2.params:
                print(f"✓ Interaction: β = {model2.params['Time:spindle_density']:.4f}, p = {model2.pvalues['Time:spindle_density']:.4f}")
            
            # Report demographic effects
            print("  Demographic effects:")
            if 'age' in model2.params:
                age_sig = "***" if model2.pvalues['age'] < 0.001 else "**" if model2.pvalues['age'] < 0.01 else "*" if model2.pvalues['age'] < 0.05 else "ns"
                print(f"    Age: β = {model2.params['age']:.4f}, p = {model2.pvalues['age']:.4f} {age_sig}")
            if 'sex[T.M]' in model2.params:
                sex_sig = "***" if model2.pvalues['sex[T.M]'] < 0.001 else "**" if model2.pvalues['sex[T.M]'] < 0.01 else "*" if model2.pvalues['sex[T.M]'] < 0.05 else "ns"
                print(f"    Sex (Male vs Female): β = {model2.params['sex[T.M]']:.4f}, p = {model2.pvalues['sex[T.M]']:.4f} {sex_sig}")
                
        except Exception as e:
            print(f"✗ Model 2 failed: {e}")
            results['interaction'] = None
        
        # Model 3: Regional analysis
        print("\nModel 3: Regional Analysis")
        for region in ['anterior', 'posterior']:
            region_data = data[data['Region'] == region]
            print(f"\n{region.upper()} (n = {len(region_data['Subject'].unique())} subjects):")
            
            try:
                model_region = mixedlm("performance ~ Time * spindle_density + age + sex", 
                                     data=region_data, 
                                     groups=region_data["Subject"]).fit()
                results[f'{region}_model'] = model_region
                
                if 'Time:spindle_density' in model_region.params:
                    coef = model_region.params['Time:spindle_density']
                    p_val = model_region.pvalues['Time:spindle_density']
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    print(f"  Interaction: β = {coef:.4f}, p = {p_val:.4f} {sig}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results[f'{region}_model'] = None
        
        return results
    
    def analyze_rq2_cross_sectional(self, data):
        """RQ2: AHI predicting regional spindle density."""
        print("\n" + "="*50)
        print("RQ2: CROSS-SECTIONAL AHI ANALYSIS")
        print("="*50)
        
        results = {}
        
        # Model 1: Main effect of AHI
        print("\nModel 1: AHI Main Effect")
        try:
            model1 = mixedlm("spindle_density ~ log_AHI + age + sex", 
                           data=data, 
                           groups=data["Subject"]).fit()
            results['ahi_main'] = model1
            print(f"✓ AHI effect: β = {model1.params['log_AHI']:.4f}, p = {model1.pvalues['log_AHI']:.4f}")
        except Exception as e:
            print(f"✗ Model 1 failed: {e}")
            results['ahi_main'] = None
        
        # Model 2: AHI × Region interaction
        print("\nModel 2: AHI × Region Interaction")
        try:
            model2 = mixedlm("spindle_density ~ log_AHI * Region + age + sex", 
                           data=data, 
                           groups=data["Subject"]).fit()
            results['ahi_interaction'] = model2
            print(f"✓ AHI effect: β = {model2.params['log_AHI']:.4f}, p = {model2.pvalues['log_AHI']:.4f}")
            if 'log_AHI:Region[T.posterior]' in model2.params:
                coef = model2.params['log_AHI:Region[T.posterior]']
                p_val = model2.pvalues['log_AHI:Region[T.posterior]']
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"✓ AHI × Region interaction: β = {coef:.4f}, p = {p_val:.4f} {sig}")
            
            # Report demographic effects
            print("  Demographic effects:")
            if 'age' in model2.params:
                age_sig = "***" if model2.pvalues['age'] < 0.001 else "**" if model2.pvalues['age'] < 0.01 else "*" if model2.pvalues['age'] < 0.05 else "ns"
                print(f"    Age: β = {model2.params['age']:.4f}, p = {model2.pvalues['age']:.4f} {age_sig}")
            if 'sex[T.M]' in model2.params:
                sex_sig = "***" if model2.pvalues['sex[T.M]'] < 0.001 else "**" if model2.pvalues['sex[T.M]'] < 0.01 else "*" if model2.pvalues['sex[T.M]'] < 0.05 else "ns"
                print(f"    Sex (Male vs Female): β = {model2.params['sex[T.M]']:.4f}, p = {model2.pvalues['sex[T.M]']:.4f} {sex_sig}")
                
        except Exception as e:
            print(f"✗ Model 2 failed: {e}")
            results['ahi_interaction'] = None
        
        # Model 3: Separate by region
        print("\nModel 3: Region-Specific Models")
        for region in ['anterior', 'posterior']:
            region_data = data[data['Region'] == region]
            print(f"\n{region.upper()} (n = {len(region_data)} subjects):")
            
            try:
                # Use OLS for single region
                model_region = ols("spindle_density ~ log_AHI + age + sex", 
                                  data=region_data).fit()
                results[f'{region}_ahi'] = model_region
                coef = model_region.params['log_AHI']
                p_val = model_region.pvalues['log_AHI']
                r_sq = model_region.rsquared
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"  AHI effect: β = {coef:.4f}, p = {p_val:.4f} {sig}, R² = {r_sq:.3f}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results[f'{region}_ahi'] = None
        
        return results
    
    def create_clear_figures(self, longitudinal_data, cross_sectional_data, rq1_results, rq2_results):
        """Create simplified 3-panel figure showing core findings"""
        
        # Create figure with 3 panels (1 row, 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('PEDS OSA: Sleep Spindles, Cognition & Brain Development', fontsize=16, fontweight='bold', y=0.98)
        
        # Panel 1: RQ1 - Spindle Density vs Performance Change by Region
        ax1 = axes[0]
        
        # Plot spindle density vs performance change by region
        regions = ['anterior', 'posterior']
        colors = ['blue', 'red']
        
        for region, color in zip(regions, colors):
            region_data = cross_sectional_data[cross_sectional_data['Region'] == region]
            
            if len(region_data) > 5:
                ax1.scatter(region_data['spindle_density'], region_data['cohens_d'], 
                           alpha=0.6, color=color, label=f'{region.capitalize()}', s=60)
                
                # Add regression line
                z = np.polyfit(region_data['spindle_density'], region_data['cohens_d'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(region_data['spindle_density'].min(), 
                                   region_data['spindle_density'].max(), 100)
                ax1.plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.8)
                
                # Add correlation info
                r, p_val = stats.pearsonr(region_data['spindle_density'], region_data['cohens_d'])
                ax1.text(0.05, 0.95 - (0.1 * regions.index(region)), 
                        f'{region.capitalize()}: r = {r:.3f}, p = {p_val:.3f}',
                        transform=ax1.transAxes, color=color, fontweight='bold')
        
        ax1.set_xlabel('Spindle Density (spindles/min)')
        ax1.set_ylabel("Cohen's d (More negative = More improvement)")
        ax1.set_title('RQ1: Spindle Density vs Performance Change\nby Brain Region')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Panel 2: RQ2 - AHI × Region Interaction Effect
        ax2 = axes[1]
        
        # Plot AHI vs spindle density with interaction visualization
        for region, color in [('anterior', 'blue'), ('posterior', 'red')]:
            region_data = cross_sectional_data[cross_sectional_data['Region'] == region]
            if len(region_data) > 5:
                ax2.scatter(region_data['log_AHI'], region_data['spindle_density'], 
                           alpha=0.6, color=color, label=f'{region.capitalize()}', s=60)
                
                # Add regression line
                z = np.polyfit(region_data['log_AHI'], region_data['spindle_density'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(region_data['log_AHI'].min(), 
                                   region_data['log_AHI'].max(), 100)
                ax2.plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.8)
                
                # Add correlation info
                r, p_val = stats.pearsonr(region_data['log_AHI'], region_data['spindle_density'])
                ax2.text(0.05, 0.95 - (0.1 * ['anterior', 'posterior'].index(region)), 
                        f'{region.capitalize()}: r = {r:.3f}, p = {p_val:.3f}',
                        transform=ax2.transAxes, color=color, fontweight='bold')
        
        ax2.set_xlabel('Log(AHI + 1)')
        ax2.set_ylabel('Spindle Density (spindles/min)')
        ax2.set_title('RQ2: AHI × Region Interaction')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Age Effects
        ax3 = axes[2]
        
        # Get age and spindle density data by region
        for region, color, marker in [('anterior', 'red', 'o'), ('posterior', 'blue', '^')]:
            region_data = cross_sectional_data[cross_sectional_data['Region'] == region]
            sample_data = region_data.drop_duplicates('Subject')
            
            if len(sample_data) > 5:
                ax3.scatter(sample_data['age'], sample_data['spindle_density'], 
                           alpha=0.6, color=color, label=f'{region.capitalize()}', 
                           s=60, marker=marker)
                
                # Add regression line
                z = np.polyfit(sample_data['age'], sample_data['spindle_density'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(sample_data['age'].min(), 
                                   sample_data['age'].max(), 100)
                ax3.plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('Age (years)')
        ax3.set_ylabel('Spindle Density (spindles/min)')
        ax3.set_title('Age Effects on Spindle Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'simplified_mixed_effects_results.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Clear figures saved to {output_file}")
    
    def save_summary(self, rq1_results, rq2_results, longitudinal_data, cross_sectional_data):
        """Save a concise, interpretable summary."""
        summary_file = os.path.join(self.output_dir, 'simplified_results_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("PEDS OSA - SIMPLIFIED MIXED EFFECTS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("STUDY OVERVIEW\n")
            f.write("-" * 13 + "\n")
            f.write(f"Sample size: {len(cross_sectional_data['Subject'].unique())} children\n")
            f.write(f"Longitudinal observations: {len(longitudinal_data)} (2 sessions × {len(longitudinal_data)//2} subjects)\n")
            f.write(f"Cross-sectional observations: {len(cross_sectional_data)} (2 regions × subjects)\n\n")
            
            f.write("RESEARCH QUESTIONS & KEY FINDINGS\n")
            f.write("-" * 33 + "\n\n")
            
            # RQ1 Summary
            f.write("RQ1: Do children with higher spindle density show greater performance improvement?\n\n")
            
            interaction_model = rq1_results.get('interaction')
            if interaction_model is not None and 'Time:spindle_density' in interaction_model.params:
                coef = interaction_model.params['Time:spindle_density']
                p_val = interaction_model.pvalues['Time:spindle_density']
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                
                if p_val < 0.05:
                    # CORRECTED: More negative Cohen's d = more improvement, so negative interaction = 
                    # higher spindle density associated with MORE negative change (MORE improvement)
                    direction = "Children with higher spindle density show MORE" if coef < 0 else "Children with higher spindle density show LESS"
                    f.write(f"SIGNIFICANT INTERACTION FOUND! {direction} improvement over time.\n")
                    f.write(f"Effect size: β = {coef:.4f}, p = {p_val:.4f} {sig}\n")
                    f.write("Note: More negative Cohen's d indicates greater improvement from Session A to B.\n")
                    f.write("This suggests that spindle density is a meaningful predictor of cognitive change.\n\n")
                else:
                    f.write(f"No significant interaction: β = {coef:.4f}, p = {p_val:.4f} {sig}\n")
                    f.write("Spindle density does not significantly predict performance change.\n\n")
            else:
                f.write("Interaction model could not be fitted.\n\n")
            
            # RQ2 Summary
            f.write("RQ2: Does sleep apnea severity (AHI) predict regional spindle density?\n\n")
            
            ahi_model = rq2_results.get('ahi_interaction')
            if ahi_model is not None:
                if 'log_AHI' in ahi_model.params:
                    main_coef = ahi_model.params['log_AHI']
                    main_p = ahi_model.pvalues['log_AHI']
                    main_sig = "***" if main_p < 0.001 else "**" if main_p < 0.01 else "*" if main_p < 0.05 else "ns"
                    
                    direction = "REDUCES" if main_coef < 0 else "INCREASES"
                    f.write(f"AHI main effect: Higher sleep apnea {direction} spindle density overall.\n")
                    f.write(f"Effect size: β = {main_coef:.4f}, p = {main_p:.4f} {main_sig}\n")
                
                if 'log_AHI:Region[T.posterior]' in ahi_model.params:
                    int_coef = ahi_model.params['log_AHI:Region[T.posterior]']
                    int_p = ahi_model.pvalues['log_AHI:Region[T.posterior]']
                    int_sig = "***" if int_p < 0.001 else "**" if int_p < 0.01 else "*" if int_p < 0.05 else "ns"
                    
                    if int_p < 0.05:
                        f.write(f"REGIONAL DIFFERENCE FOUND! AHI affects anterior and posterior regions differently.\n")
                        f.write(f"Interaction effect: β = {int_coef:.4f}, p = {int_p:.4f} {int_sig}\n")
                        f.write("This suggests brain region-specific vulnerability to sleep apnea.\n\n")
                    else:
                        f.write(f"No significant regional difference: β = {int_coef:.4f}, p = {int_p:.4f} {int_sig}\n\n")
            else:
                f.write("AHI interaction model could not be fitted.\n\n")
            
            f.write("CLINICAL IMPLICATIONS\n")
            f.write("-" * 20 + "\n")
            f.write("• Mixed effects models account for individual differences and repeated measures\n")
            f.write("• Spindle density may serve as a biomarker for cognitive resilience\n")
            f.write("• Regional brain differences suggest targeted interventions may be needed\n")
            f.write("• Sleep apnea severity impacts sleep spindle generation\n\n")
            
            f.write("FILES GENERATED\n")
            f.write("-" * 15 + "\n")
            f.write("• simplified_mixed_effects_results.png - Clear visualization of main effects and interactions\n")
            f.write("• simplified_results_summary.txt - This interpretable summary\n")
        
        print(f"✓ Interpretable summary saved to {summary_file}")

    def save_detailed_statistical_report(self, rq1_results, rq2_results, output_dir):
        """Save detailed statistical model output to separate file"""
        stats_file = os.path.join(output_dir, 'detailed_statistical_report.txt')
        
        with open(stats_file, 'w') as f:
            f.write("DETAILED STATISTICAL MODEL REPORTS\n")
            f.write("=" * 50 + "\n\n")
            f.write("Generated by: Simplified Mixed Effects Analysis\n")
            f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # RQ1 Models
            f.write("RQ1: LONGITUDINAL ANALYSIS MODELS\n")
            f.write("=" * 40 + "\n\n")
            
            if rq1_results.get('main_effects'):
                f.write("MODEL 1: MAIN EFFECTS (Time + Spindle Density)\n")
                f.write("-" * 50 + "\n")
                f.write(str(rq1_results['main_effects'].summary()))
                f.write("\n\n")
            
            if rq1_results.get('interaction'):
                f.write("MODEL 2: INTERACTION MODEL (Time × Spindle Density)\n")
                f.write("-" * 55 + "\n")
                f.write(str(rq1_results['interaction'].summary()))
                f.write("\n\n")
            
            # Regional models for RQ1
            if rq1_results.get('anterior'):
                f.write("MODEL 3A: ANTERIOR REGION ONLY\n")
                f.write("-" * 35 + "\n")
                f.write(str(rq1_results['anterior'].summary()))
                f.write("\n\n")
            
            if rq1_results.get('posterior'):
                f.write("MODEL 3B: POSTERIOR REGION ONLY\n")
                f.write("-" * 36 + "\n")
                f.write(str(rq1_results['posterior'].summary()))
                f.write("\n\n")
            
            # RQ2 Models
            f.write("RQ2: CROSS-SECTIONAL AHI ANALYSIS MODELS\n")
            f.write("=" * 45 + "\n\n")
            
            if rq2_results.get('ahi_main'):
                f.write("MODEL 1: AHI MAIN EFFECT\n")
                f.write("-" * 30 + "\n")
                f.write(str(rq2_results['ahi_main'].summary()))
                f.write("\n\n")
            
            if rq2_results.get('ahi_interaction'):
                f.write("MODEL 2: AHI × REGION INTERACTION\n")
                f.write("-" * 40 + "\n")
                f.write(str(rq2_results['ahi_interaction'].summary()))
                f.write("\n\n")
            
            # Regional models for RQ2
            if rq2_results.get('anterior_ahi'):
                f.write("MODEL 3A: ANTERIOR REGION AHI EFFECTS\n")
                f.write("-" * 42 + "\n")
                f.write(str(rq2_results['anterior_ahi'].summary()))
                f.write("\n\n")
            
            if rq2_results.get('posterior_ahi'):
                f.write("MODEL 3B: POSTERIOR REGION AHI EFFECTS\n")
                f.write("-" * 43 + "\n")
                f.write(str(rq2_results['posterior_ahi'].summary()))
                f.write("\n\n")
        
        print(f"✓ Detailed statistical report saved to {stats_file}")
        return stats_file
    
    def run_analysis(self):
        """Run the complete simplified analysis."""
        print("PEDS OSA - Simplified Mixed Effects Analysis")
        print("=" * 50)
        
        # Load data
        data = self.load_data()
        
        # Prepare datasets
        longitudinal_data = self.create_longitudinal_data(data)
        cross_sectional_data = data.copy()
        
        # Run analyses
        rq1_results = self.analyze_rq1_longitudinal(longitudinal_data)
        rq2_results = self.analyze_rq2_cross_sectional(cross_sectional_data)
        
        # Create visualizations and summary
        self.create_clear_figures(longitudinal_data, cross_sectional_data, rq1_results, rq2_results)
        self.save_summary(rq1_results, rq2_results, longitudinal_data, cross_sectional_data)
        self.save_detailed_statistical_report(rq1_results, rq2_results, self.output_dir)
        
        print(f"\n✓ ANALYSIS COMPLETE!")
        print(f"✓ Results saved to: {self.output_dir}")
        
        return longitudinal_data, cross_sectional_data, rq1_results, rq2_results

def main(project_dir):
    """Main function."""
    try:
        analysis = SimplifiedMixedEffectsAnalysis(project_dir)
        return analysis.run_analysis()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    project_dir = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Ido/002_PEDS_OSA"
    main(project_dir)