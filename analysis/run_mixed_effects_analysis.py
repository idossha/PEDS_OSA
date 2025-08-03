#!/usr/bin/env python3
"""
Simple Runner for PEDS OSA Mixed Effects Analysis
===============================================

Runs simplified mixed effects analysis for:
1. Spindle density predicting performance change over time
2. AHI predicting regional spindle density

Usage:
    python run_mixed_effects_analysis.py [project_directory]
"""

import os
import sys

def main():
    """Run the simplified mixed effects analysis."""
    
    # Get project directory
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
    else:
        # Try config file, otherwise use current directory parent
        try:
            from config import PROJECT_DIR
            project_dir = PROJECT_DIR
        except ImportError:
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"Project directory: {project_dir}")
    
    # Validate directory
    if not os.path.exists(project_dir):
        print(f"❌ Error: Project directory not found: {project_dir}")
        print(f"Usage: python {sys.argv[0]} /path/to/project")
        return False
    
    # Check for required files
    required_files = ["demographics.csv", "TOVA.csv"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(project_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    try:
        # Import and run analysis
        from regional_mixed_effects_analysis import main as run_analysis
        
        print("\nStarting Simplified Mixed Effects Analysis...")
        print("=" * 50)
        
        results = run_analysis(project_dir)
        
        if results is not None:
            longitudinal_data, cross_sectional_data, rq1_results, rq2_results = results
            
            print("\n" + "=" * 50)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            
            # Quick results summary
            print("\nQUICK RESULTS SUMMARY:")
            print("-" * 25)
            
            n_subjects = len(cross_sectional_data['Subject'].unique())
            print(f"Sample: {n_subjects} children")
            
            # RQ1 Key finding
            print("\nRQ1: Spindle Density → Performance Change")
            interaction_model = rq1_results.get('interaction')
            if interaction_model and 'Time:spindle_density' in interaction_model.params:
                p_val = interaction_model.pvalues['Time:spindle_density']
                coef = interaction_model.params['Time:spindle_density']
                if p_val < 0.05:
                    # CORRECTED: More negative Cohen's d = more improvement
                    trend = "higher spindle density → MORE improvement" if coef < 0 else "higher spindle density → less improvement"
                    print(f"   Significant interaction! ({trend})")
                    print(f"   p = {p_val:.4f}, β = {coef:.4f}")
                    print(f"   More negative Cohen's d = more improvement")
                else:
                    print(f"   No significant interaction (p = {p_val:.4f})")
            else:
                print("   Could not fit interaction model")
            
            # RQ2 Key finding
            print("\nRQ2: Sleep Apnea (AHI) → Regional Spindle Density")
            ahi_model = rq2_results.get('ahi_interaction')
            if ahi_model and 'log_AHI:Region[T.posterior]' in ahi_model.params:
                p_val = ahi_model.pvalues['log_AHI:Region[T.posterior]']
                if p_val < 0.05:
                    print(f"   Significant regional difference!")
                    print(f"   AHI affects anterior/posterior differently (p = {p_val:.4f})")
                else:
                    print(f"   No significant regional difference (p = {p_val:.4f})")
            elif ahi_model and 'log_AHI' in ahi_model.params:
                p_val = ahi_model.pvalues['log_AHI']
                coef = ahi_model.params['log_AHI']
                if p_val < 0.05:
                    effect = "reduces" if coef < 0 else "increases"
                    print(f"   AHI {effect} spindle density (p = {p_val:.4f})")
                else:
                    print(f"   No significant AHI effect (p = {p_val:.4f})")
            else:
                print("   Could not fit AHI model")
            
            # Output info
            output_dir = os.path.join(project_dir, "mixed_effects_results")
            print(f"\nResults saved to: {output_dir}")
            print("Generated files:")
            print("   - simplified_mixed_effects_results.png (clear figures)")
            print("   - simplified_results_summary.txt (interpretable summary)")
            print("   - detailed_statistical_report.txt (full model outputs)")
            print("   - mixed_effects_dataset.csv (combined analysis dataset)")
            
            return True
            
        else:
            print("Analysis failed. Check error messages above.")
            return False
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure required packages are installed:")
        print("   pip install pandas numpy scipy statsmodels matplotlib seaborn")
        return False
        
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)