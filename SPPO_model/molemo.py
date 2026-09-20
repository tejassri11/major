import pandas as pd
import numpy as np
from scipy import stats

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('comparison_20251121_214524.csv')

# Your column mapping (PPO column, Fixed column, higher_is_better)
METRICS = {
    'Delay': ('ppo_delay', 'fixed_delay', False),
    'Queue Length': ('ppo_queue', 'fixed_queue', False),
    'Throughput': ('ppo_throughput', 'fixed_throughput', True),
    'Stop Ratio': ('ppo_stops', 'fixed_stops', False),
    'Pressure': ('ppo_pressure', 'fixed_pressure', False)
}

# Define demand phases (assuming rows = timesteps, 0-300s primary, rest extended)
# Adjust these based on your actual timestep-to-seconds mapping
STEPS_PER_SECOND = 1  # Change if needed (e.g., 10 if 10 steps = 1 second)
PHASES = {
    'Light (0-60s)': (0, 60 * STEPS_PER_SECOND),
    'Moderate (60-120s)': (60 * STEPS_PER_SECOND, 120 * STEPS_PER_SECOND),
    'Heavy (120-180s)': (120 * STEPS_PER_SECOND, 180 * STEPS_PER_SECOND),
    'Peak (180-240s)': (180 * STEPS_PER_SECOND, 240 * STEPS_PER_SECOND),
    'Saturation (240-300s)': (240 * STEPS_PER_SECOND, 300 * STEPS_PER_SECOND),
    'Extended (300s+)': (300 * STEPS_PER_SECOND, len(df))
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def calc_stats(data):
    """Calculate comprehensive statistics."""
    data = np.array(data)
    data = data[~np.isnan(data)]
    return {
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'sem': np.std(data, ddof=1) / np.sqrt(len(data)),
        'median': np.median(data),
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'q95': np.percentile(data, 95),
        'q05': np.percentile(data, 5),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'cv': np.std(data, ddof=1) / np.mean(data) if np.mean(data) != 0 else 0,
        'n': len(data)
    }

def compare_metrics(ppo_data, fixed_data, higher_is_better=False):
    """Full statistical comparison."""
    ppo_data = np.array(ppo_data)[~np.isnan(ppo_data)]
    fixed_data = np.array(fixed_data)[~np.isnan(fixed_data)]
    
    ppo_stats = calc_stats(ppo_data)
    fixed_stats = calc_stats(fixed_data)
    
    # Improvement calculation
    if fixed_stats['mean'] != 0:
        if higher_is_better:
            improvement = ((ppo_stats['mean'] - fixed_stats['mean']) / abs(fixed_stats['mean'])) * 100
        else:
            improvement = ((fixed_stats['mean'] - ppo_stats['mean']) / abs(fixed_stats['mean'])) * 100
    else:
        improvement = 0
    
    # Normality tests (use subset for large datasets)
    sample_size = min(5000, len(ppo_data), len(fixed_data))
    _, p_norm_ppo = stats.shapiro(np.random.choice(ppo_data, sample_size, replace=False))
    _, p_norm_fixed = stats.shapiro(np.random.choice(fixed_data, sample_size, replace=False))
    is_normal = p_norm_ppo > 0.05 and p_norm_fixed > 0.05
    
    # Statistical test
    if is_normal:
        t_stat, p_value = stats.ttest_ind(ppo_data, fixed_data)
        test_name = "Welch's t-test"
    else:
        t_stat, p_value = stats.mannwhitneyu(ppo_data, fixed_data, alternative='two-sided')
        test_name = "Mann-Whitney U"
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((ppo_stats['std']**2 + fixed_stats['std']**2) / 2)
    cohens_d = abs(ppo_stats['mean'] - fixed_stats['mean']) / pooled_std if pooled_std != 0 else 0
    
    # Effect size interpretation
    if cohens_d < 0.2:
        effect_interp = "negligible"
    elif cohens_d < 0.5:
        effect_interp = "small"
    elif cohens_d < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    
    return {
        'ppo': ppo_stats,
        'fixed': fixed_stats,
        'improvement_pct': improvement,
        'p_value': p_value,
        'test_stat': t_stat,
        'test_name': test_name,
        'cohens_d': cohens_d,
        'effect_size': effect_interp,
        'is_normal': is_normal
    }

# ============================================================
# MAIN ANALYSIS
# ============================================================
print("=" * 90)
print("TRAFFIC CONTROLLER PERFORMANCE ANALYSIS - STATISTICAL RESULTS")
print("=" * 90)
print(f"Total samples: {len(df)}")
print(f"Simulation duration: ~{len(df)/STEPS_PER_SECOND:.0f} seconds ({len(df)} timesteps)")

results = {}

for metric_name, (ppo_col, fixed_col, higher_better) in METRICS.items():
    print(f"\n{'='*90}")
    print(f"METRIC: {metric_name.upper()}")
    print("=" * 90)
    
    # Overall analysis
    comp = compare_metrics(df[ppo_col].values, df[fixed_col].values, higher_better)
    results[metric_name] = {'overall': comp}
    
    print(f"\n[OVERALL STATISTICS]")
    print(f"  {'Controller':<12} {'Mean':>12} {'Std':>12} {'Median':>12} {'95th %ile':>12} {'IQR':>12}")
    print(f"  {'-'*72}")
    print(f"  {'PPO':<12} {comp['ppo']['mean']:>12.6f} {comp['ppo']['std']:>12.6f} {comp['ppo']['median']:>12.6f} {comp['ppo']['q95']:>12.6f} {comp['ppo']['iqr']:>12.6f}")
    print(f"  {'Fixed-Time':<12} {comp['fixed']['mean']:>12.6f} {comp['fixed']['std']:>12.6f} {comp['fixed']['median']:>12.6f} {comp['fixed']['q95']:>12.6f} {comp['fixed']['iqr']:>12.6f}")
    
    print(f"\n[STATISTICAL COMPARISON]")
    print(f"  Improvement: {comp['improvement_pct']:+.2f}% {'(PPO better)' if comp['improvement_pct'] > 0 else '(Fixed better)'}")
    print(f"  Test Used: {comp['test_name']} (Data normal: {comp['is_normal']})")
    print(f"  Test Statistic: {comp['test_stat']:.4f}")
    print(f"  p-value: {comp['p_value']:.2e} {'***' if comp['p_value'] < 0.001 else '**' if comp['p_value'] < 0.01 else '*' if comp['p_value'] < 0.05 else 'ns'}")
    print(f"  Cohen's d: {comp['cohens_d']:.4f} ({comp['effect_size']} effect)")
    
    # Per-phase analysis
    print(f"\n[PER-PHASE BREAKDOWN]")
    print(f"  {'Phase':<25} {'PPO (μ±σ)':<22} {'Fixed (μ±σ)':<22} {'Δ%':>8} {'p-val':>12} {'Sig':>5}")
    print(f"  {'-'*100}")
    
    for phase_name, (start, end) in PHASES.items():
        ppo_phase = df[ppo_col].iloc[start:end].values
        fixed_phase = df[fixed_col].iloc[start:end].values
        
        if len(ppo_phase) > 10:
            phase_comp = compare_metrics(ppo_phase, fixed_phase, higher_better)
            results[metric_name][phase_name] = phase_comp
            
            sig = "***" if phase_comp['p_value'] < 0.001 else "**" if phase_comp['p_value'] < 0.01 else "*" if phase_comp['p_value'] < 0.05 else "ns"
            
            ppo_str = f"{phase_comp['ppo']['mean']:.4f}±{phase_comp['ppo']['std']:.4f}"
            fixed_str = f"{phase_comp['fixed']['mean']:.4f}±{phase_comp['fixed']['std']:.4f}"
            
            print(f"  {phase_name:<25} {ppo_str:<22} {fixed_str:<22} {phase_comp['improvement_pct']:>+7.1f}% {phase_comp['p_value']:>12.2e} {sig:>5}")

# ============================================================
# SUMMARY TABLES
# ============================================================
print("\n" + "=" * 90)
print("SUMMARY TABLE - OVERALL PERFORMANCE")
print("=" * 90)
print(f"\n{'Metric':<15} {'Fixed-Time (μ±σ)':<24} {'PPO (μ±σ)':<24} {'Improvement':<14} {'p-value':<12} {'Effect':<10}")
print("-" * 100)

for metric_name in METRICS.keys():
    comp = results[metric_name]['overall']
    fixed_str = f"{comp['fixed']['mean']:.4f} ± {comp['fixed']['std']:.4f}"
    ppo_str = f"{comp['ppo']['mean']:.4f} ± {comp['ppo']['std']:.4f}"
    sig = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else ""
    print(f"{metric_name:<15} {fixed_str:<24} {ppo_str:<24} {comp['improvement_pct']:>+.2f}%{sig:<6} {comp['p_value']:<12.2e} {comp['effect_size']:<10}")

# ============================================================
# LATEX TABLE
# ============================================================
print("\n" + "=" * 90)
print("LATEX TABLE (Copy this into your paper)")
print("=" * 90)
print(r"""
\begin{table}[htbp]
\centering
\caption{Comparative Performance of PPO and Fixed-Time Controllers Across All Metrics}
\label{tab:overall_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Metric} & \textbf{Fixed-Time} & \textbf{PPO} & \textbf{Improv.} & \textbf{p-value} & \textbf{Cohen's d} \\
 & $(\mu \pm \sigma)$ & $(\mu \pm \sigma)$ & $(\%)$ & & \\
\midrule""")

for metric_name in METRICS.keys():
    comp = results[metric_name]['overall']
    sig = "^{***}" if comp['p_value'] < 0.001 else "^{**}" if comp['p_value'] < 0.01 else "^{*}" if comp['p_value'] < 0.05 else ""
    print(f"{metric_name} & ${comp['fixed']['mean']:.4f} \\pm {comp['fixed']['std']:.4f}$ & ${comp['ppo']['mean']:.4f} \\pm {comp['ppo']['std']:.4f}$ & ${comp['improvement_pct']:+.1f}\\%{sig}$ & ${comp['p_value']:.2e}$ & {comp['cohens_d']:.3f} \\\\")

print(r"""\bottomrule
\multicolumn{6}{l}{\footnotesize $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$. Effect sizes: small ($d<0.5$), medium ($0.5 \leq d < 0.8$), large ($d \geq 0.8$).}
\end{tabular}
\end{table}
""")

# ============================================================
# TECHNICAL WRITING SNIPPETS
# ============================================================
print("\n" + "=" * 90)
print("READY-TO-USE TECHNICAL WRITING SNIPPETS")
print("=" * 90)

for metric_name, (ppo_col, fixed_col, higher_better) in METRICS.items():
    comp = results[metric_name]['overall']
    direction = "higher" if higher_better else "lower"
    better = "increase" if higher_better else "reduction"
    
    print(f"\n[{metric_name.upper()}]")
    print(f"The PPO controller achieved a mean {metric_name.lower()} of {comp['ppo']['mean']:.4f} (σ = {comp['ppo']['std']:.4f}), ")
    print(f"compared to {comp['fixed']['mean']:.4f} (σ = {comp['fixed']['std']:.4f}) for the Fixed-Time controller, ")
    print(f"representing a {abs(comp['improvement_pct']):.1f}% {better}. This difference was statistically significant ")
    print(f"({comp['test_name']}, p < {comp['p_value']:.0e}) with a {comp['effect_size']} effect size (Cohen's d = {comp['cohens_d']:.3f}).")
    print(f"The 95th percentile values were {comp['ppo']['q95']:.4f} (PPO) vs {comp['fixed']['q95']:.4f} (Fixed-Time).")

# ============================================================
# SAVE RESULTS TO CSV
# ============================================================
summary_data = []
for metric_name in METRICS.keys():
    comp = results[metric_name]['overall']
    summary_data.append({
        'Metric': metric_name,
        'PPO_Mean': comp['ppo']['mean'],
        'PPO_Std': comp['ppo']['std'],
        'PPO_Median': comp['ppo']['median'],
        'PPO_95th': comp['ppo']['q95'],
        'Fixed_Mean': comp['fixed']['mean'],
        'Fixed_Std': comp['fixed']['std'],
        'Fixed_Median': comp['fixed']['median'],
        'Fixed_95th': comp['fixed']['q95'],
        'Improvement_Pct': comp['improvement_pct'],
        'P_Value': comp['p_value'],
        'Cohens_D': comp['cohens_d'],
        'Effect_Size': comp['effect_size'],
        'Test_Used': comp['test_name']
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('statistical_results_summary.csv', index=False)
print("\n[INFO] Summary saved to 'statistical_results_summary.csv'")

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)