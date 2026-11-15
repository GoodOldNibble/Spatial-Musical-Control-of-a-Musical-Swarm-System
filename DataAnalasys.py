# Statistical analysis and plots
#
# Computes the statistical tests and produces the figures reported in Chapter 7 from the
# per-run dataset created by RunAggregator.py. The functions analyze_rq1, analyze_rq2 and
# analyze_rq3 compute the planned tests and create the figures for each research question.
# print_additional_statistics prints every statistical result reported in Chapters 7 and 8,
# including the exploratory analyses, and saves them to statistics_report.txt.
#
# Unit of analysis: one mean value per descriptor per run (n = 50 for RQ1 and RQ3,
# n = 100 for RQ2). Significance level: alpha = 0.05.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os

def analyze_rq1(file_path, save_plots=True):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Select the runs of scenario 1
    rq1_df = df[df['rq'] == 'RQ1'].copy()
    
    # Planned test: Pearson correlation between volume and ambitus
    pearson_r, p_value = stats.pearsonr(rq1_df['mean_volume'], rq1_df['mean_ambitus'])
    
    # Linear regression of ambitus on volume
    slope, intercept, r_value, p_reg, std_err = stats.linregress(rq1_df['mean_volume'], rq1_df['mean_ambitus'])
    r_squared = r_value**2

    # Print stats
    print(f"--- RQ1 Statistical Summary ---")
    print(f"Pearson Correlation (r): {pearson_r:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"R-squared (Predictability): {r_squared:.4f}")
    print(f"Regression Equation: Ambitus = {slope:.2e} * Volume + {intercept:.2f}")
    
    # Figure 7.2: scatter plot with regression line
    plt.figure()
    sns.regplot(data=rq1_df, x='mean_volume', y='mean_ambitus', 
                scatter_kws={'alpha':0.6, 's':50}, line_kws={'color':'red', 'label':f'R²={r_squared:.3f}'})
    #plt.title(f'Convex hull colume and global ambitus\n(R² = {r_squared:.3f}, p = {p_value:.4f})')
    plt.title(f'Convex hull colume and ambitus')
    plt.xlabel('Mean convex hull volume')
    plt.ylabel('Mean ambitus')
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig(path + '/plots/RQ1_Scatter_Regression.png')
    
    # Figure 7.1: mean and standard deviation across the five runs at each cohesion weight
    summary = rq1_df.groupby('weight').agg({
        'mean_volume': ['mean', 'std'],
        'mean_ambitus': ['mean', 'std']
    }).reset_index()
    summary.columns = ['weight', 'vol_mean', 'vol_std', 'amb_mean', 'amb_std']

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Cohesion weight')
    ax1.set_ylabel('Convex hull volume', color=color)
    ax1.plot(summary['weight'], summary['vol_mean'], color=color, marker='o', linewidth=2, label='Volume')
    ax1.fill_between(summary['weight'], summary['vol_mean'] - summary['vol_std'], 
                     summary['vol_mean'] + summary['vol_std'], color=color, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Ambitus', color=color)
    ax2.plot(summary['weight'], summary['amb_mean'], color=color, marker='s', linewidth=2, linestyle='--', label='Ambitus')
    ax2.fill_between(summary['weight'], summary['amb_mean'] - summary['amb_std'], 
                     summary['amb_mean'] + summary['amb_std'], color=color, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Impact of cohesion weight on volume and ambitus')
    fig.tight_layout()
    if save_plots:
        plt.savefig(path + '/plots/RQ1_Cohesion_Impact.png')
    
    return rq1_df

def analyze_rq2(file_path, save_plots=True):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Select the runs of scenario 2
    rq2_df = df[df['rq'] == 'RQ2'].copy()
    
    # Spearman correlation between global polarity and CV_IOI
    spearman_rho, p_spearman = stats.spearmanr(rq2_df['mean_polarity_physical'], rq2_df['cv_ioi'])
    
    # Spearman correlation between alignment weight and global polarity
    spearman_weight_pol_corr, p_weight_pol = stats.spearmanr(rq2_df['weight'], rq2_df['mean_polarity_physical'])

        # Print stats
    print(f"--- RQ2 Statistical Summary ---")
    print(f"Alignment Weight and Polarity (rho): {spearman_weight_pol_corr:.6f} (p = {p_weight_pol:.6f})")
    print(f"Polarity and CV of IOIs (rho): {spearman_rho:.6f} (p = {p_spearman:.6f})")
    
    # Mean and standard deviation across the five runs at each alignment weight
    summary = rq2_df.groupby('weight').agg({
        'mean_polarity_physical': ['mean', 'std'],
        'cv_ioi': ['mean', 'std']
    }).reset_index()
    summary.columns = ['weight', 'pol_mean', 'pol_std', 'cv_mean', 'cv_std']
    
    # Figure 7.4: CV_IOI against polarity, with the Poisson reference (CV = 1) and the
    # transition threshold (CV = 0.5)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rq2_df, x='mean_polarity_physical', y='cv_ioi', hue='weight', palette='viridis', s=60)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Poisson reference (CV=1.0)')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Transition threshold (CV=0.5)')
    plt.title('Polarity as an order parameter for rhythmic transition')
    plt.xlabel('Global polarity ($\phi$)')
    plt.ylabel('CV of IOIs')
    plt.legend(title='Alignment weight', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_plots:
        plt.savefig(path + '/plots/RQ2_Scatter_Phase_Transition.png')
    
    # Figure 7.3: polarity and CV_IOI across the alignment weights
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_pol = 'tab:blue'
    ax1.set_xlabel('Alignment weight')
    ax1.set_ylabel('Global polarity ($\phi$)', color=color_pol)
    ax1.plot(summary['weight'], summary['pol_mean'], color=color_pol, marker='o', linewidth=2, label='Polarity')
    ax1.fill_between(summary['weight'], summary['pol_mean'] - summary['pol_std'], 
                     summary['pol_mean'] + summary['pol_std'], color=color_pol, alpha=0.2)
    
    ax2 = ax1.twinx()
    color_cv = 'tab:orange'
    ax2.set_ylabel('CV of IOIs', color=color_cv)
    ax2.plot(summary['weight'], summary['cv_mean'], color=color_cv, marker='s', linewidth=2, linestyle='--', label='CV IOI')
    ax2.fill_between(summary['weight'], summary['cv_mean'] - summary['cv_std'], 
                     summary['cv_mean'] + summary['cv_std'], color=color_cv, alpha=0.2)
    
    plt.title('Impact of alignment on polarity and CV of IOIs')
    fig.tight_layout()
    if save_plots:
        plt.savefig(path + '/plots/RQ2_Alignment_Impact.png')
    
    return rq2_df

def analyze_rq3(file_path, save_plots=True):
    # Load the dataset and select the runs of scenario 3
    df = pd.read_csv(file_path)
    rq3_df = df[df['rq'] == 'RQ3'].copy()
    
    # Planned tests: Pearson and Spearman correlation between mean NND and dissonance
    pearson_r, p_pearson = stats.pearsonr(rq3_df['mean_nnd'], rq3_df['mean_dissonance'])
    spearman_rho, p_spearman = stats.spearmanr(rq3_df['mean_nnd'], rq3_df['mean_dissonance'])
    
    # Linear regression of dissonance on mean NND
    slope, intercept, r_value, p_reg, std_err = stats.linregress(rq3_df['mean_nnd'], rq3_df['mean_dissonance'])
    r_squared = r_value**2
    
    # Mean and standard deviation across the five runs at each separation weight
    summary = rq3_df.groupby('weight').agg({
        'mean_nnd': ['mean', 'std'],
        'mean_dissonance': ['mean', 'std']
    }).reset_index()
    summary.columns = ['weight', 'nnd_mean', 'nnd_std', 'diss_mean', 'diss_std']
    
    # Print Stats
    print(f"--- RQ3 STATISTICAL SUMMARY ---")
    print(f"Proxy Robustness of NND and Dissonance:")
    print(f"Pearson r: {pearson_r:.4f} (p = {p_pearson:.4f})")
    print(f"Spearman rho: {spearman_rho:.4f} (p = {p_spearman:.4f})")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Regression Model: Dissonance = {slope:.2e} * NND + {intercept:.4f}")
    
    # Figure 7.6: scatter plot with regression line
    plt.figure(figsize=(10, 6))
    sns.regplot(data=rq3_df, x='mean_nnd', y='mean_dissonance', 
                scatter_kws={'alpha':0.6, 's':50}, line_kws={'color':'green', 'label':f'R²={r_squared:.3f}'})
    #plt.title(f'Mean NND and system dissonance\n(R² = {r_squared:.4f}, p = {p_reg:.4f})')
    plt.title(f'Mean NND and system dissonance')
    plt.xlabel('Average nearest neighbor distance (Mean NND)')
    plt.ylabel('System dissonance')
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig(path + '/plots/RQ3_Scatter_Regression.png')
    
    # Figure 7.5: mean NND and dissonance across the separation weights
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Separation weight')
    ax1.set_ylabel('Mean NND', color='tab:green')
    ax1.plot(summary['weight'], summary['nnd_mean'], color='tab:green', marker='o', linewidth=2, label='NND')
    ax1.fill_between(summary['weight'], summary['nnd_mean'] - summary['nnd_std'], 
                     summary['nnd_mean'] + summary['nnd_std'], color='tab:green', alpha=0.2)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('System dissonance', color='tab:purple')
    ax2.plot(summary['weight'], summary['diss_mean'], color='tab:purple', marker='s', linewidth=2, linestyle='--', label='Dissonance')
    ax2.fill_between(summary['weight'], summary['diss_mean'] - summary['diss_std'], 
                     summary['diss_mean'] + summary['diss_std'], color='tab:purple', alpha=0.2)
    
    plt.title('Impact of separation weight on mean NND and system dissonance')
    fig.tight_layout()
    if save_plots:
        plt.savefig(path + '/plots/RQ3_Separation_Impact.png')


    # Correlation between separation weight and dissonance
    pea_r, p_pea = stats.pearsonr(rq3_df['weight'], rq3_df['mean_dissonance'])
    spear_rho, p_spear = stats.spearmanr(rq3_df['weight'], rq3_df['mean_dissonance'])
    slope_w, intercept_w, r_value_w, p_reg_w, std_err_w = stats.linregress(rq3_df['weight'], rq3_df['mean_dissonance'])
    r_squared_w = r_value_w**2
    
    # Print stats
    print(f"--- RQ3 STATISTICAL SUMMARY (2) ---")
    print(f"Correlation of Separation and Dissonance:")
    print(f"Pearson r: {pea_r:.4f} (p = {p_pea:.4f})")
    print(f"Spearman rho: {spear_rho:.4f} (p = {p_spear:.4f})")
    print(f"R-squared: {r_squared_w:.4f}")
    print(f"Regression Model: Dissonance = {slope_w:.2e} * Separation Weight + {intercept_w:.4f}")

    return rq3_df

def calculate_rq_means(file_path):
    # Mean of all descriptors per research question and weight
    df = pd.read_csv(file_path)

    # List of columns to average
    metrics = [
        'mean_volume',
        'mean_polarity_ryhtmic',
        'mean_polarity_physical',
        'mean_nnd',
        'mean_dissonance',
        'mean_ambitus',
        'cv_ioi'
    ]

    # Group by RQ, Parameter, and Weight, then calculate the mean
    means_df = df.groupby(['rq', 'parameter', 'weight'])[metrics].mean().reset_index()

    return means_df



# Additional statistics in Chapters 7 and 8
def print_additional_statistics(file_path, report_file=None):
    # Prints every statistical result reported in Chapters 7 and 8, including the
    # exploratory analyses, and optionally saves them to a text file.
    # Requires the dataset with the mean number of active notes per run (active_notes).
    df = pd.read_csv(file_path)
    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    def fmt_p(p):
        return "< 0.0001" if p < 0.0001 else f"{p:.4f}"

    def pearson(label, x, y):
        r, p = stats.pearsonr(x, y)
        out(f"{label:<58} Pearson r = {r:7.4f}, p = {fmt_p(p)}, n = {len(x)}")

    def spearman(label, x, y):
        rho, p = stats.spearmanr(x, y)
        out(f"{label:<58} Spearman rho = {rho:7.4f}, p = {fmt_p(p)}, n = {len(x)}")

    def regression(label, x, y):
        res = stats.linregress(x, y)
        out(f"{label:<58} slope = {res.slope:.4e}, intercept = {res.intercept:.4f}, "
            f"R^2 = {res.rvalue ** 2:.4f}, p = {fmt_p(res.pvalue)}, n = {len(x)}")
        return res

    def partial(label, x, y, z, rank=False):
        # Correlation between x and y after removing the linear effect of z from both.
        # With rank=True the variables are rank-transformed first (partial Spearman).
        # The p-value uses a t-distribution with n - 3 degrees of freedom.
        x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
        if rank:
            x, y, z = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
        design = np.column_stack([np.ones(len(z)), z])
        res_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        res_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        r = np.corrcoef(res_x, res_y)[0, 1]
        dof = len(x) - 3
        p = 2 * stats.t.sf(abs(r * np.sqrt(dof / (1 - r ** 2))), dof)
        kind = "partial Spearman rho" if rank else "partial Pearson r"
        out(f"{label:<58} {kind} = {r:7.4f}, p = {fmt_p(p)}, n = {len(x)}")

    def header(text):
        out()
        out("=" * 76)
        out(text)
        out("=" * 76)

    def by_weight(data, columns):
        return data.groupby('weight')[columns].agg(['mean', 'std']).round(4).to_string()

    out(f"Dataset: {os.path.basename(file_path)}")
    out("Runs per research question: " + ", ".join(f"{k}: {v}" for k, v in df.groupby('rq').size().items()))

    # RQ1
    header("RQ1: convex hull volume and ambitus")
    rq1 = df[df['rq'] == 'RQ1']
    rq1_nz = rq1[rq1['weight'] > 0]
    spearman("Cohesion weight vs volume", rq1['weight'], rq1['mean_volume'])
    spearman("Cohesion weight vs ambitus", rq1['weight'], rq1['mean_ambitus'])
    spearman("Cohesion weight vs mean NND", rq1['weight'], rq1['mean_nnd'])
    spearman("Cohesion weight vs ambitus, W_coh > 0 (exploratory)", rq1_nz['weight'], rq1_nz['mean_ambitus'])
    spearman("Cohesion weight vs volume, W_coh > 0 (exploratory)", rq1_nz['weight'], rq1_nz['mean_volume'])
    out()
    pearson("Volume vs ambitus", rq1['mean_volume'], rq1['mean_ambitus'])
    res = regression("Regression ambitus ~ volume", rq1['mean_volume'], rq1['mean_ambitus'])
    vmin, vmax = rq1['mean_volume'].min(), rq1['mean_volume'].max()
    out(f"  Slope per 10^7 cubic units: {res.slope * 1e7:.2f} semitones")
    out(f"  Volume range: {vmin:.3e} to {vmax:.3e}, predicted difference in ambitus: "
        f"{res.slope * (vmax - vmin):.2f} semitones")
    out(f"  Mean volume: {rq1['mean_volume'].mean():.3e} cubic units "
        f"({rq1['mean_volume'].mean() / 500 ** 3 * 100:.1f}% of the simulation space)")
    out()
    pearson("Volume vs ambitus, W_coh > 0 (exploratory)", rq1_nz['mean_volume'], rq1_nz['mean_ambitus'])
    res_nz = regression("Regression ambitus ~ volume, W_coh > 0 (exploratory)",
                        rq1_nz['mean_volume'], rq1_nz['mean_ambitus'])
    out(f"  Slope per 10^7 cubic units: {res_nz.slope * 1e7:.2f} semitones")
    partial("Volume vs ambitus | active notes, W_coh > 0 (exploratory)",
            rq1_nz['mean_volume'], rq1_nz['mean_ambitus'], rq1_nz['active_notes'])
    pearson("Active notes vs ambitus (exploratory)", rq1['active_notes'], rq1['mean_ambitus'])
    out("\nMean and standard deviation per cohesion weight")
    out(by_weight(rq1, ['mean_volume', 'mean_ambitus', 'mean_nnd', 'mean_dissonance', 'active_notes']))
    out("\nImplied mean speed in units per frame (active notes / 2.5, Section 8.1)")
    out((rq1.groupby('weight')['active_notes'].mean() / 2.5).round(2).to_string())

    # RQ2
    header("RQ2: global polarity and rhythmic regularity")
    rq2 = df[df['rq'] == 'RQ2']
    low, high = rq2[rq2['weight'] <= 0.25], rq2[rq2['weight'] >= 0.5]
    spearman("Alignment weight vs polarity", rq2['weight'], rq2['mean_polarity_physical'])
    spearman("Alignment weight vs polarity, W_ali <= 0.25 (exploratory)", low['weight'], low['mean_polarity_physical'])
    spearman("Alignment weight vs polarity, W_ali >= 0.5 (exploratory)", high['weight'], high['mean_polarity_physical'])
    spearman("Polarity vs CV_IOI", rq2['mean_polarity_physical'], rq2['cv_ioi'])
    spearman("Alignment weight vs CV_IOI", rq2['weight'], rq2['cv_ioi'])
    cv, pol = rq2['cv_ioi'], rq2['mean_polarity_physical']
    out(f"\n  CV_IOI range: {cv.min():.4f} to {cv.max():.4f}, mean {cv.mean():.4f} (SD {cv.std():.4f})")
    out(f"  Runs with 0.90 <= CV_IOI <= 1.05: {cv.between(0.90, 1.05).sum()} of {len(cv)}")
    out(f"  Runs with CV_IOI < 0.5: {(cv < 0.5).sum()}")
    out(f"  Polarity range: {pol.min():.4f} to {pol.max():.4f}")
    out("\nMean and standard deviation per alignment weight")
    out(by_weight(rq2, ['mean_polarity_physical', 'cv_ioi', 'active_notes']))

    # RQ3
    header("RQ3: average nearest neighbour distance and system dissonance")
    rq3 = df[df['rq'] == 'RQ3']
    rq3_nz = rq3[rq3['weight'] > 0]
    spearman("Separation weight vs mean NND", rq3['weight'], rq3['mean_nnd'])
    spearman("Separation weight vs dissonance", rq3['weight'], rq3['mean_dissonance'])
    out()
    pearson("Mean NND vs dissonance", rq3['mean_nnd'], rq3['mean_dissonance'])
    spearman("Mean NND vs dissonance", rq3['mean_nnd'], rq3['mean_dissonance'])
    res3 = regression("Regression dissonance ~ mean NND", rq3['mean_nnd'], rq3['mean_dissonance'])
    nmin, nmax = rq3['mean_nnd'].min(), rq3['mean_nnd'].max()
    out(f"  NND range: {nmin:.1f} to {nmax:.1f}, predicted difference in dissonance: {res3.slope * (nmax - nmin):.4f}")
    out(f"  Dissonance range: {rq3['mean_dissonance'].min():.4f} to {rq3['mean_dissonance'].max():.4f}")
    out()
    pearson("Mean NND vs dissonance, W_sep > 0 (exploratory)", rq3_nz['mean_nnd'], rq3_nz['mean_dissonance'])
    spearman("Mean NND vs dissonance, W_sep > 0 (exploratory)", rq3_nz['mean_nnd'], rq3_nz['mean_dissonance'])
    regression("Regression dissonance ~ mean NND, W_sep > 0 (exploratory)", rq3_nz['mean_nnd'], rq3_nz['mean_dissonance'])
    spearman("Separation weight vs mean NND, W_sep > 0 (exploratory)", rq3_nz['weight'], rq3_nz['mean_nnd'])
    spearman("Separation weight vs dissonance, W_sep > 0 (exploratory)", rq3_nz['weight'], rq3_nz['mean_dissonance'])
    spearman("Separation weight vs active notes, W_sep > 0 (exploratory)", rq3_nz['weight'], rq3_nz['active_notes'])
    pearson("Active notes vs dissonance, W_sep > 0 (exploratory)", rq3_nz['active_notes'], rq3_nz['mean_dissonance'])
    partial("Separation vs dissonance | active notes, W_sep > 0 (expl.)",
            rq3_nz['weight'], rq3_nz['mean_dissonance'], rq3_nz['active_notes'], rank=True)
    out("\nMean and standard deviation per separation weight")
    out(by_weight(rq3, ['mean_nnd', 'mean_dissonance', 'active_notes']))

    # Polarity measures
    header("Frame-based vs onset-based polarity")
    pearson("Frame-based vs onset-based polarity, all runs", df['mean_polarity_physical'], df['mean_polarity_ryhtmic'])
    out(f"  Mean absolute difference: {(df['mean_polarity_physical'] - df['mean_polarity_ryhtmic']).abs().mean():.4f}")

    if report_file:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nReport saved to {report_file}")


# Path to current folder for loading data and saving plots
path = os.path.dirname(os.path.abspath(__file__))

# Set aesthetic style for the plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)


# Run the analysis (the analysis of each research question can be enabled separately)
file = path + "/Data analysis/master_results_200_runs.csv"
rq1_results = analyze_rq1(file, save_plots=True)
rq2_results = analyze_rq2(file, save_plots=True)
rq3_results = analyze_rq3(file, save_plots=True)

results = calculate_rq_means(file)
print(results)

# All statistics reported in Chapters 7 and 8, computed from the dataset that also contains
# the mean number of active notes per run
file_with_notes = path + "/Data analysis/master_results_w_notes TEST.csv"
print_additional_statistics(file_with_notes, report_file=path + "/statistics_report TEST.txt")

#print("--- RQ1--- ")
#print(rq1_results)
#print("\n--- RQ2--- ")
#print(rq2_results)
#print("\n--- RQ3--- ")
#print(rq3_results)