# Run aggregator
#
# Reads the raw output of all batch runs (one _metrics.csv and one _rhythm.csv per run),
# removes the warm-up period, and reduces each run to one row of summary statistics.
# The resulting file is the per-run dataset used for all statistical tests in the thesis.

import pandas as pd
import numpy as np
import glob
import os
import re

def parse_filename(filename):
    """
    Extracts metadata from the filename pattern: 
    BatchX_RunX_RQX_Parameter_Value_type.csv
    """
    basename = os.path.basename(filename)
    # Regex to capture the components
    pattern = r"Batch(\d+)_Run(\d+)_(RQ\d+)_([a-zA-Z]+)_([\d\.]+)_"
    match = re.search(pattern, basename)
    if match:
        return {
            "batch": int(match.group(1)),
            "run_id": int(match.group(2)),
            "rq": match.group(3),
            "parameter": match.group(4),
            "weight": float(match.group(5))
        }
    return None

def process_swarm_data(directory="Data runs"):
    all_results = []
    
    # Find all metric files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, directory)
    metric_files = glob.glob(os.path.join(folder, "*_metrics.csv"))
    
    for m_file in metric_files:
        # Parse batch, run, research question, parameter and weight from the filename
        meta = parse_filename(m_file)
        if not meta: continue
        
        # Identify the corresponding rhythm file
        r_file = m_file.replace("_metrics.csv", "_rhythm.csv")
        if not os.path.exists(r_file):
            print(f"Warning: Missing rhythm file for {m_file}")
            continue
            
        # Frame-based stream (spatial and musical descriptors)
        df_m = pd.read_csv(m_file)
        # Discard the first 300 frames (approximately 15 s) as warm-up
        df_m_clean = df_m.iloc[300:].copy()
        # Wall-clock duration of the warm-up, used to trim the event-based stream
        relative_warmup_duration = df_m.iloc[300]['timestamp'] - df_m.iloc[0]['timestamp']

        # Event-based stream (note onsets and inter-onset intervals)
        df_r = pd.read_csv(r_file)
        # Discard onsets within the same duration as the warm-up, measured from the first onset
        rhythm_start_time = df_r['timestamp'].min()
        df_r_clean = df_r[df_r['timestamp'] >= (rhythm_start_time + relative_warmup_duration)].copy()
        
        # Mean of each descriptor over the remaining frames or onsets.
        # mean_polarity_ryhtmic is polarity sampled at note onsets (the misspelt column name is
        # kept for consistency with the dataset), mean_polarity_physical is polarity per frame.
        metrics_summary = {
            "mean_volume": df_m_clean['swarm_volume'].mean(),
            "mean_polarity_ryhtmic": df_r_clean['polarity'].mean(),
            "mean_polarity_physical": df_m_clean['global_polarity'].mean(),
            "mean_nnd": df_m_clean['mean_nnd'].mean(),
            "mean_dissonance": df_m_clean['system_dissonance'].mean(),
            "mean_ambitus": df_m_clean['ambitus'].mean(), 
            "Cohesion": df_m_clean['cohesion_weight'].mean(), 
            "Alignment": df_m_clean['alignment_weight'].mean(), 
            "Separation": df_m_clean['separation_weight'].mean(),
            "active_notes": df_m_clean['num_active_notes'].mean()
        }
        
        
        # Coefficient of variation of the inter-onset intervals (CV = std / mean),
        # computed over the merged onsets of all agents
        if len(df_r_clean) > 1:
            iois = df_r_clean['ioi']
            cv_ioi = iois.std() / iois.mean() if iois.mean() != 0 else 0
        else:
            cv_ioi = np.nan
            
        # Combine everything into one row
        row = {**meta, **metrics_summary, "cv_ioi": cv_ioi}
        all_results.append(row)
    
    # Create the table with one row per run
    master_df = pd.DataFrame(all_results)
    
    # Sort for clarity
    master_df = master_df.sort_values(['rq', 'parameter', 'weight', 'batch'])
    
    # Save the result
    master_df.to_csv(script_dir + "/Data analysis/master_results_w_notes TEST.csv", index=False)
    print(f"Success! Processed {len(master_df)} runs into master_results_w_notes TEST.csv")
    return master_df

# Execute
if __name__ == "__main__":
    master_table = process_swarm_data(directory='Data runs')
    print(master_table.head())