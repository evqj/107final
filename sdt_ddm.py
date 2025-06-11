"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
"""

from datetime import datetime
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

# Mapping dictionaries for categorical variables
# These convert categorical labels to numeric codes for analysis
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from a CSV file into SDT format.
    
    Args:
        file_path: Path to the CSV file containing raw response data
        prepare_for: Type of analysis to prepare data for ('sdt' or 'delta plots')
        display: Whether to print summary statistics
        
    Returns:
        DataFrame with processed data in the requested format
    """
    # Read and preprocess data
    data = pd.read_csv(file_path)
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    
    if display:
        print("\nRaw data sample:")
        print(data.head())
        print("\nUnique conditions:", data['condition'].unique())
        print("Signal values:", data['signal'].unique())
    
    # Transform to SDT format if requested
    if prepare_for == 'sdt':
        # Group data by participant, condition, and signal presence
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        if display:
            print("\nGrouped data:")
            print(grouped.head())
        
        # Transform into SDT format (hits, misses, false alarms, correct rejections)
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Get signal and noise trials
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        
        data = pd.DataFrame(sdt_data)
        
        if display:
            print("\nSDT summary:")
            print(data)
            if data.empty:
                print("\nWARNING: Empty SDT summary generated!")
                print("Number of participants:", len(data['pnum'].unique()))
                print("Number of conditions:", len(data['condition'].unique()))
            else:
                print("\nSummary statistics:")
                print(data.groupby('condition').agg({
                    'hits': 'sum',
                    'misses': 'sum',
                    'false_alarms': 'sum',
                    'correct_rejections': 'sum',
                    'nSignal': 'sum',
                    'nNoise': 'sum'
                }).round(2))
    
    # Prepare data for delta plot analysis
    if prepare_for == 'delta plots':
        # Initialize DataFrame for delta plot data
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', 
                                      *[f'p{p}' for p in PERCENTILES]])
        
        # Process data for each participant and condition
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                # Get data for this participant and condition
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                
                # Calculate percentiles for overall RTs
                overall_rt = c_data['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['overall'],
                    **{f'p{p}': [np.percentile(overall_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for accurate responses
                accurate_rt = c_data[c_data['accuracy'] == 1]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['accurate'],
                    **{f'p{p}': [np.percentile(accurate_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for error responses
                error_rt = c_data[c_data['accuracy'] == 0]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['error'],
                    **{f'p{p}': [np.percentile(error_rt, p)] for p in PERCENTILES}
                })])
                
        if display:
            print("\nDelta plots data:")
            print(dp_data)
            
        data = pd.DataFrame(dp_data)

    return data


def apply_hierarchical_sdt_model(data):
    """Apply a hierarchical Signal Detection Theory model using PyMC.
    
    This function implements a Bayesian hierarchical model for SDT analysis,
    allowing for both group-level and individual-level parameter estimation.
    
    Args:
        data: DataFrame containing SDT summary statistics
        
    Returns:
        PyMC model object
    """
    # Get unique participants and conditions
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())
    
    # Define the hierarchical model
    with pm.Model() as sdt_model:
        # Group-level parameters
        mean_d_prime = pm.Normal('mean_d_prime', mu=0.0, sigma=1.0, shape=C)
        stdev_d_prime = pm.HalfNormal('stdev_d_prime', sigma=1.0)
        
        mean_criterion = pm.Normal('mean_criterion', mu=0.0, sigma=1.0, shape=C)
        stdev_criterion = pm.HalfNormal('stdev_criterion', sigma=1.0)
        
        # Individual-level parameters
        d_prime = pm.Normal('d_prime', mu=mean_d_prime, sigma=stdev_d_prime, shape=(P, C))
        criterion = pm.Normal('criterion', mu=mean_criterion, sigma=stdev_criterion, shape=(P, C))
        
        # Calculate hit and false alarm rates using SDT
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)
                
        # Likelihood for signal trials
        # Note: pnum is 1-indexed in the data, but needs to be 0-indexed for the model, so we change the indexing here.  The results table will show participant numbers starting from 0, so we need to interpret the results accordingly.
        pm.Binomial('hit_obs', 
                   n=data['nSignal'], 
                   p=hit_rate[data['pnum']-1, data['condition']], 
                   observed=data['hits'])
        
        # Likelihood for noise trials
        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'], 
                   p=false_alarm_rate[data['pnum']-1, data['condition']], 
                   observed=data['false_alarms'])
    
    return sdt_model

def draw_delta_plots(data, pnum, output_dir_path):
    """Draw delta plots comparing RT distributions between condition pairs.
    
    Creates a matrix of delta plots where:
    - Upper triangle shows overall RT distribution differences
    - Lower triangle shows RT differences split by correct/error responses
    
    Args:
        data: DataFrame with RT percentile data
        pnum: Participant number to plot
        output_dir_path: path of output directory
    """
    # Filter data for specified participant
    data = data[data['pnum'] == pnum]
    
    # Get unique conditions and create subplot matrix
    conditions = data['condition'].unique()
    n_conditions = len(conditions)
    
    # Create figure with subplots matrix
    fig, axes = plt.subplots(n_conditions, n_conditions, 
                            figsize=(4*n_conditions, 4*n_conditions))
    
    # Create output directory
    #OUTPUT_DIR = Path(__file__).parent.parent.parent / 'output'
    
    # Define marker style for plots
    marker_style = {
        'marker': 'o',
        'markersize': 10,
        'markerfacecolor': 'white',
        'markeredgewidth': 2,
        'linewidth': 3
    }
    
    # Create delta plots for each condition pair
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            # Add labels only to edge subplots
            if j == 0:
                axes[i,j].set_ylabel('Difference in RT (s)', fontsize=12)
            if i == len(axes)-1:
                axes[i,j].set_xlabel('Percentile', fontsize=12)
                
            # Skip diagonal and lower triangle for overall plots
            if i > j:
                continue
            if i == j:
                axes[i,j].axis('off')
                continue
            
            # Create masks for condition and plotting mode
            cmask1 = data['condition'] == cond1
            cmask2 = data['condition'] == cond2
            overall_mask = data['mode'] == 'overall'
            error_mask = data['mode'] == 'error'
            accurate_mask = data['mode'] == 'accurate'
            
            # Calculate RT differences for overall performance
            quantiles1 = [data[cmask1 & overall_mask][f'p{p}'] for p in PERCENTILES]
            quantiles2 = [data[cmask2 & overall_mask][f'p{p}'] for p in PERCENTILES]
            overall_delta = np.array(quantiles2) - np.array(quantiles1)
            
            # Calculate RT differences for error responses
            error_quantiles1 = [data[cmask1 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_quantiles2 = [data[cmask2 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_delta = np.array(error_quantiles2) - np.array(error_quantiles1)
            
            # Calculate RT differences for accurate responses
            accurate_quantiles1 = [data[cmask1 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_quantiles2 = [data[cmask2 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_delta = np.array(accurate_quantiles2) - np.array(accurate_quantiles1)
            
            # Plot overall RT differences
            axes[i,j].plot(PERCENTILES, overall_delta, color='black', **marker_style)
            
            # Plot error and accurate RT differences
            axes[j,i].plot(PERCENTILES, error_delta, color='red', **marker_style)
            axes[j,i].plot(PERCENTILES, accurate_delta, color='green', **marker_style)
            axes[j,i].legend(['Error', 'Accurate'], loc='upper left')

            # Set y-axis limits and add reference line
            axes[i,j].set_ylim(bottom=-1/3, top=1/2)
            axes[j,i].set_ylim(bottom=-1/3, top=1/2)
            axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
            axes[j,i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add condition labels
            axes[i,j].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            axes[j,i].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            plt.tight_layout()
            
    # Save the figure
    plt.savefig(output_dir_path / f'delta_plots_{pnum}.png')


def analyze_sdt_data(sdt_data, output_dir_path):
    sdt_model = apply_hierarchical_sdt_model(sdt_data)

    # Sample from the posterior distribution
    print("\nSampling from SDT model posterior... (This may take a while)")
    with sdt_model:
        sdt_trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True)

    # Check for convergence
    print("\n--- SDT Model Convergence Statistics (Rhat) ---")
    print(az.rhat(sdt_trace, var_names=['mean_d_prime', 'mean_criterion', 'stdev_d_prime', 'stdev_criterion']))

    print("\n--- SDT Model Effective Sample Sizes (ESS) ---")
    print(az.ess(sdt_trace, var_names=['mean_d_prime', 'mean_criterion', 'stdev_d_prime', 'stdev_criterion']))

    # Display posterior distributions
    print("\n--- SDT Model Posterior Distributions ---")
    # Population-level estimates
    print("\nPopulation-level (mean) parameters:")
    print(az.summary(sdt_trace, var_names=['mean_d_prime', 'mean_criterion'], round_to=2))

    # Person-specific estimates (example for first few participants and conditions)
    print("\nIndividual-level parameters (first 5 participants, all conditions):")
    # We need to map the condition indices back to names for clarity
    participant_ids = sdt_data['pnum'].unique()
    conditions = sdt_data['condition'].unique()

    # Create a mapping from numeric condition to descriptive name
    condition_idx_to_name = {v: k for k, v in MAPPINGS['stimulus_type'].items()}
    condition_idx_to_name.update({v: k for k, v in MAPPINGS['difficulty'].items()})
    condition_idx_to_name = {
        0: 'Easy Simple',
        1: 'Easy Complex',
        2: 'Hard Simple',
        3: 'Hard Complex'
    }


    # Summarize d_prime and criterion for each participant and condition
    d_prime_summary = az.summary(sdt_trace, var_names=['d_prime'], round_to=2)
    criterion_summary = az.summary(sdt_trace, var_names=['criterion'], round_to=2)

    # Print formatted individual estimates
    print("\nIndividual d_prime estimates:")
    for i, p_id in enumerate(participant_ids[:5]): # Limiting to first 5 participants
        for j, cond_idx in enumerate(conditions):
            # Arviz summary uses a flat index for shape=(P,C) parameters like 'd_prime[0,0]'
            # So we need to construct the correct index string
            dp_param_name = f'd_prime[{i}, {cond_idx}]'
            crit_param_name = f'criterion[{i}, {cond_idx}]'
                
            dp_mean = d_prime_summary.loc[dp_param_name, 'mean'] if dp_param_name in d_prime_summary.index else 'N/A'
            dp_sd = d_prime_summary.loc[dp_param_name, 'sd'] if dp_param_name in d_prime_summary.index else 'N/A'
                
            crit_mean = criterion_summary.loc[crit_param_name, 'mean'] if crit_param_name in criterion_summary.index else 'N/A'
            crit_sd = criterion_summary.loc[crit_param_name, 'sd'] if crit_param_name in criterion_summary.index else 'N/A'

            print(f"  P{p_id} - {condition_idx_to_name.get(cond_idx, f'Cond{cond_idx}')}: d_prime = {dp_mean} (SD={dp_sd}), criterion = {crit_mean} (SD={crit_sd})")

    # Plot posterior distributions for population-level parameters
    az.plot_posterior(sdt_trace, var_names=['mean_d_prime', 'mean_criterion'])
    plt.suptitle("Posterior Distributions of Population-level SDT Parameters")
    plt.savefig(output_dir_path / 'sdt_posterior_population.png')
    plt.close()

    # Plot posterior distributions for standard deviations
    az.plot_posterior(sdt_trace, var_names=['stdev_d_prime', 'stdev_criterion'])
    plt.suptitle("Posterior Distributions of SDT Parameter Standard Deviations")
    plt.savefig(output_dir_path / 'sdt_posterior_stdev.png')
    plt.close()


def run_sdt_model_analysis(data_file_path, output_dir_path):
    '''Running SDT Model Analysis

    Args:
        file_path: Path to the CSV file containing raw response data
    '''

    print("--- Running SDT Model Analysis ---")
    sdt_data = read_data(data_file_path, prepare_for='sdt', display=True)

    if not sdt_data.empty:
        #print(sdt_data)   # For debugging

        analyze_sdt_data(sdt_data, output_dir_path)
        print("SDT data analysis generated and saved in the " + str(output_dir_path) + " directory.")
    else:
        print("SDT data is empty. Skipping SDT model analysis.")


def run_delta_plot_analysis(data_file_path, output_dir_path):
    '''Running Delta Plot Analysis

    Args:
        file_path: Path to the CSV file containing raw response data
    '''
    # --- Delta Plot Analysis ---
    print("\n--- Running Delta Plot Analysis ---")
    delta_plot_data = read_data(data_file_path, prepare_for='delta plots', display=True)

    if not delta_plot_data.empty:
        #print(delta_plot_data)   # For debugging

        # Get unique participant numbers for delta plots
        participant_numbers_for_delta_plots = delta_plot_data['pnum'].unique()

        # Draw delta plots for each participant
        for pnum in participant_numbers_for_delta_plots:
            print(f"Generating delta plots for participant {pnum}...")
            draw_delta_plots(delta_plot_data, pnum, output_dir_path)
        print("Delta plots generated and saved in the " + str(output_dir_path) + " directory.")
    else:
        print("Delta plot data is empty. Skipping delta plot analysis.")


# Main execution
if __name__ == "__main__":
    data_file_path = Path(__file__).parent /'data.csv'

    # Create output directory
    OUTPUT_DIR_NAME = 'output_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    OUTPUT_DIR_PATH = Path(__file__).parent / OUTPUT_DIR_NAME
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

    ### For debugging
    #with open(data_file_path, 'r') as file:
    #    print(file.read())
    ###

    # --- SDT Model Analysis ---
    run_sdt_model_analysis(data_file_path, OUTPUT_DIR_PATH)

    # --- Delta Plot Analysis ---
    run_delta_plot_analysis(data_file_path, OUTPUT_DIR_PATH)
