
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set clean style
plt.style.use('default')
sns.set_palette("husl")

def load_results(csv_path):
    """Load simulation results from CSV"""
    return pd.read_csv(csv_path)

def plot_mse_vs_df(df, save_path=None):
    """Plot MSE vs degrees of freedom for each method"""
    fig, ax = plt.subplots(figsize=(5, 3))
    
    # Aggregate results
    grouped = df.groupby(['method', 'df'])['mse'].mean().reset_index()
    
    # Plot each method
    methods = df['method'].unique()
    colors = sns.color_palette("husl", len(methods))
    
    for i, method in enumerate(methods):
        data = grouped[grouped['method'] == method]
        ax.plot(data['df'], data['mse'], 'o-', 
               color=colors[i], label=method, linewidth=2, markersize=4)
    
    ax.set_xlabel('Degrees of Freedom', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_small_multiples_snr(df, save_path=None):
    """Small multiples: MSE vs df by SNR"""
    snr_values = sorted(df['SNR'].unique())
    n_plots = len(snr_values)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2), 
                            sharey=True, sharex=True)
    if n_plots == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    methods = df['method'].unique()
    colors = sns.color_palette("husl", len(methods))
    
    for i, snr in enumerate(snr_values):
        if i >= len(axes):
            break
            
        ax = axes[i]
        subset = df[df['SNR'] == snr]
        grouped = subset.groupby(['method', 'df'])['mse'].mean().reset_index()
        
        for j, method in enumerate(methods):
            data = grouped[grouped['method'] == method]
            ax.plot(data['df'], data['mse'], 'o-', 
                   color=colors[j], label=method if i == 0 else "", 
                   linewidth=1.5, markersize=3)
        
        ax.set_title(f'SNR = {snr}', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Remove empty subplots
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])
    
    # Add common labels
    fig.text(0.5, 0.02, 'Degrees of Freedom', ha='center', fontsize=11)
    fig.text(0.02, 0.5, 'MSE', va='center', rotation=90, fontsize=11)
    
    if n_plots > 0:
        axes[0].legend(frameon=False, fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, left=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_small_multiples_aspect_ratio(df, save_path=None):
    """Small multiples: MSE vs df by aspect ratio (n/p)"""
    df = df.copy()
    df['aspect_ratio'] = df['n'] / df['p']
    
    # Round aspect ratios for cleaner grouping
    df['ar_rounded'] = np.round(df['aspect_ratio'], 1)
    ar_values = sorted(df['ar_rounded'].unique())
    
    n_plots = len(ar_values)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2),
                            sharey=True, sharex=True)
    if n_plots == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    methods = df['method'].unique()
    colors = sns.color_palette("husl", len(methods))
    
    for i, ar in enumerate(ar_values):
        if i >= len(axes):
            break
            
        ax = axes[i]
        subset = df[df['ar_rounded'] == ar]
        grouped = subset.groupby(['method', 'df'])['mse'].mean().reset_index()
        
        for j, method in enumerate(methods):
            data = grouped[grouped['method'] == method]
            ax.plot(data['df'], data['mse'], 'o-', 
                   color=colors[j], label=method if i == 0 else "", 
                   linewidth=1.5, markersize=3)
        
        ax.set_title(f'n/p = {ar}', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Remove empty subplots
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])
    
    # Add common labels
    fig.text(0.5, 0.02, 'Degrees of Freedom', ha='center', fontsize=11)
    fig.text(0.02, 0.5, 'MSE', va='center', rotation=90, fontsize=11)
    
    if n_plots > 0:
        axes[0].legend(frameon=False, fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, left=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_all_plots(csv_path, output_dir=None):
    """Create all plots from CSV file"""
    df = load_results(csv_path)
    
    print(f"Loaded {len(df)} simulation results")
    print(f"Methods: {', '.join(df['method'].unique())}")
    
    # Main plot
    save_path = f"{output_dir}/mse_vs_df.png" if output_dir else None
    plot_mse_vs_df(df, save_path)
    
    # Small multiples
    save_path = f"{output_dir}/mse_by_snr.png" if output_dir else None
    plot_small_multiples_snr(df, save_path)
    
    save_path = f"{output_dir}/mse_by_aspect_ratio.png" if output_dir else None
    plot_small_multiples_aspect_ratio(df, save_path)

def create_example_data(n_simulations=500, save_csv=True, csv_filename='example_simulation_results.csv'):
    """
    Create example simulation data for testing
    
    Parameters:
    -----------
    n_simulations : int
        Number of simulation runs to generate
    save_csv : bool
        Whether to save data to CSV file
    csv_filename : str
        Name of CSV file to save
    
    Returns:
    --------
    pd.DataFrame
        Generated simulation data
    """
    np.random.seed(42)
    
    data = []
    methods = ['OLS', 'LAD', 'Huber']
    for rep in range(n_simulations):
        method = np.random.choice(methods)
        n = np.random.choice([50, 100, 200])
        p = np.random.choice([10, 20])
        df = np.random.choice([1, 3, 10, 30])
        rho = np.random.choice([0.0, 0.5, 0.8])
        snr = np.random.choice([1, 3, 5])
        
        # Simulate realistic MSE based on method characteristics
        # OLS: sensitive to outliers and heavy tails (higher MSE with low df)
        # LAD: robust to outliers (less sensitive to df)
        # Huber: balance between OLS and LAD
        base_mse = {'OLS': 1.0, 'LAD': 0.9, 'Huber': 0.85}[method]
        
        # Effect of degrees of freedom (heavy tails)
        if method == 'OLS':
            # OLS performs worse with heavy tails (low df)
            df_effect = 1 + 1.0 / df if df < 30 else 1.0
        elif method == 'LAD':
            # LAD is robust to heavy tails
            df_effect = 1 + 0.2 / df if df < 30 else 1.0
        else:  # Huber
            # Huber is intermediate
            df_effect = 1 + 0.6 / df if df < 30 else 1.0
        
        # Effect of correlation (all methods affected similarly)
        rho_effect = 1 + 0.3 * rho
        
        # Effect of SNR (inverse relationship)
        snr_effect = 1.0 / (1 + 0.3 * snr)
        
        # Effect of aspect ratio (n/p)
        aspect_ratio = n / p
        ar_effect = 1.0 / (1 + 0.1 * aspect_ratio)
        
        # Combine effects with noise
        mse = base_mse * df_effect * rho_effect * snr_effect * ar_effect
        mse += np.random.normal(0, 0.05)  # Reduced noise for more stable patterns
        mse = max(0.05, mse)  # Ensure positive MSE
        
        data.append({
            'method': method, 'n': n, 'p': p, 'df': df, 
            'rho': rho, 'SNR': snr, 'rep': rep, 'mse': mse
        })
    
    df = pd.DataFrame(data)
    
    if save_csv:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(csv_filename)
        if dir_path:  # Only create directory if there's a path component
            os.makedirs(dir_path, exist_ok=True)
        
        df.to_csv(csv_filename, index=False)
        print(f"Example data saved to '{csv_filename}'")
    
    return df

def run_complete_analysis(csv_path=None, output_dir=None, create_example=False):
    """
    Run complete simulation analysis with all plots
    
    Parameters:
    -----------
    csv_path : str, optional
        Path to CSV file with simulation results. If None and create_example=True,
        will generate example data
    output_dir : str, optional
        Directory to save plots. If None, plots will be displayed only
    create_example : bool
        Whether to create example data if csv_path is None
    
    Returns:
    --------
    pd.DataFrame
        The dataframe used for analysis
    """
    
    # Handle data loading/creation
    if csv_path is None:
        if create_example:
            print("Creating example simulation data...")
            df = create_example_data(n_simulations=500, save_csv=True, 
                                   csv_filename='artifacts/example_simulation_results.csv')
            csv_path = 'artifacts/example_simulation_results.csv'
        else:
            raise ValueError("Must provide csv_path or set create_example=True")
    else:
        print(f"Loading data from {csv_path}...")
        df = load_results(csv_path)
    
    # Print data summary
    print(f"Loaded {len(df)} simulation results")
    print(f"Methods: {', '.join(sorted(df['method'].unique()))}")
    print(f"Sample sizes (n): {sorted(df['n'].unique())}")
    print(f"Features (p): {sorted(df['p'].unique())}")
    print(f"Degrees of freedom: {sorted(df['df'].unique())}")
    print(f"SNR values: {sorted(df['SNR'].unique())}")
    print(f"Correlation values: {sorted(df['rho'].unique())}")
    
    print("\nGenerating plots...")
    
    # Create all plots
    create_all_plots(csv_path, output_dir)
    
    print("Analysis complete!")
    return df

# Example usage
if __name__ == "__main__":
    # Run complete analysis with example data
    df = run_complete_analysis(create_example=True)
    
    # Or run with your own CSV file:
    # df = run_complete_analysis('your_simulation_results.csv', output_dir='plots')