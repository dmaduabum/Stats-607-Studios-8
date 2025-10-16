import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from run_simulation import run_simulation_study

def run_full_simulation(output_file='artifacts/full_simulation_results.csv'):
    """
    Run the complete simulation study
    """
    print("Running full simulation study...")
    
    return run_simulation_study(
        n_values=[50, 100, 200],
        gamma_values=[0.2, 0.5],
        rho_values=[ 0.5, 0.8],
        df_values=[1, 5, 10],
        snr_values=[1.0, 3.0, 5.0],
        methods=['OLS', 'LAD', 'Huber'],
        n_reps=20,
        output_file=output_file,
        random_seed=42
    )


if __name__ == "__main__":

    with open('your_file_name.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    # Run the simulation study
    print("Running simulation study...")
    df = run_full_simulation()
    
    # Display summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Total simulations: {len(df)}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Sample sizes: {sorted(df['n'].unique())}")
    print(f"Features (p): {sorted(df['p'].unique())}")
    print(f"MSE range: {df['mse'].min():.4f} - {df['mse'].max():.4f}")
    print("="*60)
    
    # Optionally run evaluation
    run_eval = input("\nRun evaluation plots? (y/n): ").strip().lower()
    if run_eval in ['y', 'yes']:
        from evaluation import create_all_plots
        csv_path = 'artifacts/full_simulation_results.csv'
        create_all_plots(csv_path)
