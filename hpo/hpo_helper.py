#!/usr/bin/env python3
"""
Script to initialize and run wandb sweeps, and analyze results
"""

import wandb
import yaml
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import os

def create_sweep():
    """Create a new wandb sweep"""
    
    # Load sweep configuration
    sweep_config = {
        'method': 'bayes',  # Can be 'grid', 'random', 'bayes'
        'project': 'SVR_hpo',
        'metric': {
            'goal': 'maximize',
            'name': 'custom_score'
        },
        'parameters': {
            'gamma1': {
                'values': [0.05, 0.1, 0.2]
            },
            'gamma2': {
                'values': [0.2, 0.5, 0.7]
            },
            'gamma3': {
                'values': [0.1, 0.5, 1]
            }
        },
        'name': 'gamma_optimization_sweep',
        'description': 'Hyperparameter optimization for gamma1, gamma2, gamma3 parameters'
    }
    
    # Optionally add early termination
    sweep_config['early_terminate'] = {
        'type': 'hyperband',
        'min_iter': 100,
        'eta': 3
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project='SVR_hpo')
    print(f"Created sweep with ID: {sweep_id}")
    print(f"Run the following command to start agents:")
    print(f"wandb agent {sweep_id}")
    
    return sweep_id

def run_sweep_agent(sweep_id, project=None, count=None):
    """Run a sweep agent"""
    cmd = ['wandb', 'agent', sweep_id]
    cmd.extend(['--project', project])
    if count:
        cmd.extend(['--count', str(count)])
    cmd.extend(['--entity', "aysintumay-uc-san-diego"])  # Replace with your wandb entity if needed
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running sweep agent: {result.stderr}")
        return False
    
    print("Sweep agent completed successfully")
    return True

def analyze_sweep_results(project_name, sweep_id=None):
    """Analyze results from a completed sweep"""
    
    api = wandb.Api()
    
    if sweep_id:
        # Get specific sweep
        sweep = api.sweep(f"{project_name}/{sweep_id}")
        runs = sweep.runs
    else:
        # Get all runs from project
        runs = api.runs(project_name)
    
    # Extract data from runs
    results = []
    for run in runs:
        if run.state == 'finished':
            summary = run.summary._json_dict
            config = run.config
            
            result = {
                'run_id': run.id,
                'run_name': run.name,
                'gamma1': config.get('gamma1', None),
                'gamma2': config.get('gamma2', None),
                'gamma3': config.get('gamma3', None),
                'custom_score': summary.get('custom_score', None),
                'avg_reward': summary.get('avg_reward', None),
                'avg_ws': summary.get('avg_ws', None),
                'avg_acp': summary.get('avg_acp', None),
                'avg_air': summary.get('avg_air', None),
                'created_at': run.created_at,
                'runtime': run.summary.get('_runtime', None)
            }
            results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("No completed runs found!")
        return None
    
    # Sort by custom score
    df = df.sort_values('custom_score', ascending=False)
    
    print("="*80)
    print("SWEEP RESULTS ANALYSIS")
    print("="*80)
    print(f"Total completed runs: {len(df)}")
    print(f"Best custom score: {df.iloc[0]['custom_score']:.4f}")
    print(f"Best parameters:")
    print(f"  gamma1: {df.iloc[0]['gamma1']:.4f}")
    print(f"  gamma2: {df.iloc[0]['gamma2']:.4f}")
    print(f"  gamma3: {df.iloc[0]['gamma3']:.4f}")
    print(f"Best run ID: {df.iloc[0]['run_id']}")
    
    print("\nTop 5 results:")
    print(df[['gamma1', 'gamma2', 'gamma3', 'custom_score', 'avg_reward', 'avg_ws', 'avg_acp', 'avg_air']].head().to_string(index=False))
    
    # Create visualizations
    create_sweep_visualizations(df, project_name)
    
    # Save results
    output_dir = Path("sweep_results")
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / f"{project_name}_sweep_results.csv", index=False)
    print(f"\nResults saved to: {output_dir / f'{project_name}_sweep_results.csv'}")
    
    return df

def create_sweep_visualizations(df, project_name):
    """Create visualizations for sweep results"""
    
    output_dir = Path("sweep_results")
    output_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Hyperparameter Sweep Results - {project_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Custom Score Distribution
    axes[0, 0].hist(df['custom_score'].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Custom Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Custom Score Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Gamma1 vs Custom Score
    scatter = axes[0, 1].scatter(df['gamma1'], df['custom_score'], 
                                c=df['custom_score'], cmap='viridis', alpha=0.7, s=60)
    axes[0, 1].set_xlabel('Gamma1')
    axes[0, 1].set_ylabel('Custom Score')
    axes[0, 1].set_title('Custom Score vs Gamma1')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # Plot 3: Gamma2 vs Custom Score
    scatter = axes[0, 2].scatter(df['gamma2'], df['custom_score'], 
                                c=df['custom_score'], cmap='viridis', alpha=0.7, s=60)
    axes[0, 2].set_xlabel('Gamma2')
    axes[0, 2].set_ylabel('Custom Score')
    axes[0, 2].set_title('Custom Score vs Gamma2')
    axes[0, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 2])
    
    # Plot 4: Gamma3 vs Custom Score
    scatter = axes[1, 0].scatter(df['gamma3'], df['custom_score'], 
                                c=df['custom_score'], cmap='viridis', alpha=0.7, s=60)
    axes[1, 0].set_xlabel('Gamma3')
    axes[1, 0].set_ylabel('Custom Score')
    axes[1, 0].set_title('Custom Score vs Gamma3')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # Plot 5: Component Analysis (Top 10 runs)
    top_10 = df.head(10)
    x = range(len(top_10))
    width = 0.2
    
    axes[1, 1].bar([i - 1.5*width for i in x], top_10['avg_reward'], width, label='Reward', alpha=0.8)
    axes[1, 1].bar([i - 0.5*width for i in x], top_10['avg_ws'], width, label='WS', alpha=0.8)
    axes[1, 1].bar([i + 0.5*width for i in x], top_10['avg_acp'], width, label='ACP', alpha=0.8)
    axes[1, 1].bar([i + 1.5*width for i in x], top_10['avg_air'], width, label='AIR', alpha=0.8)
    
    axes[1, 1].set_xlabel('Top 10 Runs (Rank)')
    axes[1, 1].set_ylabel('Metric Values')
    axes[1, 1].set_title('Metric Components (Top 10 Runs)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Parameter Correlation Heatmap
    param_cols = ['gamma1', 'gamma2', 'gamma3', 'custom_score', 'avg_reward', 'avg_ws', 'avg_acp', 'avg_air']
    corr_data = df[param_cols].corr()
    
    im = axes[1, 2].imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 2].set_xticks(range(len(param_cols)))
    axes[1, 2].set_yticks(range(len(param_cols)))
    axes[1, 2].set_xticklabels(param_cols, rotation=45)
    axes[1, 2].set_yticklabels(param_cols)
    axes[1, 2].set_title('Parameter Correlation Matrix')
    
    # Add correlation values as text
    for i in range(len(param_cols)):
        for j in range(len(param_cols)):
            text = axes[1, 2].text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{project_name}_sweep_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate 3D scatter plot for parameter space
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(df['gamma1'], df['gamma2'], df['gamma3'], 
                        c=df['custom_score'], cmap='viridis', s=100, alpha=0.7)
    
    ax.set_xlabel('Gamma1')
    ax.set_ylabel('Gamma2')
    ax.set_zlabel('Gamma3')
    ax.set_title('Parameter Space Exploration (Color = Custom Score)')
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Custom Score')
    
    plt.savefig(output_dir / f"{project_name}_parameter_space_3d.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Wandb sweep runner and analyzer")
    parser.add_argument("--action", choices=["create", "run", "analyze"], required=True,
                       help="Action to perform")
    parser.add_argument("--sweep_id", type=str, help="Sweep ID (required for run and analyze)")
    parser.add_argument("--project", type=str, default="SVR_hpo", help="Wandb project name")
    parser.add_argument("--count", type=int, help="Number of runs for agent (optional)")
    
    args = parser.parse_args()
    
    if args.action == "create":
        args.sweep_id = create_sweep()
        
    elif args.action == "run":
        if not args.sweep_id:
            print("Error: --sweep_id is required for run action")
            return
        run_sweep_agent(args.sweep_id, args.project, args.count)
        
    elif args.action == "analyze":
        analyze_sweep_results(args.project, args.sweep_id)

if __name__ == "__main__":
    main()