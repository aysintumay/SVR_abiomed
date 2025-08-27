import numpy as np
import torch
import gym

import yaml
import argparse
import os
# import d4rl
import random
import json
import wandb
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from tqdm import trange
import gymnasium as gy
import pickle
import sys
import matplotlib.pyplot as plt
from itertools import product
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SVR_old.main import train
from utils import get_env_data
import utils
from evaluate import eval_policy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from noisy_mujoco.abiomed_env.rl_env import AbiomedRLEnvFactory

def create_search_visualization(results_df, save_path):
    """
    Create visualizations of the hyperparameter search results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Custom Score vs Trial
    axes[0, 0].plot(results_df.index, results_df['custom_score'], 'b-', alpha=0.7)
    axes[0, 0].scatter(results_df.index, results_df['custom_score'], alpha=0.5)
    axes[0, 0].set_xlabel('Trial')
    axes[0, 0].set_ylabel('Custom Score')
    axes[0, 0].set_title('Custom Score vs Trial')
    axes[0, 0].grid(True)
    
    # Plot 2: Score vs gamma1
    axes[0, 1].scatter(results_df['gamma1'], results_df['custom_score'], alpha=0.6)
    axes[0, 1].set_xlabel('Gamma1')
    axes[0, 1].set_ylabel('Custom Score')
    axes[0, 1].set_title('Custom Score vs Gamma1')
    axes[0, 1].grid(True)
    
    # Plot 3: Score vs gamma2
    axes[0, 2].scatter(results_df['gamma2'], results_df['custom_score'], alpha=0.6)
    axes[0, 2].set_xlabel('Gamma2')
    axes[0, 2].set_ylabel('Custom Score')
    axes[0, 2].set_title('Custom Score vs Gamma2')
    axes[0, 2].grid(True)
    
    # Plot 4: Score vs gamma3
    axes[1, 0].scatter(results_df['gamma3'], results_df['custom_score'], alpha=0.6)
    axes[1, 0].set_xlabel('Gamma3')
    axes[1, 0].set_ylabel('Custom Score')
    axes[1, 0].set_title('Custom Score vs Gamma3')
    axes[1, 0].grid(True)
    
    # Plot 5: Component breakdown for top result
    top_result = results_df.iloc[0]
    components = ['Reward', 'WS', 'ACP', 'AIR']
    values = [top_result['avg_reward'], top_result['avg_ws'], top_result['avg_acp'], top_result['avg_air']]
    weights = [0.3, 0.3, 0.2, 0.2]
    weighted_values = [v * w for v, w in zip(values, weights)]
    
    x_pos = np.arange(len(components))
    axes[1, 1].bar(x_pos, values, alpha=0.7, label='Raw Values')
    axes[1, 1].bar(x_pos, weighted_values, alpha=0.7, label='Weighted Values')
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Values')
    axes[1, 1].set_title('Best Model Components')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(components)
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot 6: Top 10 scores
    top_10 = results_df.head(10)
    axes[1, 2].barh(range(len(top_10)), top_10['custom_score'])
    axes[1, 2].set_yticks(range(len(top_10)))
    axes[1, 2].set_yticklabels([f"T{row['trial_id']}" for _, row in top_10.iterrows()])
    axes[1, 2].set_xlabel('Custom Score')
    axes[1, 2].set_title('Top 10 Trials')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'hyperparameter_search_results.png'), dpi=300, bbox_inches='tight')
    wandb.log({"hyperparameter_search_results": wandb.Image(plt)})
    plt.close()


def run_hyperparameter_search(args, gamma_ranges):
    """
    Run hyperparameter search over gamma values
    """
    
    # Generate all combinations of gamma values
    # gamma1_values = np.linspace(gamma_ranges['gamma1'][0], gamma_ranges['gamma1'][1], gamma_ranges['gamma1'][2])
    # gamma2_values = np.linspace(gamma_ranges['gamma2'][0], gamma_ranges['gamma2'][1], gamma_ranges['gamma2'][2])
    # gamma3_values = np.linspace(gamma_ranges['gamma3'][0], gamma_ranges['gamma3'][1], gamma_ranges['gamma3'][2])
    
    gamma_combinations = list(product(np.array(gamma_ranges['gamma1']), np.array(gamma_ranges['gamma2']), np.array(gamma_ranges['gamma3'])))
    
    (f"Total combinations to evaluate: {len(gamma_combinations)}")
    
    results = []
    best_score = float('-inf')
    best_params = None
    best_model_path = None
    t0 = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for trial_id, (gamma1, gamma2, gamma3) in enumerate(gamma_combinations):
        try:
            # Create a copy of args for this trial
            trial_args = argparse.Namespace(**vars(args))
            trial_args.gamma1 = float(gamma1)
            trial_args.gamma2 = float(gamma2)
            trial_args.gamma3 = float(gamma3)
            
            run = wandb.init(
                project="SVR_hpo",
                group=f"SVR_{t0}",
                name=f"trial_{trial_id}_gamma1_{trial_args.gamma1}_gamma2_{trial_args.gamma2}_gamma3_{trial_args.gamma3}",
                config={
                    "gamma1": args.gamma1,
                    "gamma2": args.gamma2,
                    "gamma3": args.gamma3,
                    "trial_id": trial_id,
                    "seed": trial_args.seed,
                    "env": trial_args.env,
                },
                reinit=True,
            )
            
            print(f"Starting trial {trial_id} with gamma1={gamma1}, gamma2={gamma2}, gamma3={gamma3}")
            policy, custom_metric, metrics, save_path = train(trial_args)
            if not os.path.exists(os.path.join(args.save_path, "SVR_hpo", save_path, t0)):
                os.makedirs(os.path.join(args.save_path, "SVR_hpo", save_path, t0))
            model_path = os.path.join(args.save_path, "SVR_hpo", save_path, t0, f"svr_seed_{args.seed}_{trial_id}_gamma1_{gamma1}_gamma2_{gamma2}_gamma3_{gamma3}.pth")
            policy.save(model_path)
            print(f"Saved policy to {model_path}")
            wandb.log({
                'custom_score': custom_metric,
                'avg_reward': metrics['avg_reward'],
                'avg_ws': metrics['weaning_score'],
                'avg_acp': metrics['acp'],
                'avg_air': metrics['aggregate_air'],
            })

            
            result = {
                'trial_id': trial_id,
                'gamma1': gamma1,
                'gamma2': gamma2,
                'gamma3': gamma3,
                'custom_score': custom_metric,
                'avg_reward': metrics['avg_reward'],
                'avg_ws': metrics['weaning_score'],
                'avg_acp': metrics['acp'],
                'avg_air': metrics['aggregate_air'],
                'model_path': model_path
            }
            
            results.append(result)
            
            if custom_metric > best_score:
                best_score = custom_metric
                best_params = (gamma1, gamma2, gamma3)
                best_model_path = model_path
            
            wandb.run.summary["best_custom_score"] = best_score
            wandb.run.summary["best_params"] = {
                "gamma1": best_params[0],
                "gamma2": best_params[1],
                "gamma3": best_params[2],
            }
            
            
        except Exception as e:
            print(f"Trial {trial_id} failed: {str(e)}")
            wandb.alert(title="Trial Failed", text=f"Trial {trial_id} failed:\n{e}")
            continue
    
    # Save results
    result_path = os.path.join(args.save_path, "SVR_hpo", save_path, t0)
    results_df = pd.DataFrame(results)
    results_path = os.path.join(result_path, "hyperparameter_search_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Sort by custom score and display top results

    results_df_sorted = results_df.sort_values('custom_score', ascending=False)
    
    print("=" * 80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 80)
    print(f"Best Parameters: gamma1={best_params[0]:.4f}, gamma2={best_params[1]:.4f}, gamma3={best_params[2]:.4f}")
    print(f"Best Score: {best_score:.4f}")
    print(f"Best Model Path: {best_model_path}")
    print("\nTop 5 Results:")
    print(results_df_sorted.head().to_string(index=False))
    
    # Create visualization
    

    create_search_visualization(results_df_sorted, result_path)
    wandb.run.summary["best_model_path"] = best_model_path
    run.finish()
    return results_df_sorted, best_params, best_model_path 

if __name__ == "__main__":
    print("Running", __file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/train/svr/abiomed.yaml")
    args, remaining_argv = parser.parse_known_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    #=========== SVR arguments ============
    parser.add_argument("--env", default="abiomed")        # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=100, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--max_timesteps", default=1000, type=int)   # Max time steps to run environment
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument('--folder', default='train_rl')
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument('--no_schedule', action="store_true")
    parser.add_argument('--snis', action="store_true")
    parser.add_argument("--alpha", default=0.008, type=float)
    parser.add_argument('--sample_std', default=0.5, type=float) #increase if the beta_prob is 0.0
    parser.add_argument('--devid', default=6, type=int)
    parser.add_argument("--data_path", type=str, default="/abiomed/intermediate_data_d4rl/farama_sac_expert/Hopper-v2_expert_1000.pkl")

    #=========== noisy env arguments ============
    parser.add_argument("--noise_rate_action", type=float, help="Portion of action to be noisy with probability", default=0.01)
    parser.add_argument("--noise_rate_transition", type=float, help="Portion of transitions to be noisy with probability", default=0.01)
    parser.add_argument("--loc", type=float, default=0.0, help="Mean of the noise distribution")
    parser.add_argument("--scale_action", type=float, default=0.001, help="Standard deviation of the action noise distribution")
    parser.add_argument("--scale_transition", type=float, default=0.001, help="Standard deviation of the transition noise distribution")
    parser.add_argument("--action", action='store_true', help="Create dataset with noisy actions")
    parser.add_argument("--transition", action='store_true', help="Create dataset with noisy transitions")
    #============ abiomed environment arguments ============
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path_wm", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--gamma1", type=float, nargs='+', default=[0.05, 0.1, 0.2])
    parser.add_argument("--gamma2", type=float, nargs='+', default=[0.2, 0.5, 0.7])
    parser.add_argument("--gamma3", type=float, nargs='+', default=[0.1, 0.5, 1])
    # parser.add_argument("--gamma1", type=float, nargs='+', default=[0.05])
    # parser.add_argument("--gamma2", type=float, nargs='+', default=[0.2])
    # parser.add_argument("--gamma3", type=float, nargs='+', default=[0.1, 0.5])


    parser.add_argument("--normalize_rewards", action='store_true', help="Normalize rewards in the Abiomed environment")
    parser.add_argument("--action_space_type", type=str, default="continuous", choices=["continuous", "discrete"], help="Type of action space for the environment") 
    parser.add_argument('--fs', action= "store_true", help= "Use feature selection for the policy model")
    parser.add_argument('--save_path', type=str, default='/abiomed/models/policy_models/', help='Path to save model and results')

    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)

    device = torch.device(f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu")
    print(device)
    print("---------------------------------------")
    print(f"Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    #start wandb logger

    run = wandb.init(
                project='SVR_hpo',
                group="SVR",
                config=vars(args),
                )
    
    gamma_ranges = {
        'gamma1': args.gamma1, #ACP: max 8 min 0
        'gamma2': args.gamma2, #WS: max 2 min -1
        'gamma3': args.gamma3 #AIR: max 1 min 0
    }

    results_df, best_params, best_model_path = run_hyperparameter_search(args, gamma_ranges)