import numpy as np
import torch
import gym
import yaml
import argparse
import os
import random
import json
import wandb
from SVR_old.SVR import SVR
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from tqdm import trange
import gymnasium as gy
import pickle
import sys
import matplotlib.pyplot as plt
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SVR_old.main import train
from utils import get_env_data
import utils
from evaluate import eval_policy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_single_trial():
    """
    Run a single trial with wandb sweep parameters
    """
    # Initialize wandb run
    wandb.init()
    
    # Get sweep parameters
    config = wandb.config
    
    print(f"Running trial with gamma1={config.gamma1}, gamma2={config.gamma2}, gamma3={config.gamma3}")
    
    # Parse base arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/train/svr/abiomed.yaml")
    
    # Add all the arguments from your original script
    parser.add_argument("--env", default="abiomed")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--max_timesteps", default=1000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99)
    parser.add_argument("--tau", default=0.005)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument('--folder', default='train_rl')
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument('--no_schedule', action="store_true")
    parser.add_argument('--snis', action="store_true")
    parser.add_argument("--alpha", default=0.008, type=float)
    parser.add_argument('--sample_std', default=0.5, type=float)
    parser.add_argument('--devid', default=4, type=int)
    parser.add_argument("--data_path", type=str, default="/abiomed/intermediate_data_d4rl/farama_sac_expert/Hopper-v2_expert_1000.pkl")
    
    # Noise arguments
    parser.add_argument("--noise_rate_action", type=float, default=0.01)
    parser.add_argument("--noise_rate_transition", type=float, default=0.01)
    parser.add_argument("--loc", type=float, default=0.0)
    parser.add_argument("--scale_action", type=float, default=0.001)
    parser.add_argument("--scale_transition", type=float, default=0.001)
    parser.add_argument("--action", action='store_true')
    parser.add_argument("--transition", action='store_true')
    
    # Abiomed arguments
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path_wm", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--gamma1", type=float, default=0.0)
    parser.add_argument("--gamma2", type=float, default=0.0)
    parser.add_argument("--gamma3", type=float, default=0.0)
    parser.add_argument("--normalize_rewards", action='store_true')
    parser.add_argument("--action_space_type", type=str, default="continuous", choices=["continuous", "discrete"])
    parser.add_argument('--fs', action="store_true")
    parser.add_argument('--save_path', type=str, default='/abiomed/models/policy_models/')
    
    # Parse arguments
    args = parser.parse_args([])  # Empty list since we're not using command line args
    
    # Load config if exists
    if hasattr(args, 'config') and args.config:
        try:
            with open(args.config, 'r') as f:
                file_config = yaml.safe_load(f)
                for key, value in file_config.items():
                    if hasattr(args, key):
                        setattr(args, key, value)
        except FileNotFoundError:
            logger.warning(f"Config file {args.config} not found, using defaults")
    
    # Override with sweep parameters
    args.gamma1 = config.gamma1
    args.gamma2 = config.gamma2
    args.gamma3 = config.gamma3
    
    # Log the configuration
    wandb.config.update({
        "env": args.env,
        "seed": args.seed,
        "max_timesteps": args.max_timesteps,
        "batch_size": args.batch_size,
        "discount": args.discount,
        "tau": args.tau,
        "alpha": args.alpha,
        "sample_std": args.sample_std
    })
    
    try:
        # Run training
        logger.info(f"Starting training with gamma1={args.gamma1}, gamma2={args.gamma2}, gamma3={args.gamma3}")
        policy, custom_metric, metrics, save_path = train(args)
        
        # Create unique model save path
        t0 = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        trial_id = wandb.run.name  # Use wandb run name as trial ID
        
        model_dir = os.path.join(args.save_path, "SVR_hpo_sweep", save_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        model_path = os.path.join(
            model_dir, 
            f"svr_sweep_{trial_id}_{t0}_gamma1_{args.gamma1}_gamma2_{args.gamma2}_gamma3_{args.gamma3}.pth"
        )
        
        # Save the model
        policy.save(model_path)
        logger.info(f"Saved policy to {model_path}")
        
        # Log results to wandb
        results_to_log = {
            'custom_score': custom_metric,
            'avg_reward': metrics['avg_reward'],
            'avg_ws': metrics['weaning_score'],
            'avg_acp': metrics['acp'],
            'avg_air': metrics['aggregate_air'],
            'gamma1': args.gamma1,
            'gamma2': args.gamma2,
            'gamma3': args.gamma3,
            'model_path': model_path
        }
        
        # Log all metrics to wandb
        wandb.log(results_to_log)
        
        # Log model as artifact
        model_artifact = wandb.Artifact(
            f"model_{trial_id}", 
            type="model",
            metadata={
                "gamma1": args.gamma1,
                "gamma2": args.gamma2, 
                "gamma3": args.gamma3,
                "custom_score": custom_metric
            }
        )
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)
        
        # Create summary table
        summary_data = [[
            args.gamma1, args.gamma2, args.gamma3, 
            custom_metric, metrics['avg_reward'], 
            metrics['weaning_score'], metrics['acp'], metrics['aggregate_air']
        ]]
        
        summary_table = wandb.Table(
            data=summary_data,
            columns=["gamma1", "gamma2", "gamma3", "custom_score", "avg_reward", "avg_ws", "avg_acp", "avg_air"]
        )
        wandb.log({"trial_summary": summary_table})
        
        logger.info(f"Trial completed successfully. Custom score: {custom_metric:.4f}")
        
    except Exception as e:
        logger.error(f"Trial failed with error: {str(e)}")
        # Log the failure to wandb
        wandb.log({"trial_status": "failed", "error": str(e)})
        raise e
    
    finally:
        # Finish the wandb run
        wandb.finish()

if __name__ == "__main__":
    run_single_trial()