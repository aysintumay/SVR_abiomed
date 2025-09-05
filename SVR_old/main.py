import numpy as np
import torch
import gym

import yaml
import argparse
import os

# import d4rl
import random
import json


from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from tqdm import trange
import gymnasium as gy
import pickle
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_env_data
import SVR_old.SVR as SVR
import utils
from evaluate import eval_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from noisy_mujoco.abiomed_env.rl_env import AbiomedRLEnvFactory


def custom_evaluation_metric(reward, ws, acp, air):
    """
    Calculate the custom evaluation metric: 0.3*reward + 0.3*WS - 0.2*ACP + 0.2*AIR
    """
    return 0.3 * reward + 0.3 * ws - 0.2 * acp + 0.2 * air


def train(args):
    device = torch.device(f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu")
    print(device)
    env, dataset = get_env_data(args)
    work_dir = "./runs/{}/{}/{}/alpha{}_seed{}_{}".format(
        os.getcwd().split("/")[-1],
        args.folder,
        args.env,
        args.alpha,
        args.seed,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    # Set seeds
    # env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    writer = SummaryWriter(work_dir)

    if args.env == "abiomed":
        replay_buffer = utils.ReplayBufferAbiomed(state_dim, action_dim, device=device)
        print(env.gamma1, env.gamma2, env.gamma3)
        replay_buffer.convert_abiomed(dataset, env, args.fs)

    else:
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device=device)
        replay_buffer.convert_D4RL(dataset)

    # minimal reward in all datasets of each environment
    if (args.env == "hopper") or (args.env == "Hopper"):
        Q_min = -125
    elif args.env == "halfcheetah":
        Q_min = -366
    elif args.env == "walker2d":
        Q_min = -471
    elif args.env == "pen":
        Q_min = -715
    else:
        Q_min = replay_buffer.reward.min() / (1 - args.discount)

    if not args.no_normalize:
        mean, std = replay_buffer.normalize_states()
        print("Normalizing...")
    else:
        mean, std = 0, 1

    suffix_parts = [args.env]
    if args.action:
        suffix_parts.append(f"action")
    if args.transition:
        suffix_parts.append(f"obs")
    if "synthetic" in args.config:
        suffix_parts.append("synthetic")
        suffix_parts.append(f"rate{args.noise_rate}")
        suffix_parts.append(f"scale{args.noise_scale}")
    suffix_parts.append(f"g1-{args.gamma1}_g2-{args.gamma2}_g3-{args.gamma3}")
    save_path = "_".join(suffix_parts)
    bc_model_path = f"{args.save_path}/SVR_bcmodels/bcmodel_" + save_path + ".pt"
    behav = SVR.Actor(state_dim, action_dim, max_action)
    behav.load_state_dict(torch.load(bc_model_path))
    behav.to(device)
    behav.eval()

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "replay_buffer": replay_buffer,
        "discount": args.discount,
        "tau": args.tau,
        "policy_freq": args.policy_freq,
        "schedule": not args.no_schedule,
        "Q_min": Q_min,
        "snis": args.snis,
        "behav": behav,
        "alpha": args.alpha,
        "sample_std": args.sample_std,
        "device": device,
    }
    if (
        env.gamma1 != 0.0 or env.gamma2 != 0.0 or env.gamma3 != 0.0
    ):  # turn off gammas for evaluation
        eval_env = AbiomedRLEnvFactory.create_env(
            model_name=args.model_name,
            model_path=args.model_path,
            data_path=args.data_path_wm,
            max_steps=args.max_steps,
            action_space_type=args.action_space_type,
            reward_type="smooth",
            normalize_rewards=True,
            seed=42,
            device=f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu",
        )
        print("EVAL ENVIRONMENT GAMMAS SET TO 0")
    else:
        eval_env = env
    policy = SVR.SVR(**kwargs)

    for t in trange(int(args.max_timesteps)):
        policy.train(args.batch_size, writer)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t+1}")
            d4rl_score, _ = eval_policy(
                policy,
                eval_env,
                args.env,
                mean,
                std,
                args.seed,
                eval_episodes=args.eval_episodes
                if t != int(args.max_timesteps) - 1
                else 100,
                plot=True if t == int(args.max_timesteps) - 1 else False,
                writer=writer,
            )
            writer.add_scalar("eval/reward_score", d4rl_score["avg_reward"], t)
            custom_scorer = custom_evaluation_metric(
                d4rl_score["avg_reward"],
                d4rl_score["weaning_score"],
                d4rl_score["acp"],
                d4rl_score["aggregate_air"],
            )
        # save policy
        # if (t+1) % 100000 == 0 or t == int(args.max_timesteps)-1:
    time.sleep(10)
    return policy, custom_scorer, d4rl_score, save_path


if __name__ == "__main__":
    print("Running", __file__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/train/synthetic/svr.yaml"
    )
    args, remaining_argv = parser.parse_known_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    # =========== SVR arguments ============
    parser.add_argument("--env", default="abiomed")  # OpenAI gym environment name
    parser.add_argument(
        "--seed", default=1, type=int
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--eval_freq", default=100, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument(
        "--max_timesteps", default=1000, type=int
    )  # Max time steps to run environment
    parser.add_argument(
        "--batch_size", default=256, type=int
    )  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument(
        "--policy_freq", default=2, type=int
    )  # Frequency of delayed policy updates
    parser.add_argument("--folder", default="train_rl")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--no_schedule", action="store_true")
    parser.add_argument("--snis", action="store_true")
    parser.add_argument("--alpha", default=0.008, type=float)
    parser.add_argument(
        "--sample_std", default=0.5, type=float
    )  # increase if the beta_prob is 0.0
    parser.add_argument("--devid", default=3, type=int)
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
    )

    # =========== noisy env arguments ============
    parser.add_argument(
        "--noise_rate_action",
        type=float,
        help="Portion of action to be noisy with probability",
        default=0.01,
    )
    parser.add_argument(
        "--noise_rate_transition",
        type=float,
        help="Portion of transitions to be noisy with probability",
        default=0.01,
    )
    parser.add_argument(
        "--loc", type=float, default=0.0, help="Mean of the noise distribution"
    )
    parser.add_argument(
        "--scale_action",
        type=float,
        default=0.001,
        help="Standard deviation of the action noise distribution",
    )
    parser.add_argument(
        "--scale_transition",
        type=float,
        default=0.001,
        help="Standard deviation of the transition noise distribution",
    )
    parser.add_argument(
        "--action", action="store_true", help="Create dataset with noisy actions"
    )
    parser.add_argument(
        "--transition",
        action="store_true",
        help="Create dataset with noisy transitions",
    )
    # ============ abiomed environment arguments ============
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path_wm", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--gamma1", type=float, default=0.0)
    parser.add_argument("--gamma2", type=float, default=0.0)
    parser.add_argument("--gamma3", type=float, default=0.0)
    parser.add_argument(
        "--noise_rate",
        type=float,
        help="Portion of data to be noisy with probability",
        default=0.0,
    )
    parser.add_argument(
        "--noise_scale", type=float, help="magnitude of noise", default=0.0
    )
    parser.add_argument(
        "--normalize_rewards",
        action="store_true",
        help="Normalize rewards in the Abiomed environment",
    )
    parser.add_argument(
        "--action_space_type",
        type=str,
        default="continuous",
        choices=["continuous", "discrete"],
        help="Type of action space for the environment",
    )
    parser.add_argument(
        "--fs", action="store_true", help="Use feature selection for the policy model"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/abiomed/models/policy_models/",
        help="Path to save model and results",
    )

    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)

    print("---------------------------------------")
    print(f"Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    policy, custom_metric, all_scores, save_path = train(args)
    t0 = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(os.path.join(args.save_path, "SVR", save_path)):
        os.makedirs(os.path.join(args.save_path, "SVR", save_path))
    policy.save(
        os.path.join(
            args.save_path,
            "SVR",
            save_path,
            f"svr_seed_{args.seed}_{t0}_{args.max_timesteps+1}.pth",
        )
    )
    print(
        f"Saved policy to {os.path.join(args.save_path, 'SVR', save_path, f'svr_seed_{args.seed}_{t0}_{args.max_timesteps+1}.pth')}"
    )
