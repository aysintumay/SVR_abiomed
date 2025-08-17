import numpy as np
import torch
import gym
from utils import get_env_data
import yaml

import argparse
import os
# import d4rl
import random
import json
import utils
# import SVR_old.SVR as SVR
import SVR_discrete.SVR as SVR_discrete
import SVR_old.SVR as SVR
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from tqdm import trange
import gymnasium as gy
import pickle
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from noisy_mujoco.abiomed_env.cost_func import (compute_acp_cost,
                                                overall_acp_cost,
                                                compute_map_model_air,
                                                compute_hr_model_air,
                                                compute_pulsatility_model_air,
                                                aggregate_air_model,
                                                weaning_score_model,
                                                unstable_percentage_model,
                                                    super_metric
                                                    )

def get_svr(args):
    
    device = torch.device(f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu")

    env, dataset = get_env_data(args)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])


    if args.env == "abiomed":
        replay_buffer = utils.ReplayBufferAbiomed(state_dim, action_dim, device=device)
        replay_buffer.convert_abiomed(dataset, env)

    else:
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device=device)
        replay_buffer.convert_D4RL(dataset)

    # minimal reward in all datasets of each environment
    if (args.env =="hopper") or (args.env =="Hopper" ):
        Q_min = -125
    elif args.env == "halfcheetah":
        Q_min = -366
    elif args.env == "walker2d":
        Q_min = -471
    elif args.env ==  "pen":
        Q_min = -715
    else:
        Q_min = replay_buffer.reward.min() / (1 - args.discount)

    

    suffix_parts = [args.env]
    if args.action:
        suffix_parts.append(f"action")
    if args.transition:
        suffix_parts.append(f"obs")
    save_path = "_".join(suffix_parts)
    bc_model_path=f'{args.save_path}/SVR_bcmodels/bcmodel_'+save_path+'.pt'
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

    policy = SVR.SVR(**kwargs) 
    
    filepath = args.policy_path
    print('ENV NAME IS', args.env)
    print("Loading policy from", filepath)
    policy.load(filepath)
    
    
    return policy, replay_buffer, env

def get_bc(args):
    device = torch.device(f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu")

    env, dataset = get_env_data(args)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if env.action_space.__class__.__name__ == 'Box' else env.action_space.n
    max_action = float(env.action_space.high[0]) if env.action_space.__class__.__name__ == 'Box' else env.action_space.n + 1

    if args.env == "abiomed":
        replay_buffer = utils.ReplayBufferAbiomed(state_dim, action_dim, device=device)
        replay_buffer.convert_abiomed(dataset, env)

    else:
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device=device)
        replay_buffer.convert_D4RL(dataset)


    suffix_parts = [args.env]
    if args.action:
        suffix_parts.append(f"action")
    if args.transition:
        suffix_parts.append(f"obs")
    save_path = "_".join(suffix_parts)
    if args.action_space_type == "discrete":
        bc_model_path=f'{args.save_path}SVR_discrete_bcmodels/bcmodel_'+save_path+'.pt'
    else:
        bc_model_path=f'{args.save_path}SVR_bcmodels/bcmodel_'+save_path+'.pt'
    print("BC model path:", bc_model_path)
    if args.action_space_type == "continuous":
        behav = SVR.Actor(state_dim, action_dim, max_action)
    else:
        behav = SVR_discrete.DiscreteActor(state_dim, action_dim)
    behav.load_state_dict(torch.load(bc_model_path))
    behav.to(device)
    behav.device = device
    return behav, replay_buffer, env


def eval_policy(policy, eval_env, env_name, mean, std, seed_offset=100,
                eval_episodes=10, plot=None, writer=None):
    """
    Evaluates policy and computes ACP, AIR metrics, Weaning score (abiomed env).
    Requires:
      - eval_env.episode_actions (list of actions from last episode)
      - compute_* functions available in scope
      - eval_env (or env) exposing `world_model` used by the AIR functions
    """

    if env_name == 'abiomed':
        avg_reward = 0.0
        avg_acp = 0.0

        # ---- aggregated metrics over episodes ----
        total_map_air_sum = 0.0
        total_hr_air_sum = 0.0
        total_pulsatility_air_sum = 0.0
        total_aggregate_air_sum = 0.0
        total_unstable_percentage_sum = 0.0
        total_super_sum = 0.0
        wean_score_sum = 0.0


        for k in range(eval_episodes):
            ep_states = []           # store normalized states per step (like in _evaluate)
            (state, info), done = eval_env.reset(), False  # state is normalized
            all_states = info['all_states']                # normalized
            all_states = np.concatenate([state.reshape(1, -1), all_states], axis=0)
            truncated = False

            while not (done or truncated):
                # normalize state before policy (kept from your original code)
                s_norm = (np.array(state).reshape(1, -1) - mean) / std
                action = policy.select_action(s_norm)      # action in env's range

                next_state, reward, done, truncated, _ = eval_env.step(action)
                avg_reward += reward

                ep_states.append(state)   # store the *current* obs like _evaluate
                state = next_state
            

            # ---- per-episode metrics (same as _evaluate) ----
            # ACP (per-timestep across the episode)
            avg_acp += overall_acp_cost([eval_env.episode_actions])
            

            # Convert states list to numpy for AIR models
            ep_states_np = np.asarray(ep_states, dtype=np.float32)

            # NOTE: _evaluate uses env.world_model; if that's actually eval_env.world_model,
            # change the following `env.world_model` to `eval_env.world_model`.
            wm = getattr(eval_env, 'world_model', None)
            if wm is None:
                wm = env.world_model  # fallback to global `env` if that's how you access it

            # AIR metrics
            total_map_air_sum          += compute_map_model_air(wm, ep_states_np, eval_env.episode_actions)
            total_hr_air_sum           += compute_hr_model_air(wm, ep_states_np, eval_env.episode_actions)
            total_pulsatility_air_sum  += compute_pulsatility_model_air(wm, ep_states_np, eval_env.episode_actions)
            total_aggregate_air_sum    += aggregate_air_model(wm, ep_states_np, eval_env.episode_actions)
            total_super_sum            += super_metric(wm, ep_states_np, eval_env.episode_actions)

            # Weaning + unstable percentage
            wean_score_sum             += weaning_score_model(wm, ep_states_np, eval_env.episode_actions)
            total_unstable_percentage_sum += unstable_percentage_model(wm, ep_states_np)

            next_state_l = ep_states.copy()
            next_state_l.append(state)
            if (k == 2) and plot:
                utils.plot_policy(eval_env, next_state_l[1:], all_states, writer)

        # ---- episode averages ----
        avg_reward /= eval_episodes
        acp_mean = avg_acp / eval_episodes

        map_air_mean          = total_map_air_sum / eval_episodes
        hr_air_mean           = total_hr_air_sum / eval_episodes
        puls_air_mean         = total_pulsatility_air_sum / eval_episodes
        aggregate_air_mean    = total_aggregate_air_sum / eval_episodes
        unstable_hours_mean   = total_unstable_percentage_sum / eval_episodes
        weaning_score_mean    = wean_score_sum / eval_episodes
        super_mean            = total_super_sum / eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: "
              f"Return {avg_reward:.3f}")
        print(f"ACP {acp_mean:.4f}")
        print(f"MAP AIR/ep: {map_air_mean:.5f} | HR AIR/ep: {hr_air_mean:.5f} "
              f"| Pulsatility AIR/ep: {puls_air_mean:.5f}")
        print(f"Aggregate AIR/ep: {aggregate_air_mean:.5f}")
        print(f"Unstable hours (%): {unstable_hours_mean}")
        print(f"Weaning score: {weaning_score_mean}")
        print(f"Super metric: {super_mean:.5f}")
        print("---------------------------------------")

        return {
            "avg_reward": avg_reward,
            "acp": acp_mean,
            "map_air": map_air_mean,
            "hr_air": hr_air_mean,
            "pulsatility_air": puls_air_mean,
            "aggregate_air": aggregate_air_mean,
            "unstable_hours_pct": unstable_hours_mean,
            "weaning_score": weaning_score_mean,
            "super_metric": super_mean,
        }

    else:

        avg_reward = 0.
        for _ in range(eval_episodes):
            (state, _), done = eval_env.reset(), False
            truncated = False
            
            while not (done or truncated):
                state = (np.array(state).reshape(1,-1) - mean)/std
                action = policy.select_action(state)
                state, reward, done, truncated, _ = eval_env.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {avg_reward:.3f}")
        print("---------------------------------------")
        return {"avg_reward": avg_reward}

def eval_bc(actor, eval_env, env_name, mean, std, writer=None, seed_offset=100, eval_episodes=10, plot=None):
    
    if env_name == 'abiomed':
        avg_reward = 0.0
        avg_acp = 0.0

        # ---- aggregated metrics over episodes ----
        total_map_air = 0.0
        total_hr_air = 0.0
        total_puls_air = 0.0
        total_agg_air = 0.0
        total_unstable_pct = 0.0
        total_wean_score = 0.0
        total_super_sum = 0.0

        for k in range(eval_episodes):
            ep_states = []  # store normalized states over the episode (before step)
            
            (state, info), done = eval_env.reset(), False  # state is normalized
            all_states = info['all_states']                # normalized
            all_states = np.concatenate([state.reshape(1, -1), all_states], axis=0)
            truncated = False

            while not (done or truncated):
                s_norm = (np.array(state).reshape(1, -1) - mean) / std
                eval_states_tensor = torch.tensor(s_norm, dtype=torch.float32, device=actor.device)

                if eval_env.action_space_type == "discrete":
                    action, _ = actor(eval_states_tensor, deterministic=True)
                else:
                    action = actor(eval_states_tensor)

                ep_states.append(state) 
                next_state, reward, done, truncated, _ = eval_env.step(action)
                avg_reward += reward
                state = next_state
            next_state_l = ep_states.copy()
            next_state_l.append(state)
            # optional plot
            if (k == 3) and plot:
                utils.plot_policy(eval_env, next_state_l[1:], all_states, writer)

            # ---- per-episode metrics ----
            avg_acp += overall_acp_cost([eval_env.episode_actions])

            ep_states_np = np.asarray(ep_states, dtype=np.float32)
            wm = getattr(eval_env, 'world_model', None)  # prefer eval_env.world_model
            if wm is None:
                wm = env.world_model  # fallback if stored globally

            total_map_air        += compute_map_model_air(wm, ep_states_np, eval_env.episode_actions)
            total_hr_air         += compute_hr_model_air(wm, ep_states_np, eval_env.episode_actions)
            total_puls_air       += compute_pulsatility_model_air(wm, ep_states_np, eval_env.episode_actions)
            total_agg_air        += aggregate_air_model(wm, ep_states_np, eval_env.episode_actions)
            total_wean_score     += weaning_score_model(wm, ep_states_np, eval_env.episode_actions)
            total_unstable_pct   += unstable_percentage_model(wm, ep_states_np)
            total_super_sum            += super_metric(wm, ep_states_np, eval_env.episode_actions)

        # ---- averages over episodes ----
        avg_reward /= eval_episodes
        acp = avg_acp / eval_episodes

        map_air_avg      = total_map_air / eval_episodes
        hr_air_avg       = total_hr_air / eval_episodes
        puls_air_avg     = total_puls_air / eval_episodes
        agg_air_avg      = total_agg_air / eval_episodes
        unstable_avg_pct = total_unstable_pct / eval_episodes
        wean_avg         = total_wean_score / eval_episodes
        super_avg        = total_super_sum / eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes:")
        print(f"  Return: {avg_reward:.3f}")
        print(f"  ACP score: {acp:.4f}")
        print(f"  MAP AIR/ep: {map_air_avg:.5f} | HR AIR/ep: {hr_air_avg:.5f} | "
            f"Pulsatility AIR/ep: {puls_air_avg:.5f}")
        print(f"  Aggregate AIR/ep: {agg_air_avg:.5f}")
        print(f"  Unstable hours (%): {unstable_avg_pct:.3f}")
        print(f"  Weaning score: {wean_avg:.5f}")
        print(f"  Super metric: {super_avg:.5f}")
        print("---------------------------------------")
        return {
            "avg_reward": avg_reward,
            "acp": acp,
            "map_air": map_air_avg,
            "hr_air": hr_air_avg,
            "pulsatility_air": puls_air_avg,
            "aggregate_air": agg_air_avg,
            "unstable_hours_pct": unstable_avg_pct,
            "weaning_score": wean_avg,
            "super_metric": super_avg,
        }
    else:
        # eval_env = gym.make(env_name)

        # eval_env.seed(seed + seed_offset)
        # eval_env.action_space.seed(seed + seed_offset)
        avg_reward = 0.
        for _ in range(eval_episodes):
            (state, _), done = eval_env.reset(), False
            truncated = False
            
            while not (done or truncated):
                state = (np.array(state).reshape(1,-1) - mean)/std
                
                eval_states_tensor = torch.Tensor(state).to(actor.device)
                action = actor(eval_states_tensor)
                state, reward, done, truncated, _ = eval_env.step(action.cpu().detach().numpy().squeeze())
                avg_reward += reward

        avg_reward /= eval_episodes
        # d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {avg_reward:.3f}")
        print("---------------------------------------")
    return {"avg_reward": avg_reward,}

if __name__ == "__main__":
    print("Running", __file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/evaluation/bc/abiomed.yaml")
    args, remaining_argv = parser.parse_known_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    #=========== SVR arguments ============
    parser.add_argument("--env", default="abiomed")        # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
    # parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument('--snis', action="store_true")
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument('--sample_std', default=0.2, type=float)

    parser.add_argument('--folder', default='train_rl')
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument('--no_schedule', action="store_true")
    parser.add_argument('--devid', default=5, type=int)
    parser.add_argument("--policy_path", type=str, default="/abiomed/models/policy_models/SVR_kde/abiomed/svr_kde_seed_1_2025-08-01_09-03-42_130000.pth")

    # #=========== noisy env arguments ============
    parser.add_argument("--noise_rate_action", type=float, help="Portion of action to be noisy with probability", default=0.01)
    parser.add_argument("--noise_rate_transition", type=float, help="Portion of transitions to be noisy with probability", default=0.01)
    parser.add_argument("--loc", type=float, default=0.0, help="Mean of the noise distribution")
    parser.add_argument("--scale_action", type=float, default=0.001, help="Standard deviation of the action noise distribution")
    parser.add_argument("--scale_transition", type=float, default=0.001, help="Standard deviation of the transition noise distribution")
    parser.add_argument("--action", action='store_true', help="Create dataset with noisy actions")
    parser.add_argument("--transition", action='store_true', help="Create dataset with noisy transitions")

    # #============ abiomed environment arguments ============
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path_wm", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--normalize_rewards", action='store_true', help="Normalize rewards in the Abiomed environment")

    parser.add_argument("--action_space_type", type=str, default="continuous", choices=["continuous", "discrete"], help="Type of action space for the environment") 

    parser.add_argument('--save_path', type=str, default='/abiomed/models/policy_models/', help='Path to save model and results')
    #=========== SVR arguments ============
    parser.add_argument("--model_name_rl", type=str, default="bc", choices=["svr", "bc"], help="RL model to use for evaluation")

    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)

    device = torch.device(f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu")
    print(device)
    print("---------------------------------------")
    print(f"Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    work_dir = './runs/{}/{}/{}/alpha{}_seed{}_{}'.format(
        os.getcwd().split('/')[-1], args.folder, args.env, args.alpha, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    writer = SummaryWriter(work_dir)

    if args.model_name_rl == "svr":
        policy, replay_buffer, env = get_svr(args)
    else:
        policy, replay_buffer, env = get_bc(args)

    if not args.no_normalize:
        mean,std = replay_buffer.normalize_states() 
    else:
        mean,std = 0,1
    if args.model_name_rl == "svr":
        reward = eval_policy(policy, env,  args.env, mean, std, eval_episodes=args.eval_episodes, plot=True, writer=writer)
    else:
        # print(policy)
        reward = eval_bc(policy, env, args.env, mean, std, eval_episodes=args.eval_episodes, plot=True, writer=writer)
    writer.add_scalar('eval/reward', reward['avg_reward'], 0)
    if args.env == "abiomed":
        writer.add_scalar('eval/acp', reward['acp'], 0)
        writer.add_scalar('eval/map_air', reward['map_air'], 0)
        writer.add_scalar('eval/hr_air', reward['hr_air'], 0)
        writer.add_scalar('eval/pulsatility_air', reward['pulsatility_air'], 0)
        writer.add_scalar('eval/aggregate_air', reward['aggregate_air'], 0)
        writer.add_scalar('eval/unstable_hours_pct', reward['unstable_hours_pct'], 0)
        writer.add_scalar('eval/weaning_score', reward['weaning_score'], 0)

    time.sleep( 10 )
