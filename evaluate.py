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
from noisy_mujoco.abiomed_env.cost_func import overall_acp_cost
from noisy_mujoco.abiomed_env.weaning_score import weaning_score


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


def eval_policy(policy, eval_env, env_name, mean, std, writer=None, seed_offset=100, eval_episodes=10, plot=None):

    if env_name == 'abiomed':
		
        avg_reward = 0.
        avg_acp = 0.
        avg_weaning = 0.
		
        for k in range(eval_episodes):
            state_ = []
            next_state_ = []

            (state, _), done = eval_env.reset(), False #state is normalized
            # if k == np.random.randint(1, eval_episodes-1):
            truncated = False
			
            while not (done or truncated):
                state = (np.array(state).reshape(1,-1) - mean)/std
                action = policy.select_action(state) #action is in [2,10], state is already normalized
                next_state, reward, done, truncated, _ = eval_env.step(action)
                avg_reward += reward

                state_.append(state)
                next_state_.append(next_state)
                state = next_state
            if (k == 2) & plot:
                #unnormalize
                max_steps = eval_env.max_steps
                forecast_n = eval_env.world_model.forecast_horizon
                action_unnorm  = np.repeat(eval_env.episode_actions,forecast_n)
                state_unnorm = eval_env.world_model.unnorm_output(np.array(state_).reshape(max_steps, forecast_n, -1))
                next_state_unnorm = eval_env.world_model.unnorm_output(np.array(next_state_).reshape(max_steps, forecast_n, -1))
                utils.plot_policy(action_unnorm, state_unnorm, next_state_unnorm, writer)
            # print([[eval_env.episode_actions]])
            avg_acp += overall_acp_cost([eval_env.episode_actions])
            avg_weaning += weaning_score(eval_env.episode_actions, eval_env.episode_rewards)
        avg_reward /= eval_episodes
        acp = avg_acp / eval_episodes
        wean_s = avg_weaning / eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, ACP score: {acp:.4f}, Weaning Score: {wean_s:.4f}")
        print("---------------------------------------")
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
    return avg_reward

def eval_bc(actor, eval_env, env_name, mean, std, writer=None, seed_offset=100, eval_episodes=10, plot=None):
    
    if env_name == 'abiomed':
        avg_reward = 0.
        avg_acp = 0.
        avg_weaning = 0.
        for k in range(eval_episodes):
            state_ = []
            next_state_ = []

            (state, _), done = eval_env.reset(), False #state is normalized
            # if k == np.random.randint(1, eval_episodes-1):
            truncated = False
            
            while not (done or truncated):
                state = (np.array(state).reshape(1,-1) - mean)/std
                eval_states_tensor = torch.Tensor(state).to(actor.device)
                
                if eval_env.action_space_type == "discrete":
                    # print(eval_states_tensor.max(), eval_states_tensor.min())
                    
                    action, _ = actor(eval_states_tensor, deterministic= True)
                    # print("action", action)
                else:
                    action = actor(eval_states_tensor)

                next_state, reward, done, truncated, _ = eval_env.step(action)
                avg_reward += reward

                state_.append(state)
                next_state_.append(next_state)
                state = next_state

            if (k == 3) & plot:
                #unnormalize
                max_steps = eval_env.max_steps
                forecast_n = eval_env.world_model.forecast_horizon
                action_unnorm  = np.repeat(eval_env.episode_actions,forecast_n)
                state_unnorm = eval_env.world_model.unnorm_output(np.array(state_).reshape(max_steps, forecast_n, -1))
                next_state_unnorm = eval_env.world_model.unnorm_output(np.array(next_state_).reshape(max_steps, forecast_n, -1))
                utils.plot_policy(action_unnorm, state_unnorm, next_state_unnorm, writer)
       
            avg_acp += overall_acp_cost([eval_env.episode_actions])
            avg_weaning += weaning_score(eval_env.episode_actions, eval_env.episode_rewards)
            
        avg_reward /= eval_episodes
        acp = avg_acp / eval_episodes
        wean_s = avg_weaning / eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, ACP score: {acp:.4f}, Weaning Score: {wean_s:.4f}")
        print("---------------------------------------")
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
    return avg_reward

if __name__ == "__main__":
    print("Running", __file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args, remaining_argv = parser.parse_known_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    #=========== SVR arguments ============
    # parser.add_argument("--env", default="abiomed")        # OpenAI gym environment name
    # parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=100, type=int)       # How often (time steps) we evaluate
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
    # parser.add_argument("--noise_rate_action", type=float, help="Portion of action to be noisy with probability", default=0.01)
    # parser.add_argument("--noise_rate_transition", type=float, help="Portion of transitions to be noisy with probability", default=0.01)
    parser.add_argument("--loc", type=float, default=0.0, help="Mean of the noise distribution")
    # parser.add_argument("--scale_action", type=float, default=0.001, help="Standard deviation of the action noise distribution")
    # parser.add_argument("--scale_transition", type=float, default=0.001, help="Standard deviation of the transition noise distribution")
    parser.add_argument("--action", action='store_true', help="Create dataset with noisy actions")
    parser.add_argument("--transition", action='store_true', help="Create dataset with noisy transitions")

    # #============ abiomed environment arguments ============
    parser.add_argument("--model_name", type=str, default="10min_1hr_window")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path_wm", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=24)
    parser.add_argument("--action_space_type", type=str, default="continuous", choices=["continuous", "discrete"], help="Type of action space for the environment") 

    parser.add_argument('--save_path', type=str, default='/abiomed/models/policy_models/', help='Path to save model and results')
    #=========== SVR arguments ============
    # parser.add_argument("--model_name_rl", type=str, default="svr", choices=["svr", "bc"], help="RL model to use for evaluation")

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
    writer.add_scalar('eval/reward', reward, 0)

    time.sleep( 10 )
