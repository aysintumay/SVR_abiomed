import argparse
import numpy as np
import torch
import torch.nn.functional as F
import gym
from tqdm import tqdm
from SVR import Actor
# import d4rl

import random
import gymnasium as gy
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_env_data
import utils




parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default='abiomed')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_iters', type=int, default=int(1e5))
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--no_normalize', default=False, action='store_true')
parser.add_argument('--eval_data', default=0.0, type=float) # proportion of data used for evaluation
parser.add_argument('--devid', default=1, type=int)
parser.add_argument("--data_path", type=str, default="")

parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--data_path_wm", type=str, default=None)
parser.add_argument("--max_steps", type=int, default=6)
parser.add_argument("--normalize_rewards", action='store_true', help="Normalize rewards in the Abiomed environment")
parser.add_argument("--action_space_type", type=str, default="continuous", choices=["continuous", "discrete"], help="Type of action space for the environment") 
parser.add_argument('--fs', action= "store_true", help= "Use feature selection for the policy model")

parser.add_argument('--save_path', type=str, default='/abiomed/models/policy_models/', help='Path to save model and results')

#=========== noisy env arguments ============
parser.add_argument("--noise_rate_action", type=float, help="Portion of action to be noisy with probability", default=0.01)
parser.add_argument("--noise_rate_transition", type=float, help="Portion of transitions to be noisy with probability", default=0.01)
parser.add_argument("--loc", type=float, default=0.0, help="Mean of the noise distribution")
parser.add_argument("--scale_action", type=float, default=0.001, help="Standard deviation of the action noise distribution")
parser.add_argument("--scale_transition", type=float, default=0.001, help="Standard deviation of the transition noise distribution")
parser.add_argument("--action", action='store_true', help="Create dataset with noisy actions")
parser.add_argument("--transition", action='store_true', help="Create dataset with noisy transitions")

args = parser.parse_args()

# Set seeds
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = torch.device(f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu")



env, dataset = get_env_data(args)
print(env.observation_space.shape)
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0] if env.observation_space.__class__.__name__ == 'Box' else 1
# max_action = env.action_space.high[0] if hasattr(env.action_space, 'high') else 10
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])
if args.env == "abiomed":
    replay_buffer = utils.ReplayBufferAbiomed(state_dim, action_dim)
    replay_buffer.convert_abiomed(dataset, env)
    # replay_buffer_val = utils.ReplayBufferAbiomed(state_dim, action_dim)
    # replay_buffer_val.convert_abiomed(dataset_val, env)
else:
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(dataset)
if not args.no_normalize:
    mean, std = replay_buffer.normalize_states()
else:
    print("No normalize")
states = replay_buffer.state
actions = replay_buffer.action

if args.eval_data:
    eval_size = int(states.shape[0] * args.eval_data)
    eval_idx = np.random.choice(states.shape[0], eval_size, replace=False)
    train_idx = np.setdiff1d(np.arange(states.shape[0]), eval_idx)
    eval_states = states[eval_idx]
    eval_actions = actions[eval_idx]
    states = states[train_idx]
    actions = actions[train_idx]
else:
    eval_states = None
    eval_actions = None

actor = Actor(state_dim, action_dim, max_action).to(device)

optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)

for step in tqdm(range(args.num_iters + 1), desc='train'):
    idx = np.random.choice(states.shape[0], args.batch_size)
    train_states = torch.from_numpy(states[idx].astype(np.float32)).to(device)
    train_actions = torch.from_numpy(actions[idx].astype(np.float32)).to(device)
    
    pi = actor(train_states)
    loss = F.mse_loss(pi, train_actions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 5000 == 0:
        print('step: %s, train_loss: %s' % (step,loss.item()))
        if eval_states is not None and eval_actions is not None:
            actor.eval()
            with torch.no_grad():
                eval_states_tensor = torch.from_numpy(eval_states.astype(np.float32)).to(device)
                eval_actions_tensor = torch.from_numpy(eval_actions.astype(np.float32)).to(device)
                pi = actor(eval_states_tensor)
                loss = F.mse_loss(pi, eval_actions_tensor)
                print('step: %s, eval_loss: %s' % (step,loss.item()))
            actor.train()
    if step == args.num_iters:
        suffix_parts = [args.env]

        if args.action:
            suffix_parts.append(f"action")
        if args.transition:
            suffix_parts.append(f"obs")

        suffix = "_".join(suffix_parts)
        save_path = f"{args.save_path}SVR_bcmodels/bcmodel_{suffix}.pt"

        torch.save(actor.state_dict(), save_path)
