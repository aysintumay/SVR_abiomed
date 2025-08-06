import numpy as np
import torch
import gym
import argparse
import os
# import d4rl
from pathlib import Path
import random
import json
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from tqdm import trange
import gymnasium as gy
import pickle
import sys
import faiss
from kde_nn import PercentileThresholdKDE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_env_data
from SVR_kde import SVR 

import utils

from noisy_mujoco.abiomed_env.rl_env import AbiomedRLEnvFactory

    
def eval_policy(policy, env_name, seed, mean, std, writer=None, seed_offset=100, eval_episodes=10, plot=None):
	if env_name == 'abiomed':
		eval_env = AbiomedRLEnvFactory.create_env(
									model_name=args.model_name,
									model_path=args.model_path,
									data_path=args.data_path_wm,
									max_steps=args.max_steps,
									action_space_type="continuous",
									reward_type="smooth",
									normalize_rewards=True,
									seed=42,
									device = policy.device if torch.cuda.is_available() else "cpu",
									)
		eval_env.seed(seed + seed_offset)
		eval_env.action_space.seed(seed + seed_offset)
		avg_reward = 0.
		
		for k in range(eval_episodes):
			state_ = []
			next_state_ = []

			(state, _), done = eval_env.reset(), False #state is normalized
			truncated = False
			
			while not (done or truncated):
				state = (np.array(state).reshape(1,-1) - mean)/std #if no_normalize mean=0 std =1
				action = policy.select_action(state) #action is in [2,10], state is already normalized
				next_state, reward, done, truncated, _ = eval_env.step(action)
				avg_reward += reward

				state_.append(state)
				next_state_.append(next_state)
				state = next_state

			if (k == 1) & plot:
				#unnormalize
				max_steps = eval_env.max_steps
				forecast_n = eval_env.world_model.forecast_horizon
				action_unnorm  = np.repeat(eval_env.episode_actions,forecast_n)
				state_unnorm = eval_env.world_model.unnorm_output(np.array(state_).reshape(max_steps, forecast_n, -1))
				next_state_unnorm = eval_env.world_model.unnorm_output(np.array(next_state_).reshape(max_steps, forecast_n, -1))
				utils.plot_policy(action_unnorm, state_unnorm, next_state_unnorm, writer)
	else:
		eval_env = gym.make(env_name)

		# eval_env.seed(seed + seed_offset)
		eval_env.action_space.seed(seed + seed_offset)
		eval_env.observation_space.seed(seed + seed_offset)
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
	# d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	#=========== SVR arguments ============
	parser.add_argument("--env", default="Hopper-v2")        # OpenAI gym environment name
	parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=2e4, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--eval_episodes", default=10, type=int)
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument('--folder', default='train_rl')
	parser.add_argument("--no_normalize", action="store_true")
	parser.add_argument('--no_schedule', action="store_true")
	parser.add_argument('--snis', action="store_true")
	parser.add_argument("--alpha", default=0.1, type=float)
	parser.add_argument('--sample_std', default=0.2, type=float)
	parser.add_argument('--devid', default=5, type=int)
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
	parser.add_argument("--model_name", type=str, default="10min_1hr_window")
	parser.add_argument("--model_path", type=str, default=None)
	parser.add_argument("--data_path_wm", type=str, default=None)
	parser.add_argument("--max_steps", type=int, default=24)
	parser.add_argument("--action_space_type", type=str, default="continuous", choices=["continuous", "discrete"], help="Type of action space for the environment") 

	# parser.add_argument('--classifier_path', type=str, default='/abiomed/models/', help='Path to save model and results')
	parser.add_argument('--save_path', type=str, default='/abiomed/models/policy_models/', help='Path to save model and results')

	parser.add_argument("--classifier_model_name", type=str, default="trained_kde")
	args = parser.parse_args()

	device = torch.device(f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu")
	print(device)
	print("---------------------------------------")
	print(f"Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	env, dataset = get_env_data(args)
	work_dir = './runs/{}/{}/{}/alpha{}_seed{}_{}'.format(
     os.getcwd().split('/')[-1], args.folder, args.env, args.alpha, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
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

	if not args.no_normalize:
		print('Normalizing the state...')
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1

	
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

	working_dir = Path.cwd()
	# with open(os.path.join(working_dir, f"saved_models/kde/{args.classifier_model_name}.pkl"), "rb") as f:
	# 	classifier_dict = pickle.load(f)
	res = faiss.StandardGpuResources()
	
	# index = faiss.read_index("saved_models/kde/trained_kde.faiss")
	# n_features = (dataset['observations']).shape[1]
	# samples = (dataset['observations']).shape[0]
	# # if n_features <= 64 and samples < 1000000:
	# # 	index_cpu = faiss.IndexFlatL2(n_features)
	# # else:
	# # 	nlist = min(int(np.sqrt(samples)), 4096)
	# # 	quantizer = faiss.IndexFlatL2(n_features)
	# # 	index_cpu = faiss.IndexIVFFlat(quantizer, n_features, nlist)
	# gpu_index = faiss.index_cpu_to_gpu(res, args.devid, index)
	# classifier_dict = {"model": gpu_index, 
	# 					"thr":-1.27 }

	classifier_dict = PercentileThresholdKDE.load_model(f"/abiomed/models/kde/{args.classifier_model_name}", use_gpu=True, devid = args.devid)

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
		"classifier":classifier_dict,
		"alpha": args.alpha,
		"sample_std": args.sample_std,
		"device": device,
	}
	
	policy = SVR.SVR(**kwargs)
	
	for t in trange(int(args.max_timesteps)):
		policy.train(args.batch_size, writer)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			d4rl_score = eval_policy(policy, args.env, args.seed, mean, std, eval_episodes=args.eval_episodes, plot=True if t == int(args.max_timesteps)-1 else False, writer = writer)
			writer.add_scalar('eval/d4rl_score', d4rl_score, t)
	t0 = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	if not os.path.exists(os.path.join(args.save_path, "SVR_kde", save_path)):
		os.makedirs(os.path.join(args.save_path, "SVR_kde", save_path))
	policy.save(os.path.join(args.save_path, "SVR_kde", save_path, f"svr_kde_seed_{args.seed}_{t0}_{t+1}.pth"))
	time.sleep( 10 )
