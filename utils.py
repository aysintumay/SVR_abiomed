import numpy as np
import torch
import tqdm
import os
import sys
import gymnasium as gym
import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from noisy_mujoco.abiomed_env.rl_env import AbiomedRLEnvFactory
from noisy_mujoco.wrappers import RandomNormalNoisyTransitionsActions, RandomNormalNoisyActions, RandomNormalNoisyTransitions

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cpu"):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device =device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]




	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std
	

class ReplayBufferAbiomed(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), timesteps = 6, feature_dim=12, device="cpu"):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.timesteps = timesteps
		self.feature_dim = feature_dim

		self.device =device
		self.action_space_type  = "continuous" if action_dim == 1 else "discrete"

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert_abiomed(self, dataset, env, length=90):

		
		reward_l = []
		done_l = []


		observation = dataset.data.reshape(-1,self.timesteps*(self.feature_dim))
		next_observation = torch.cat([dataset.labels.reshape(-1, self.timesteps, self.feature_dim-1), dataset.pl.reshape(-1, self.timesteps, 1)], axis = 2)
		next_observation = next_observation.reshape(-1,self.timesteps*(self.feature_dim))
		
		action = dataset.pl
		#take one number with majority voting among 6 numbers
		# if self.action_space_type == "continuous":
		action_unnorm = np.array(env.world_model.unnorm_pl(action))
		action = np.array([np.bincount(a.astype(int)).argmax() for a in action_unnorm]).reshape(-1,1)
		#normalize back
		action = env.world_model.normalize_pl(torch.Tensor(action))
		
		for i in tqdm.tqdm(range(action.shape[0])):
		
			reward = env._compute_reward(next_observation[i].reshape(-1,self.timesteps, self.feature_dim))
			reward_l.append(reward)
			done_l.append(np.array([0]))
		print(type(observation))
		print(type(action))
		self.state = np.array(observation)
		self.action =  np.array(action)
		self.next_state =  np.array(next_observation)
		self.reward =  np.array(reward_l).reshape(-1,1)
		self.not_done = 1. -  np.array(done_l).reshape(-1,1)
		self.size = self.state.shape[0]




	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		#do not take p-level mean
		self.next_state = (self.next_state - mean)/std
		return mean, std
	

def get_env_data(args, val=None):
	if args.env[0].isupper():
		env = gym.make(args.env)
		if args.action and not args.transition:
			print("Environment with noisy actions")
			env = RandomNormalNoisyActions(env=env, noise_rate=args.noise_rate_action, loc = args.loc, scale = args.scale_action)
		elif args.transition and not args.action:
			print("Environment with noisy transitions")
			env = RandomNormalNoisyTransitions(env=env, noise_rate=args.noise_rate_transition, loc = args.loc, scale = args.scale_transition)
		elif args.transition and args.action:
			print("Environment with noisy actions and transitions")
			env = RandomNormalNoisyTransitionsActions(env=env, noise_rate_action=args.noise_rate_action, loc = args.loc, scale_action = args.scale_action,\
															noise_rate_transition=args.noise_rate_transition, scale_transition = args.scale_transition)
		else:
			print("Environment without noise")
			env = env
		with open(args.data_path, 'rb') as f:
			print('opening')
			dataset = pickle.load(f)
		return env, dataset
	elif args.env == 'abiomed':
		print("In env", args.action_space_type)
		env = AbiomedRLEnvFactory.create_env(
									model_name=args.model_name,
									model_path=args.model_path,
									data_path=args.data_path_wm,
									max_steps=args.max_steps,
									action_space_type=args.action_space_type,
									reward_type="smooth",
									normalize_rewards=True,
									seed=42,
									device = f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu",
									)
		dataset = env.world_model.data_train
		if val:
			print("Validation data is supported for Abiomed dataset")
			dataset_val = env.world_model.data_test
			return env, dataset, dataset_val
		else:
			print("No validation data for Abiomed dataset")
			dataset_val = None
			return env, dataset
	else:
		env = gym.make(args.env)
		dataset =  d4rl.qlearning_dataset(env)
		if args.env == "hopper-expert-v2":
			with open(args.data_path, 'rb') as f:
				dataset = pickle.load(f)
		return env, (dataset, dataset_val)


def plot_policy(action, state, next_state, writer):


	input_color = 'tab:blue'
	pred_color = 'tab:orange' #label="input",
	gt_color = 'tab:green'
	rl_color = 'tab:red'

	fig, ax1 = plt.subplots(figsize = (8,5.8), dpi=300)
									
	default_x_ticks = range(0, 181, 18)
	x_ticks = np.array(list(range(0, 31, 3)))
	plt.xticks(default_x_ticks, x_ticks)
	x1 = len(state[:, :, 0].reshape(-1,1))
	ax1.axvline(x=x1, linestyle='--', c='black', alpha =0.7)
	

	plt.plot(range(x1), state[:, :, 0].reshape(-1,1), label ='Observed MAP', color=input_color)
	plt.plot(range(x1, 2*x1), next_state[:, :, 0].reshape(-1,1),  label ='Predicted MAP', color=pred_color)
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax2.plot(range(x1,2*x1), action.reshape(-1,1),'--',label ='Recommended PL', color=rl_color)
	ax1.legend(loc=3)
	ax2.legend(loc=1)

	ax1.set_ylabel('MAP (mmHg)',  fontsize=20)
	ax2.set_ylabel('P-level',  fontsize=20)
	ax1.set_xlabel('Time (min)', fontsize=20)
	ax1.set_title(f"MAP Prediction and P-level")
	# wandb.log({f"plot_batch_{iter}": wandb.Image(fig)})

	canvas = FigureCanvas(fig)
	buf = io.BytesIO()
	canvas.print_png(buf)
	buf.seek(0)
	import PIL.Image
	img = PIL.Image.open(buf)
	img_array = np.array(img).transpose(2, 0, 1)  # (C, H, W) for TensorBoard
	writer.add_image("eval/samples", img_array)
	plt.close(fig)

	plt.show()