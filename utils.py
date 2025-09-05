import numpy as np
import torch
import tqdm
import os
import sys
import gymnasium as gym
import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import copy
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from noisy_mujoco.abiomed_env.rl_env import AbiomedRLEnvFactory
from noisy_mujoco.wrappers import (
    RandomNormalNoisyTransitionsActions,
    RandomNormalNoisyActions,
    RandomNormalNoisyTransitions,
)


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

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

    def convert_D4RL(self, dataset):
        self.state = dataset["observations"]
        self.action = dataset["actions"]
        self.next_state = dataset["next_observations"]
        self.reward = dataset["rewards"].reshape(-1, 1)
        self.not_done = 1.0 - dataset["terminals"].reshape(-1, 1)
        self.size = self.state.shape[0]

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std


class ReplayBufferAbiomed(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_size=int(1e6),
        timesteps=6,
        feature_dim=12,
        device="cpu",
    ):
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

        self.device = device
        self.action_space_type = "continuous" if action_dim == 1 else "discrete"

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

    def convert_abiomed(self, dataset, env, shuffle=None):
        if isinstance(dataset, dict):  # check if data is already in D4RL format
            self.state = dataset["observations"]
            self.action = dataset["actions"]
            self.next_state = dataset["next_observations"]
            self.reward = dataset["rewards"].reshape(-1, 1)
            self.not_done = 1.0 - dataset["terminals"].reshape(-1, 1)
            self.size = self.state.shape[0]
            # normalize actions
            self.action = np.asarray(
                env.world_model.normalize_pl(torch.Tensor(self.action))
            ).reshape(-1, 1)
            print("D4RL dataset for Abiomed")
        else:  # convert abiomed data into D4RL type
            if isinstance(dataset, list):
                all_x = torch.cat(
                    [dataset[0].data, dataset[1].data, dataset[2].data], axis=0
                )
                all_pl = torch.cat(
                    [dataset[0].pl, dataset[1].pl, dataset[2].pl], axis=0
                )
                all_labels = torch.cat(
                    [dataset[0].labels, dataset[1].labels, dataset[2].labels], axis=0
                )

                print(all_x.shape, all_pl.shape, all_labels.shape)

            else:
                all_x = dataset.data
                all_pl = dataset.pl
                all_labels = dataset.labels

            reward_l = []
            done_l = []
            observation = all_x.reshape(-1, self.timesteps * (self.feature_dim))
            next_observation = torch.cat(
                [
                    all_labels.reshape(-1, self.timesteps, self.feature_dim - 1),
                    all_pl.reshape(-1, self.timesteps, 1),
                ],
                axis=2,
            )
            next_observation = next_observation.reshape(
                -1, self.timesteps * (self.feature_dim)
            )

            action = all_pl
            # take one number with majority voting among 6 numbers
            action_unnorm = np.array(env.world_model.unnorm_pl(action))
            action_1 = np.array(
                [np.bincount(np.rint(a).astype(int)).argmax() for a in action_unnorm]
            ).reshape(-1, 1)
            # normalize back
            action = env.world_model.normalize_pl(torch.Tensor(action_1))
            obs_reshaped = (
                observation.reshape(-1, self.timesteps, self.feature_dim)
            ).clone()  # shape (1,6,12)
            for i in tqdm.tqdm(range(action.shape[0])):
                if (env.gamma1 != 0.0) or (env.gamma2 != 0.0) or (env.gamma3 != 0.0):
                    # change the last column of obs_reshaped with all_pl[i-1] after i==0
                    if i > 0:
                        obs_reshaped[i, :, -1] = all_pl[i - 1]
                    reward = env._compute_reward(
                        next_observation[i].reshape(
                            -1, self.timesteps, self.feature_dim
                        ),
                        obs_reshaped[i],
                        action_1[i],
                    )
                    # action2 = [action2[1], action_1[i+1]] if (i+1)< action.shape[0] else None

                else:
                    reward = env._compute_reward(
                        next_observation[i].reshape(
                            -1, self.timesteps, self.feature_dim
                        )
                    )

                reward_l.append(reward)
                done_l.append(np.array([0]))

            self.state = np.array(observation)
            self.action = np.array(action)
            self.next_state = np.array(next_observation)
            self.reward = np.array(reward_l).reshape(-1, 1)
            self.not_done = 1.0 - np.array(done_l).reshape(-1, 1)
            self.size = self.state.shape[0]

            # print(self.state.min(), self.state.max())
            # print(self.action.min(), self.action.max())
            # print(self.next_state.min(), self.next_state.max())
            # print(self.reward.min(), self.reward.max())
            # print(self.not_done.min(), self.not_done.max())

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        # do not take p-level mean
        self.next_state = (self.next_state - mean) / std
        return mean, std


def get_env_data(args, val=None):
    if args.env[0].isupper():
        env = gym.make(args.env)
        if args.action and not args.transition:
            print("Environment with noisy actions")
            env = RandomNormalNoisyActions(
                env=env,
                noise_rate=args.noise_rate_action,
                loc=args.loc,
                scale=args.scale_action,
            )
        elif args.transition and not args.action:
            print("Environment with noisy transitions")
            env = RandomNormalNoisyTransitions(
                env=env,
                noise_rate=args.noise_rate_transition,
                loc=args.loc,
                scale=args.scale_transition,
            )
        elif args.transition and args.action:
            print("Environment with noisy actions and transitions")
            env = RandomNormalNoisyTransitionsActions(
                env=env,
                noise_rate_action=args.noise_rate_action,
                loc=args.loc,
                scale_action=args.scale_action,
                noise_rate_transition=args.noise_rate_transition,
                scale_transition=args.scale_transition,
            )
        else:
            print("Environment without noise")
            env = env
        with open(args.data_path, "rb") as f:
            print("opening")
            dataset = pickle.load(f)
        return env, dataset
    elif args.env == "abiomed":
        print("In env", args.action_space_type)
        env = AbiomedRLEnvFactory.create_env(
            model_name=args.model_name,
            model_path=args.model_path,
            data_path=args.data_path_wm,
            max_steps=args.max_steps,
            gamma1=args.gamma1,
            gamma2=args.gamma2,
            gamma3=args.gamma3,
            action_space_type=args.action_space_type,
            reward_type="smooth",
            normalize_rewards=True,
            noise_rate=args.noise_rate,
            noise_scale=args.noise_scale,
            seed=42,
            device=f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu",
        )
        if args.data_path is None:
            dataset1 = env.world_model.data_train
            dataset2 = env.world_model.data_val
            dataset3 = env.world_model.data_test
            dataset = [dataset1, dataset2, dataset3]
        else:
            try:
                with open(args.data_path, "rb") as f:
                    dataset = pickle.load(f)
                print("opened the pickle file for synthetic dataset")
            except:
                dataset = np.load(args.data_path)
                dataset = {k: dataset[k] for k in dataset.files}
                print("opened the npz file for synthetic dataset")

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
        dataset = d4rl.qlearning_dataset(env)
        if args.env == "hopper-expert-v2":
            with open(args.data_path, "rb") as f:
                dataset = pickle.load(f)
        return env, (dataset, dataset_val)


def plot_policy(eval_env, state, all_states, title, writer, legend=None):
    """

    Plot the policy for the given state and environment.
    Args:
            eval_env: The evaluation environment.
            state ([max_steps, forecast_horizon*num_features]): The predicted state to plot. Includes the first p-level.
            all_states ([max_steps+1, forecast_horizon*num_features]): The real states including the first inputted state.
            writer: The writer to log the plot.
    """

    input_color = "tab:blue"
    pred_color = "tab:red"  # label="input",
    gt_color = "tab:red"
    rl_color = "tab:blue"
    hr_color = "tab:orange"
    pulsat_color = "tab:green"

    max_steps = eval_env.max_steps
    forecast_n = eval_env.world_model.forecast_horizon
    action_unnorm = np.repeat(eval_env.episode_actions, forecast_n)

    state_unnorm = eval_env.world_model.unnorm_output(
        np.array(state).reshape(max_steps, forecast_n, -1)
    )
    all_state_unnorm = eval_env.world_model.unnorm_output(
        np.array(all_states).reshape(max_steps + 1, forecast_n, -1)
    )
    first_action_unnorm = state_unnorm[0, :, -1]  # normalized

    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=300)  # Smaller plot size

    default_x_ticks = range(0, 181, 18)
    x_ticks = np.array(list(range(0, 31, 3)))
    plt.xticks(default_x_ticks, x_ticks)
    x1 = len(all_state_unnorm[0, :, 0].reshape(-1, 1))
    x2 = len(all_state_unnorm[1:, :, 0].reshape(-1, 1))
    ax1.axvline(x=x1, linestyle="--", c="black", alpha=0.7)

    (line_obs,) = ax1.plot(
        range(0, x1 + x2),
        all_state_unnorm[:, :, 0].reshape(-1, 1),
        "--",
        label="Observed MAP",
        alpha=0.5,
        color=gt_color,
        linewidth=2.0,
    )
    (line_obs2,) = ax1.plot(
        range(0, x1 + x2),
        all_state_unnorm[:, :, 9].reshape(-1, 1),
        "--",
        label="Observed HR",
        alpha=0.5,
        color=hr_color,
        linewidth=2.0,
    )
    (line_obs3,) = ax1.plot(
        range(0, x1 + x2),
        all_state_unnorm[:, :, 7].reshape(-1, 1),
        "--",
        label="Observed pulsat",
        alpha=0.5,
        color=pulsat_color,
        linewidth=2.0,
    )
    (line_pred2,) = ax1.plot(
        range(x1, x1 + x2),
        state_unnorm[:, :, 9].reshape(-1, 1),
        label="Predicted HR",
        alpha=0.5,
        color=hr_color,
        linewidth=2.0,
    )
    (line_pred3,) = ax1.plot(
        range(x1, x1 + x2),
        state_unnorm[:, :, 7].reshape(-1, 1),
        label="Predicted PULSAT",
        alpha=0.5,
        color=pulsat_color,
        linewidth=2.0,
    )
    (line_pred1,) = ax1.plot(
        range(x1, x1 + x2),
        state_unnorm[:, :, 0].reshape(-1, 1),
        label="Predicted MAP",
        color=pred_color,
        linewidth=2.0,
    )
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    (line_pl1,) = ax2.plot(
        range(0, x1 + x2),
        all_state_unnorm[:, :, -1].reshape(-1),
        "--",
        label="Input PL",
        alpha=0.5,
        color=input_color,
        linewidth=2.0,
    )
    (line_pl2,) = ax2.plot(
        range(x1, x1 + x2),
        action_unnorm.reshape(-1, 1),
        label="Recommended PL",
        color=rl_color,
        linewidth=2.0,
    )

    # Combined legend for all lines
    lines = [
        line_obs,
        line_obs2,
        line_obs3,
        line_pred2,
        line_pred3,
        line_pred1,
        line_pl1,
        line_pl2,
    ]
    labels = [
        "Observed MAP",
        "Observed HR",
        "Observed PULSAT",
        "Predicted HR",
        "Predicted PULSAT",
        "Predicted MAP",
        "Input PL",
        "Recommended PL",
    ]
    if legend:
        ax1.legend(
            lines,
            labels,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.37),
            fancybox=True,
            ncol=3,
            fontsize="medium",
        )  # Legend at the bottom

    # ax1.set_ylabel('MAP (mmHg)', size="x-large", color='tab:red')
    ax1.tick_params(axis="y", colors="tab:red")
    ax1.set_xlabel("Time (hour)", size="x-large")
    ax1.set_title(f"{title}", size="x-large", fontweight="bold")
    ax2.set_ylabel("", size="x-large", color="tab:blue")
    ax2.set_ylabel("P-level", size="x-large", color="tab:blue", labelpad=10)
    ax2.tick_params(axis="y", colors="tab:blue")
    ax2.set_ylim(1, 10)
    ax1.set_ylim(10, 130)
    ax1.grid()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    # wandb.log({f"plot_batch_{iter}": wandb.Image(fig)})

    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    import PIL.Image

    img = PIL.Image.open(buf)
    img_array = np.array(img).transpose(2, 0, 1)  # (C, H, W) for TensorBoard
    writer.add_image("eval/samples", img_array)
    # savefig
    fig.savefig(
        os.path.join(writer.log_dir, f"eval_samples_{title.replace('','_')}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    plt.show()

    return fig
