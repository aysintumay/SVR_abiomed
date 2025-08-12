# Supported Value Regularization for Offline Reinforcement Learning

Original PyTorch implementation of the SVR algorithm from [Supported Value Regularization for Offline Reinforcement Learning](https://openreview.net/forum?id=fze7P9oy6l).

## Environment
Paper results were collected with [MuJoCo 210](https://mujoco.org/) (and [mujoco-py 2.1.2.14](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.23.1](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/Farama-Foundation/D4RL). Networks are trained using [PyTorch 1.11.0](https://github.com/pytorch/pytorch) and [Python 3.7](https://www.python.org/).

## Usage

```
python main.py --env HalfCheetah-v2 --eval_freq 20000 --devid 3 --data_path /abiomed/intermediate_data_d4rl/farama_tqc_expert/HalfCheetah-v2_action_noisy_0.08_1.0.pkl --action --noise_rate_action 1.0 --scale_action 0.08

python main.py --env HalfCheetah-v2 --eval_freq 20000 --devid 1 --data_path /abiomed/intermediate_data_d4rl/farama_tqc_expert/HalfCheetah-v2_action_obs_noisy_0.01_0.5_0.005_1.0.pkl --transition --noise_rate_transition 1.0 --scale_transition 0.005 --action --noise_rate_action 0.5 --scale_action 0.01

python main.py --env HalfCheetah-v2 --eval_freq 20000 --devid 6 --data_path /abiomed/intermediate_data_d4rl/farama_tqc_expert/HalfCheetah-v2_expert_1086.pkl

python main.py --env HalfCheetah-v2 --eval_freq 20000 --devid 4 --data_path /abiomed/intermediate_data_d4rl/farama_tqc_expert/HalfCheetah-v2_obs_noisy_0.005_1.0.pkl --transition --noise_rate_transition 1.0 --scale_transition 0.005
```
### Run noiseless main
```
python main.py --env HalfCheetah-v2 --eval_freq 20000 --devid 7 --data_path /abiomed/intermediate_data_d4rl/farama_tqc_expert/HalfCheetah-v2_expert_1086.pkl
```
### Run noisy pretraining
```
python pretrain.py --env HalfCheetah-v2 --seed 1 --eval_data 0.1 --devid 7 --data_path "/abiomed/intermediate_data_d4rl/farama_tqc_expert/HalfCheetah-v2_action_noisy_0.08_1.0.pkl" --action --noise_rate_action 1.0 --scale_action 0.08
```
### Run KDE model with noisy data

```
python kde_nn.py     --optimization_metric density_range     --bandwidth 1     --k_neighbors 100     --plot     --save_model trained_kde     --save_results results.pkl --data_path "/abiomed/intermediate_data_d4rl/farama_sac_expert/Hopper-v2_obs_noisy_0.005_1.0.pkl" --transition
```

### Pretrained Models

We have uploaded pretrained behavior models in SVR_bcmodels/ to facilitate experiment reproduction. 

You can also pretrain behavior models by running:
```
./run_pretrain.sh
```

### Offline RL


You can train SVR on D4RL datasets by running:
```
./run_experiments.sh
```

### Logging

This codebase uses tensorboard. You can view saved runs with:

```
tensorboard --logdir <run_dir>
```

## Bibtex
```
@inproceedings{mao2023supported,
	title={Supported Value Regularization for Offline Reinforcement Learning},
	author={Yixiu Mao and Hongchang Zhang and Chen Chen and Yi Xu and Xiangyang Ji},
	booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
	year={2023},
	url={https://openreview.net/forum?id=fze7P9oy6l}
}
```

## Acknowledgement

This repo borrows heavily from [TD3+BC](https://github.com/sfujim/TD3_BC) and [SPOT](https://github.com/thuml/SPOT).
