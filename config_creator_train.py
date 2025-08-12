import os
import sys

# Check for algorithm type input
# if len(sys.argv) != 2 or sys.argv[1] not in ["bc", "svr"]:
    # print("Usage: python config_creator.py <algo_type>")
    # print("<algo_type> must be either 'bc' or 'svr'")
    # sys.exit(1)

# Get the algorithm type from the command-line argument
algo_type = sys.argv[1]
noise_type = sys.argv[2]
# List of environments and corresponding data paths

if noise_type == "action_transition":
    environments = [
        {
            "env": "Walker2d-v2",
            "data_path": "/abiomed/intermediate_data_d4rl/farama_sac_expert/Walker2d-v2_action_obs_noisy_0.01_0.5_0.005_1.0.pkl"
        },
        {
            "env": "HalfCheetah-v2",
            "data_path": "/abiomed/intermediate_data_d4rl/farama_tqc_expert/HalfCheetah-v2_action_obs_noisy_0.01_0.5_0.005_1.0.pkl"
        },
        {
            "env": "Hopper-v2",
            "data_path": "/abiomed/intermediate_data_d4rl/farama_sac_expert/Hopper-v2_action_obs_noisy_0.01_0.5_0.005_1.0.pkl"
        },
        
    ]
    # Base configuration template
    base_config = """env: {env}
seed: 1
eval_episodes: 10
data_path: "{data_path}"

transition: True
noise_rate_transition: 1.0
scale_transition: 0.005

action: True
noise_rate_action: 0.5
scale_action: 0.01

alpha: 0.02

"""
    if algo_type == "svr_kde":
        base_config = base_config + """classifier_model_name: trained_kde_action_obs"""
elif noise_type == "action":
    environments = [
        {
            "env": "Walker2d-v2",
            "data_path": "/abiomed/intermediate_data_d4rl/farama_sac_expert/Walker2d-v2_action_noisy_0.08_1.0.pkl"
        },
        {
            "env": "HalfCheetah-v2",
            "data_path": "/abiomed/intermediate_data_d4rl/farama_tqc_expert/HalfCheetah-v2_action_noisy_0.08_1.0.pkl"
        },
        {
            "env": "Hopper-v2",
            "data_path": "/abiomed/intermediate_data_d4rl/farama_sac_expert/Hopper-v2_action_noisy_0.08_1.0.pkl"
        },
        
    ]
    # Base configuration template
    base_config = """env: {env}
seed: 1
eval_episodes: 10
data_path: "{data_path}"

action: True
noise_rate_action: 1.0
scale_action: 0.08

alpha: 0.02
"""
    if algo_type == "svr_kde":
        base_config = base_config + """classifier_model_name: trained_kde_action"""
elif noise_type == "transition":
    environments = [
        {
            "env": "Walker2d-v2",
            "data_path": "/abiomed/intermediate_data_d4rl/farama_sac_expert/Walker2d-v2_obs_noisy_0.005_1.0.pkl"
        },
        {
            "env": "HalfCheetah-v2",
            "data_path": "/abiomed/intermediate_data_d4rl/farama_tqc_expert/HalfCheetah-v2_obs_noisy_0.005_1.0.pkl"
        },
        {
            "env": "Hopper-v2",
            "data_path": "/abiomed/intermediate_data_d4rl/farama_sac_expert/Hopper-v2_obs_noisy_0.005_1.0.pkl"
        },
        
    ]
    # Base configuration template
    base_config = """env: {env}
seed: 1
eval_episodes: 10
data_path: "{data_path}"

transition: True
noise_rate_transition: 1.0
scale_transition: 0.005
alpha: 0.02
"""
    if algo_type == "svr_kde":
        base_config = base_config + """classifier_model_name: trained_kde_obs"""
else:
    environments = [
        {
            "env": "Walker2d-v2",
            "data_path": "/abiomed/intermediate_data_d4rl/farama_sac_expert/Walker2d-v2_expert_1086.pkl"
        },
        {
            "env": "HalfCheetah-v2",
            "data_path": "/abiomed/intermediate_data_d4rl/farama_tqc_expert/HalfCheetah-v2_expert_1086.pkl"
        },
        {
            "env": "Hopper-v2",
            "data_path": "/abiomed/intermediate_data_d4rl/farama_sac_expert/Hopper-v2_aexpert_1000.pkl"
        },
    ]
    # Base configuration template
    base_config = """env: {env}
seed: 1
eval_episodes: 10
data_path: "{data_path}"
alpha: 0.02
"""
    if algo_type == "svr_kde":
            base_config = base_config + """classifier_model_name: trained_kde"""


# Directory to save the configuration files
output_dir = f"configs/train/{algo_type}"
os.makedirs(output_dir, exist_ok=True)

# Generate configuration files
for env in environments:
    config_content = base_config.format(env=env["env"], data_path=env["data_path"])
    file_name = f"{env['env'].lower().split('-')[0]}_{noise_type}.yaml"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w") as file:
        file.write(config_content)

print(f"Configuration files created in {output_dir}")