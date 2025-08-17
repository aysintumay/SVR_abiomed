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
    base_config = """optimization_metric: density_range
bandwidth: 1
k_neighbors: 100
plot: True
save_model: trained_kde
save_results: results.pkl

data_path: {data_path}
env: {env}

transition: True


action: True

"""
   
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
    base_config = """optimization_metric: density_range
bandwidth: 1
k_neighbors: 100
plot: True
save_model: trained_kde
save_results: results.pkl

data_path: {data_path}
env: {env}

action: True

"""
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
    base_config = """optimization_metric: density_range
bandwidth: 1
k_neighbors: 100
plot: True
save_model: trained_kde
save_results: results.pkl

data_path: {data_path}
env: {env}

transition: True

"""
    
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
    base_config = """optimization_metric: density_range
bandwidth: 1
k_neighbors: 100
plot: True
save_model: trained_kde
save_results: results.pkl

env: {env}
data_path: {data_path}
"""
    if algo_type == "svr_kde":
            base_config = base_config + """classifier_model_name: {env}/trained_kde"""



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