python SVR_kde/main_kde.py --config configs/train/svr_kde/abiomed.yaml --devid 1 --alpha 0.01 --max_timesteps 300000
python SVR_kde/main_kde.py --config configs/train/svr_kde/abiomed.yaml --devid 1 --alpha 0.008 --max_timesteps 300000
# python SVR_kde/main_kde.py --config configs/train/svr/abiomed.yaml --devid 0 --alpha 0.008 --max_timesteps 180000
# python SVR_kde/main_kde.py --config configs/train/svr_kde/abiomed.yaml --devid 0 --alpha 0.006 --max_timesteps 200000 #best
python SVR_kde/main_kde.py --config configs/train/svr_kde/abiomed.yaml --devid 1 --alpha 0.006 --max_timesteps 130000 #best

