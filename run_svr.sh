# python SVR_old/main.py --config configs/train/svr/abiomed.yaml --devid 4 --alpha 0.008 --max_timesteps 400000 #best 0.008 200k
python SVR_old/main.py --config configs/train/svr/abiomed.yaml --devid 2 --alpha 0.006 --max_timesteps 400000
#best 0.006 100k 0.5
# python SVR_old/main.py --config configs/train/svr/abiomed.yaml --devid 1 --alpha 0.01 --max_timesteps 400000 --sample_Std 0.3 #diverging at 400k
# python SVR_old/main.py --config configs/train/svr/abiomed.yaml --devid 1 --alpha 0.01 --max_timesteps 400000
# python SVR_old/main.py --config configs/train/svr/abiomed.yaml --devid 1 --alpha 0.01 --max_timesteps 300000

