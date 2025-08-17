python SVR_kde/kde_nn.py --config configs/kde/halfcheetah_transition.yaml
python SVR_kde/kde_nn.py --config configs/kde/hopper_transition.yaml
python SVR_kde/kde_nn.py --config configs/kde/walker2d_transition.yaml

python SVR_kde/kde_nn.py --config configs/kde/halfcheetah_action.yaml
python SVR_kde/kde_nn.py --config configs/kde/hopper_action.yaml
python SVR_kde/kde_nn.py --config configs/kde/walker2d_action.yaml

python SVR_kde/kde_nn.py --config configs/kde/halfcheetah_action_transition.yaml
python SVR_kde/kde_nn.py --config configs/kde/hopper_action_transition.yaml
python SVR_kde/kde_nn.py --config configs/kde/walker2d_action_transition.yaml

python SVR_kde/kde_nn.py --config configs/kde/hopper_expert.yaml
python SVR_kde/kde_nn.py --config configs/kde/walker2d_expert.yaml
python SVR_kde/kde_nn.py --config configs/kde/halfcheetah_expert.yaml