"""
cliff walking的dqn的sb的实现
"""

import sys
import os
sys.path.append(os.getcwd())

import gymnasium
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.envwrappers import RecordRewWrapper
from utils.scorer_funcs import make_linear_decayable_scorer_func
from utils.scorers import RLFCScorer, FeedForward
from utils.utils import ZFilter, set_seed
from utils.algorithmwrappers import FVRL_PPO_3, FVRL_DQN
from config.config_cw_dqn_rlfc import params


def main():
    env = gymnasium.make(params["env_name"])
    env = RecordRewWrapper(env, rew_log_dir=f"./ckp/log/rew/{params['env_name']}/dqn_fvrl_decal_2.0_10w", avg_n=10)
    env = DummyVecEnv([lambda: env])

    scorer = RLFCScorer(
        input_dim=int(env.observation_space.n * params["time_len"]),
        output_dim=1,
        activation=torch.nn.LeakyReLU(),
        hidden=[32*3, 16*3], last_activation=params["scorer_activation"]
    ).to(params["device"])
    scorer.load_state_dict(
        torch.load(params["SCORER_PATH"]), strict=True
    )
    scorer_func = make_linear_decayable_scorer_func(2, 100_000)

    model = FVRL_DQN(
        MlpPolicy, env, scorer=scorer, scorer_func=scorer_func, params=params, verbose=1, tensorboard_log=f"./ckp/log/{params['env_name']}_dqn_fvrl/",
    )
    print(model.policy)
    model.learn(total_timesteps=500_000)
    model.save(f"./ckp/pth/{params['env_name']}_dqn_fvrl_50w.pth")
    env.close()

    

if __name__ == "__main__":
    set_seed(42)
    main()
