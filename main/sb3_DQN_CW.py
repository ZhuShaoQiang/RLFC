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
from utils.algorithmwrappers import FVRL_PPO_3
from config.config_cw_dqn_rlfc import params


def main():
    env = gymnasium.make(params["env_name"])
    env = RecordRewWrapper(env, rew_log_dir=f"./ckp/log/rew/{params['env_name']}/dqn", avg_n=10)
    env = DummyVecEnv([lambda: env])
    model = DQN(
        MlpPolicy, env, verbose=1, tensorboard_log=f"./ckp/log/{params['env_name']}_dqn/",
    )
    print(model.policy)
    model.learn(total_timesteps=500_000)
    model.save(f"./ckp/pth/{params['env_name']}_dqn_50w.pth")
    env.close()

    

if __name__ == "__main__":
    set_seed(42)
    main()
