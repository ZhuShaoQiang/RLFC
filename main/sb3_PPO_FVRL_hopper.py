import sys
import os
sys.path.append(os.getcwd())

import gymnasium
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

import argparse

from utils.envwrappers import RecordRewWrapper
from config.config_hopper_ppo_rlfc import params
from utils.scorers import FeedForward
from utils.scorer_funcs import make_linear_scorer_func
from utils.utils import set_seed, ZFilter
from utils.algorithmwrappers import FVRL_PPO_2


def main():
    """
    使用sb3写的PPO算法
    """
    env = gymnasium.make(params["env_name"], render_mode="rgb_array")
    env = RecordRewWrapper(env, rew_log_dir=f"./ckp/log/rew/{params['env_name']}/ppo_fvrl", avg_n=10)
    env = DummyVecEnv([lambda: env])

    ### 家在scorer和scorer_func
    scorer = FeedForward(
        input_dim=11*params["time_len"], output_dim=1,
        hidden=[128, 128], last_activation=params["scorer_activation"],
    ).to(params["device"])
    scorer.load_state_dict(
        torch.load(params["SCORER_PATH"])
    )
    scorer_func = make_linear_scorer_func(0.15)

    ### 然后是做running state的
    # 下面的数据都是我训练scorer时得到的数据
    zf = ZFilter((11,), clip=None)
    zf.rs._M = np.array([1.3480943, -0.1120296, -0.5515859, -0.13156778, -0.00396417, 2.61002,
                         0.02249699, -0.01588676, -0.07048377, -0.05156435, 0.03788375], dtype=np.float32)
    zf.rs._S = np.array([2.0393014e+04, 1.6149427e+03, 1.6529029e+04, 2.4963338e+04, 2.7929269e+05,
                         2.8400144e+05, 1.8890055e+06, 5.2990088e+05, 3.2445728e+06, 4.6406270e+06,
                         2.7277886e+07], dtype=np.float32)
    zf.rs._n = 798145


    model = FVRL_PPO_2(
        MlpPolicy, env, scorer=scorer, scorer_func=scorer_func, params=params, zf=zf,
        verbose=1, tensorboard_log=f"./ckp/log/{params['env_name']}_ppo_fvrl/", gae_lambda=params["gae_lambda"],
    )
    print(model.policy)
    model.learn(total_timesteps=1_000_000)
    model.save(f"./ckp/pth/{params['env_name']}_ppo_fvrl_100w.pth")
    env.close()
    
if __name__ == "__main__":
    set_seed(params["seed"])
    main()