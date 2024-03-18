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
from config.config_walker2d_ppo_rlfc import params
from utils.scorers import FeedForward
from utils.scorer_funcs import make_linear_scorer_func
from utils.utils import set_seed, ZFilter
from utils.algorithmwrappers import FVRL_PPO_2


def main():
    """
    使用sb3写的PPO算法
    """
    env = gymnasium.make(params["env_name"])
    env = RecordRewWrapper(env, rew_log_dir=f"./ckp/log/rew/{params['env_name']}/ppo_fvrl", avg_n=10)
    env = DummyVecEnv([lambda: env])

    ### 家在scorer和scorer_func
    scorer = FeedForward(
        input_dim=17*params["time_len"], output_dim=1,
        hidden=[128, 128], last_activation=params["scorer_activation"],
    ).to(params["device"])
    scorer.load_state_dict(
        torch.load(params["SCORER_PATH"])
    )
    scorer_func = make_linear_scorer_func(0.15)

    ### 然后是做running state的
    # 下面的数据都是我训练scorer时得到的数据
    zf = ZFilter((17,), clip=None)
    zf.rs._M = np.array([1.23724556e+00, 1.94960609e-01, -1.04840055e-01, -1.86109930e-01,
                         2.27848172e-01, 2.29245853e-02, -3.74142468e-01, 3.40103954e-01,
                         3.92503405e+00, -5.22093428e-03, 2.52020825e-02, -3.59374401e-03,
                         -1.85715072e-02, -4.82054889e-01, 1.39282073e-03, -1.13219686e-03,
                         6.86046761e-03], dtype=np.float32)
    zf.rs._S = np.array([3.5312812e+03, 2.3014244e+04, 2.4009773e+04, 3.8081957e+04, 4.4468591e+05,
                         4.5188315e+02, 1.1123111e+05, 3.0795775e+05, 7.5246044e+05, 4.2556634e+05,
                         1.7969398e+06, 4.9785975e+06, 9.8378780e+06, 2.2978520e+07, 4.9612191e+05,
                         1.4875054e+07, 3.0440262e+07], dtype=np.float32)
    zf.rs._n = 797995


    model = FVRL_PPO_2(
        MlpPolicy, env, scorer=scorer, scorer_func=scorer_func, params=params, zf=zf,
        verbose=1, tensorboard_log=f"./ckp/log/{params['env_name']}_ppo_fvrl/", gae_lambda=0.0,
    )
    print(model.policy)
    model.learn(total_timesteps=1_000_000)
    model.save(f"./ckp/pth/{params['env_name']}_ppo_fvrl_100w.pth")
    env.close()
    
if __name__ == "__main__":
    set_seed(params["seed"])
    main()