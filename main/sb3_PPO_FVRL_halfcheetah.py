"""
"""
import sys
import os
sys.path.append(os.getcwd())

import gymnasium
import torch
import numpy as np
from utils.algorithmwrappers import FVRL_PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

import argparse

from utils.envwrappers import RecordRewWrapper
from utils.scorer_funcs import make_linear_scorer_func
from utils.scorers import RLFCScorer, FeedForward
from utils.utils import ZFilter
from config.config_halfcheetah_ppo_rlfc import params

def main():
    """
    使用sb3写的PPO算法
    """
    env = gymnasium.make("HalfCheetah-v3", render_mode="rgb_array")
    env = RecordRewWrapper(env, rew_log_dir="ckp/log/rew/fvrl_wo_gae_0.15", avg_n=params["last_n_avg_rewards"])
    env = DummyVecEnv([lambda: env])

    ### 加载scorer和scorer_func
    scorer = FeedForward(
        input_dim=17*4, output_dim=1,
        hidden=[128, 128],
        last_activation=torch.nn.Sigmoid,
    ).to(params["device"])
    scorer.load_state_dict(torch.load("ckp/pth/halfcheetah_scorer.pth"))
    scorer_func = make_linear_scorer_func(0.15)

    ### running state   
    zf = ZFilter((17,), clip=None)
    zf.rs._M = np.array([-0.04478317, 0.04914692, 0.06074467, -0.17108327, -0.19509517, -0.05842201,
                0.09679054, 0.03292991, 11.043298, -0.07985519, -0.32117411, 0.3643959,
                0.41783863, 0.40012282, 1.1037232, -0.48573354, -0.07186417], dtype=np.float32)
    zf.rs._S = np.array([1.2669216e+03, 3.8950841e+05, 2.3464983e+05, 1.3784634e+05, 4.5199145e+04,
                3.0757291e+05, 7.2468898e+04, 3.7745648e+04, 3.8957368e+06, 2.6245556e+05,
                2.3822355e+06, 1.1178512e+08, 1.1601897e+08, 3.9641892e+07, 1.4541214e+08,
                4.1301924e+07, 2.0188344e+07], dtype=np.float32)
    zf.rs._n = 797995
    
    model = FVRL_PPO(
        MlpPolicy, env, scorer=scorer, scorer_func=scorer_func, params=params, zf=zf, gae_lambda=0.0,
        verbose=1, tensorboard_log="ckp/log/rew_fvrl_wo_gae_0.15"
    )
    print(model.policy)
    model.learn(total_timesteps=1_000_000)
    model.save("./ckp/pth/halfcheetah_ppo_fvrl_wo_gae_100w.pth")
    env.close()

    
if __name__ == "__main__":
    main()