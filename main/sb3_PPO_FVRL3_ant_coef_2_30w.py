"""
FVRL3的，score加到reward上的
"""
import sys
import os
sys.path.append(os.getcwd())

import gymnasium
import torch
import numpy as np
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

import argparse

from utils.envwrappers import RecordRewWrapper
from config.config_ant_ppo_rlfc import params
from utils.scorers import FeedForward
from utils.scorer_funcs import make_linear_decayable_scorer_func
from utils.utils import set_seed, ZFilter
from utils.algorithmwrappers import FVRL_PPO_3


def main():
    """
    使用sb3写的PPO算法
    """
    env = gymnasium.make(params["env_name"], render_mode="rgb_array")
    env = RecordRewWrapper(env, rew_log_dir=f"./ckp/log/rew/{params['env_name']}/ppo_fvrl3_decay_2.0_30w", avg_n=10)
    env = DummyVecEnv([lambda: env])

    ### 家在scorer和scorer_func
    scorer = FeedForward(
        input_dim=27*params["time_len"], output_dim=1,
        hidden=[128, 128], last_activation=params["scorer_activation"],
    ).to(params["device"])
    scorer.load_state_dict(
        torch.load(params["SCORER_PATH"])
    )
    scorer_func = make_linear_decayable_scorer_func(0.2, 300_000)

    ### 然后是做running state的
    # 下面的数据都是我训练scorer时得到的数据
    zf = ZFilter((27,), clip=None)
    zf.rs._M = np.array([5.7204503e-01, 8.8205910e-01, -4.4409309e-02, -1.1213569e-02,
                         3.3100584e-01, -3.8856622e-03, 5.3868991e-01, 4.9549326e-01,
                         -5.4437119e-01, -4.8728917e-02, -6.0110921e-01,-4.8644340e-01,
                         5.8493084e-01, 4.8682885e+00, -6.0119557e-01, -8.5669669e-04,
                         -5.0273221e-03, -1.3729665e-02, 9.4576078e-03, -7.6166041e-02,
                         1.7013421e-02, 1.3555721e-03, -2.3573169e-02, 6.3144073e-02,
                         -2.1758540e-02, 9.4291260e-03, 2.2186229e-02], dtype=np.float32)
    zf.rs._S = np.array([1.29025068e+04, 3.24626738e+04, 9.41849316e+03, 2.00972949e+04,
                         2.59153203e+04, 1.51730484e+05, 4.62857715e+03, 2.17508828e+04,
                         1.08091465e+04, 1.54977484e+05, 1.92107734e+04, 2.46792656e+04,
                         2.41501289e+04, 2.07638188e+06, 7.67140875e+05, 1.04290844e+06,
                         1.20104700e+06, 1.03787425e+06, 5.36509250e+05, 2.40221260e+07,
                         2.57940094e+05, 1.38937812e+05, 2.32204969e+05, 2.72608860e+07,
                         4.31451450e+06, 3.90981250e+05, 1.45166788e+06], dtype=np.float32)
    zf.rs._n = 798145

    model = FVRL_PPO_3(
        MlpPolicy, env, scorer=scorer, scorer_func=scorer_func, params=params, zf=zf,
        verbose=1, tensorboard_log=f"./ckp/log/{params['env_name']}_ppo_fvrl3_decay_2.0_30w/", gae_lambda=params["gae_lambda"],
    )
    print(model.policy)
    model.learn(total_timesteps=1_000_000)
    model.save(f"./ckp/pth/{params['env_name']}_ppo_fvrl3_decay_2.0_30w_100w.pth")
    env.close()
    
if __name__ == "__main__":
    set_seed(params["seed"])
    main()