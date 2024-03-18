# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())

import torch
import numpy as np
import gymnasium
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.scorer_funcs import make_linear_decayable_scorer_func
from utils.scorers import RLFCScorer, FeedForward
from utils.utils import ZFilter, set_seed

from utils.envwrappers import RecordRewWrapper
from utils.algorithmwrappers import FVRL_SAC
from config.config_hopper_sac_rlfc import params

def main():
    env = gymnasium.make(params["env_name"], render_mode="rgb_array")
    env = RecordRewWrapper(env, rew_log_dir=f"./ckp/log/rew/{params['env_name']}/sac_s_fvrl",
                           avg_n=10)  ## sac_fvrl
    env = DummyVecEnv([lambda: env])

    ### 加载scorer和scorer_func
    scorer = FeedForward(
        input_dim=11*params["time_len"], output_dim=1,
        hidden=[128, 128], last_activation=params["scorer_activation"],
    ).to(params["device"])
    scorer.load_state_dict(
        torch.load(params["SCORER_PATH"])
    )
    scorer_func = make_linear_decayable_scorer_func(0.20, 300_000)
    ### running state   
    zf = ZFilter((11,), clip=None)
    zf.rs._M = np.array([1.3480943, -0.1120296, -0.5515859, -0.13156778, -0.00396417, 2.61002,
                         0.02249699, -0.01588676, -0.07048377, -0.05156435, 0.03788375], dtype=np.float32)
    zf.rs._S = np.array([2.0393014e+04, 1.6149427e+03, 1.6529029e+04, 2.4963338e+04, 2.7929269e+05,
                         2.8400144e+05, 1.8890055e+06, 5.2990088e+05, 3.2445728e+06, 4.6406270e+06,
                         2.7277886e+07], dtype=np.float32)
    zf.rs._n = 798145

    
    ## 我不看这个log
    model = FVRL_SAC(
        MlpPolicy, env, verbose=1, 
        tensorboard_log=None,
        zf=zf, scorer=scorer, scorer_func=scorer_func, seed=params["seed"], params=params,
        policy_kwargs={"n_critics": 1},
    )
    print(model.policy)
    model.learn(total_timesteps=500_000, tb_log_name="SAC_S_RLFC")  # halfcheetah很简单，到这里就行了
    model.save(f"./ckp/pth/{params['env_name']}_sac_s_rlfc_50w.pth")
    env.close()

if __name__ == "__main__":
    set_seed(params["seed"])
    main()
