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
from config.config_halfcheetah_sac_rlfc import params

def main():
    env = gymnasium.make(params["env_name"], render_mode="rgb_array")
    env = RecordRewWrapper(env, rew_log_dir=f"./ckp/log/rew/{params['env_name']}/sac_s_fvrl",
                           avg_n=10)  ## sac_fvrl
    env = DummyVecEnv([lambda: env])

    ### 加载scorer和scorer_func
    scorer = FeedForward(
        input_dim=17*params["time_len"], output_dim=1,
        hidden=[128, 128], last_activation=params["scorer_activation"],
    ).to(params["device"])
    scorer.load_state_dict(torch.load(params["SCORER_PATH"]))
    scorer_func = make_linear_decayable_scorer_func(0.20, 300_000)

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
