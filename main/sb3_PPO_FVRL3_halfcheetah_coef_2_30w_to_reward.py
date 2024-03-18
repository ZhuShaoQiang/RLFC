"""
这个是score加到reward上的，之前的实现都有问题，加到了advantage上，不是reward上
coef的系数从2到0
20w轮衰减时间，效果好一点，但是并不太好
30w轮衰减时间，效果非常好，2-0.0, 30w轮,
打分器的打分加到return上，使用FVRL_PPO_3上
"""
import sys
import os
sys.path.append(os.getcwd())

import gymnasium
import torch
import numpy as np
from utils.algorithmwrappers import FVRL_PPO_3
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.envwrappers import RecordRewWrapper
from utils.scorer_funcs import make_linear_decayable_scorer_func
from utils.scorers import RLFCScorer, FeedForward
from utils.utils import ZFilter, set_seed
from config.config_halfcheetah_ppo_rlfc import params

def main():
    """
    使用sb3写的PPO算法
    """
    env = gymnasium.make(params["env_name"], render_mode="rgb_array")
    env = RecordRewWrapper(env, rew_log_dir=f"./ckp/log/rew/{params['env_name']}/ppo_fvrl3_coef_2_30w", avg_n=10)
    env = DummyVecEnv([lambda: env])

    ### 加载scorer和scorer_func
    scorer = FeedForward(
        input_dim=17*params["time_len"], output_dim=1,
        hidden=[128, 128], last_activation=params["scorer_activation"],
    ).to(params["device"])
    scorer.load_state_dict(torch.load("ckp/pth/halfcheetah_scorer.pth"))
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
    
    model = FVRL_PPO_3(
        MlpPolicy, env, scorer=scorer, scorer_func=scorer_func, params=params, zf=zf,
        verbose=1, tensorboard_log=f"./ckp/log/{params['env_name']}_ppo_fvrl3_coef_2_30w/", gae_lambda=params["gae_lambda"],
    )
    print(model.policy)
    model.learn(total_timesteps=1_000_000)
    model.save(f"./ckp/pth/{params['env_name']}_ppo_fvrl3_coef_2_30w_100w.pth")
    env.close()

    
if __name__ == "__main__":
    set_seed(params["seed"])
    main()
