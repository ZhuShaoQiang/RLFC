# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())

import gymnasium
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.envwrappers import RecordRewWrapper
from utils.utils import set_seed
from config.config_halfcheetah_sac_rlfc import params

def main():
    env = gymnasium.make(params["env_name"], render_mode="rgb_array")
    env = RecordRewWrapper(env, rew_log_dir=f"./ckp/log/rew/{params['env_name']}/sac_s", avg_n=10)
    env = DummyVecEnv([lambda: env])
    model = SAC(
        # MlpPolicy, env, verbose=1, tensorboard_log=f"./ckp/log/{params['env_name']}_sac/"
        MlpPolicy, env, verbose=1, tensorboard_log=None, policy_kwargs={"n_critics":1},
        seed=params["seed"],
    )
    print(model.policy)
    model.learn(total_timesteps=500_000)  # halfcheetah很简单，到这里就行了
    model.save(f"./ckp/pth/{params['env_name']}_sac_normal_50w.pth")
    env.close()

if __name__ == "__main__":
    set_seed(params["seed"])
    main()
