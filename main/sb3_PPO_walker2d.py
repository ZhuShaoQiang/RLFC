import sys
import os
sys.path.append(os.getcwd())

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

import argparse

from utils.envwrappers import RecordRewWrapper
from config.config_walker2d_ppo_rlfc import params
from utils.utils import set_seed


def main():
    """
    使用sb3写的PPO算法
    """
    env = gymnasium.make(params["env_name"])
    env = RecordRewWrapper(env, rew_log_dir=f"./ckp/log/rew/{params['env_name']}/ppo", avg_n=10)
    env = DummyVecEnv([lambda: env])
    model = PPO(
        MlpPolicy, env, verbose=1, tensorboard_log=f"./ckp/log/{params['env_name']}_ppo/", gae_lambda=0.0,
    )
    print(model.policy)
    model.learn(total_timesteps=1000_000)
    model.save(f"./ckp/pth/{params['env_name']}_ppo_100w.pth")
    env.close()
    
if __name__ == "__main__":
    set_seed(params["seed"])
    main()