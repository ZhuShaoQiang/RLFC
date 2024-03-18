# -*- coding: utf-8 -*-

"""
cliff walking的dqn的sb的实现
"""

import sys
import os
sys.path.append(os.getcwd())

import gymnasium
import torch
from torch import nn
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy, CnnPolicy
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

## 分别记录胜利的平均奖励和胜利的平均步数
from utils.envwrappers import RecordRewWrapper, RecordWinStepWrapper
from utils.scorer_funcs import make_linear_decayable_scorer_func
from utils.scorers import RLFCScorer, FeedForward
from utils.utils import ZFilter, set_seed
from utils.algorithmwrappers import FVRL_PPO_3, FVRL_DQN
from config.config_dqn_minigrid_rlfc import params

# 来自minigrid官方的对于网络结构的修改，自己不做修改
### https://minigrid.farama.org/content/training/
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def main():
    env = gymnasium.make(params["env_name"])
    env = ImgObsWrapper(env)
    env = RecordRewWrapper(env, rew_log_dir=f"./ckp/log/rew/{params['env_name']}/dqn_fvrl_decal_2.0_10w", avg_n=10)
    env = RecordWinStepWrapper(env, steps_log_dir=f"./ckp/log/steps/{params['env_name']}/dqn_fvrl_decal_2.0_10w", avg_n=10)
    env = DummyVecEnv([lambda: env])

    """
    scorer = RLFCScorer(
        input_dim=int(env.observation_space.n * params["time_len"]),
        output_dim=1,
        activation=torch.nn.LeakyReLU(),
        hidden=[32*3, 16*3], last_activation=params["scorer_activation"]
    ).to(params["device"])
    scorer.load_state_dict(
        torch.load(params["SCORER_PATH"]), strict=True
    )
    scorer_func = make_linear_decayable_scorer_func(2, 100_000)
    """
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    model = DQN(
        "CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
        tensorboard_log=f"./ckp/log/{params['env_name']}_dqn/",
    )
    print(model.policy)

    model.learn(200_000)
    env.close()


if __name__ == "__main__":
    set_seed(42)
    main()
