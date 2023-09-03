# -*- coding: utf-8 -*-

"""
这个里面写一些env的wrappers
通用的wrapper，比如把obs变成tensor的wrapper
"""

from typing import Optional, Tuple
import gym
from gym.core import Env
from gym import spaces
import torch
from torchvision import transforms
import numpy as np

class ToTensorWrapper(gym.Wrapper):
    def __init__(self, env: Env):
        super(ToTensorWrapper, self).__init__(env)
        pass
    
    def reset(self):
        obs, info = self.env.reset()
        return self._to_tensor(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_tensor(obs), reward, terminated, truncated, info
    
    def _to_tensor(self, obs):
        return torch.from_numpy(obs).float().unsqueeze(0)


class ImgToTensor(gym.Wrapper):
    """
    这个不仅仅会把img变成一个tensor，还会对img除以255.进行归一化
    """
    def __init__(self, env: Env, new_height, new_weight):
        super(ImgToTensor, self).__init__(env)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((new_height, new_weight)),
            transforms.ToTensor(),
        ])

    def reset(self):
        obs, info = self.env.reset()
        return self._to_imgtensor(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_imgtensor(obs), reward, terminated, truncated, info
    
    def _to_imgtensor(self, obs):
        return self.transform(obs).unsqueeze(0)

class ClipAction(gym.Wrapper):
    """
    这个会裁切动作到自己限定的最大最小值
    """
    def __init__(self, env: Env):
        super(ClipAction, self).__init__(env)

    def reset(self):
        obs, info = self.env.reset()
        return obs, info
    
    def step(self, action):
        if isinstance(self.env.action_space, gym.spaces.box.Box):
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

class FrameSkip(gym.Wrapper):
    def __init__(self, env: Env, skip: int=4):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class MultiEnvWrapper(gym.Env):
    def __init__(self, env: gym.Env, num_envs:int) -> None:
        """
        nums代表打包的env的数量
        """
        self.envs = [gym.make(env.spec.id) for _ in range(num_envs)]  # 创建n个相同的环境
        self.num_envs = num_envs

        self.observation_space = spaces.Tuple([self.envs[0].observation_space]*num_envs)
        self.action_space = spaces.Tuple([self.envs[0].action_space]*num_envs)
    
    def reset(self, *, seed: int = None, options: dict = None):
        """
        同时重置所有的环境
        """
        return tuple([env.reset(seed=seed, options=options) for env in self.envs])
    
    def step(self, action):
        obs, rewards, terminates, truncates, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            ob, reward, terminate, truncate, info = env.step(action[i])

            obs.append(ob)
            rewards.append(reward)
            terminates.append(terminate)
            truncates.append(truncate)
            infos.append(info)
        
        
            
        return super().step(action)


### 下面是进行一些攻击的方法
class RandomNoiseAttack(gym.Wrapper):
    """
    对env的产生的img state加一个随即噪音攻击

    注意，这个地方需要先执行ImgToTensor，后进行攻击
    """
    def __init__(self, env: gym.Env, epsilon=0.8, regularization=255.0):
        """
        epsilon: 噪音的无穷范数最大值，用于限制噪音的攻击范围，默认是0.8

        regularization: 因为img转为tensor之后，数值除以了255，所以这个地方的噪音也要除以255
        """
        super(RandomNoiseAttack, self).__init__(env)
        self.epsilon = epsilon / regularization

    def reset(self):
        obs, info = self.env.reset()
        obs = self._apply_random_noise(obs)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._apply_random_noise(obs), reward, terminated, truncated, info
    
    def _apply_random_noise(self, obs):
        """
        对传入的obs进行添加噪音的攻击
        注意这个时候是ndarray，因为还没有转成一个img tensor，转成之后就除以255了

        uniform是随机噪音，normal是高斯噪音
        """
        noise = torch.rand_like(obs, dtype=torch.float) * (2 * self.epsilon) - self.epsilon
        return torch.clamp(obs + noise, 0.0, 1.0)  # 防止超界