"""
一些env的包裹，因为sb3不记录奖励曲线，使用env的wrapper手动记录每个回合的奖励曲线
"""

import gymnasium
import numpy as np

from .logger import Logger

class RecordRewWrapper(gymnasium.Wrapper):
    """
    记录奖励的wrapper
    """
    def __init__(self, env, rew_log_dir, avg_n, *args, **kwargs) -> None:
        super(RecordRewWrapper, self).__init__(env, *args, **kwargs)
        self.logger = Logger(rew_log_dir)
        self.single_rew = 0
        self.rews = []
        self.rew_idx = 0
        self.avg_n = avg_n
    
    def step(self, *args, **kwargs):
        tmp = super().step(*args, **kwargs)
        self.single_rew += tmp[1]  # 不管什么版本的，都是下标为1是奖励
        return tmp
    
    def reset(self, *args, **kwargs):
        ### 处理之前的分数并记录
        self.rews.append(self.single_rew)
        self.logger.record_num("reward", self.single_rew, self.rew_idx)
        self.logger.record_num("reward_avg_n", np.mean(self.rews[-self.avg_n:]), self.rew_idx)
        ### 清理之前的信息
        self.single_rew = 0
        self.rew_idx += 1

        return super().reset(*args, **kwargs)

class RecordWinStepWrapper(gymnasium.Wrapper):
    """
    记录完成一个任务关卡的步数
    """
    def __init__(self, env, steps_log_dir, avg_n, *args, **kwargs) -> None:
        super(RecordWinStepWrapper, self).__init__(env, *args, **kwargs)
        self.logger = Logger(steps_log_dir)
        self.single_step = 0
        self.steps = []
        self.step_idx = 0
        self.avg_n = avg_n
    
    def step(self, *args, **kwargs):
        tmp = super().step(*args, **kwargs)
        self.single_step += 1  # 不管什么版本的，都是下标为1是奖励
        return tmp
    
    def reset(self, *args, **kwargs):
        ### 处理之前的分数并记录
        self.steps.append(self.single_step)
        self.logger.record_num("steps", self.single_step, self.step_idx)
        self.logger.record_num("steps_avg_n", np.mean(self.steps[-self.avg_n:]), self.step_idx)
        ### 清理之前的信息
        self.single_step = 0
        self.step_idx += 1

        return super().reset(*args, **kwargs)

