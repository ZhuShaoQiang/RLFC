# -*- coding: utf-8 -*-

from gym import Env

"""
存放算法的文件，如PPO
此处的PPO算法我重新写，不按照之前的写了，之前的每收集2048步训练一次不妥当，在这个环境下应该每次死亡后，训练一次
"""

from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    """
    本文件所有的算法都会继承这个类，作为基础算法
    """
    def __init__(self, env: Env, params: dict) -> None:
        """
        env: 环境
        params: 一个字典，是config文件中的配置文件信息中的字典
        """
        super().__init__()
        self.env = env
        self.params = params


class VanillaPPO(BaseAlgorithm):
    """
    普通的PPO的算法
    """
    pass

# TODO: 先实现自己的算法
class FVRL_GRID(BaseAlgorithm):
    """
    自己的算法，但是是网格世界版本
    """
    def __init__(self, env: Env, params: dict) -> None:
        super().__init__(env, params)
