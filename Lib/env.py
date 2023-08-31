# -*- coding: utf-8 -*-

"""
自定义的实验环境
"""

from typing import Tuple
import numpy as np

from abc import ABC, abstractmethod

class BaseGrid(ABC):
    def __init__(self, step_reward: float, dead_reward: float, goal_reward: float,
                 total_col: int, total_row: int) -> None:
        super().__init__()
        self.step_reward = step_reward  # 每一步的奖励
        self.dead_reward = dead_reward  # 死亡奖励
        self.goal_reward = goal_reward  # 到达终点的奖励
        self.total_col = total_col
        self.total_row = total_row  # 行列数

        self.pos = None  # 使用时候必须实现初始化

    @abstractmethod
    def reset(self):
        """
        重置环境
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        执行动作
        """
        pass

class CliffWalking(BaseGrid):
    """
    悬崖徒步的环境代码
    nrows
         0  1  2  3  4  5  6  7  8  9  10   11  ncols
    ---------------------------------------
    0  |   |  |  |  |  |  |  |  |  |  |   |   |
    ---------------------------------------
    1  |   |  |  |  |  |  |  |  |  |  |   |   |
    ---------------------------------------
    2  |   |  |  |  |  |  |  |  |  |  |   |   |
    ---------------------------------------
    3  | * |       cliff                  | ^ |
    *: start point
    cliff: cliff
    ^: goal
    """
    def __init__(self, step_reward: float=-0.1, dead_reward: float=-10, goal_reward: float=10,
                 total_col: int=12, total_row: int=4) -> None:
        super().__init__(step_reward, dead_reward, goal_reward,
                         total_col=total_col, total_row=total_row)
        self.move = np.array([
            [-1, 0],    # 向上，就是x-1， y不动,
            [1, 0],     # 向下，就是x+1， y不动,
            [0, -1],    # 向左，就是y-1， x不动,
            [0, 1],     # 向右，就是y+1， x不动,
        ], dtype=np.int8)
        """
        假设现在位置是[1, 3]，分别执行动作(self.pos + self.move[action])后：
        0 上: 0, 3
        1 下：2, 3
        2 左：1, 2
        3 右: 1, 4
        得证：move是正确的，但是需要在step中编写防止超界得代码
        """

        self.win_pos = np.array([self.total_row-1, self.total_col-1], dtype=np.int8)
        self.die_pos = np.array([], dtype=np.int8)
        for c in range(2, self.total_col-1):
            np.vstack((self.die_pos, [self.total_row-1, c]), dtype=np.int8)


    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        初始化agent的位置，初始位置必在左下角的起点
        """
        self.pos = np.array([self.total_row-1, 0], dtype=np.int8)  # 位置不可能为负
        return self.pos, {"pos": self.pos}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行动作
        action: 0: 上, 1: 下, 2: 左, 3: 右
        如果在边界处执行位置到达了边界之外（不是掉下悬崖），那么位置不变
        返回值： 新位置，奖励，通关(成功为True), 死亡(死亡为True)，info信息
        """
        if action < 0 or action >= len(self.move):
            # 如果动作<0或者>=4，都是超界了
            raise f"[-] 期望动作范围是[0, 3], 分别代表 上 下 左 右，得到了动作: {action}"
        self.pos = self.pos + self.move[action]  # 执行动作
        self.__process_exceedings()  # 处理超界坐标

        win = self.__is_win()
        die = self.__is_die()
        assert not (win and die), f"pos有问题，die和win同时发生了"  # 这个地方不能同时为True

        return self.pos, self.step_reward, win, die, {"win": win, "die": die, "pos": self.pos, "reward": self.step_reward}

    def __process_exceedings(self):
        """
        处理超界坐标
        """
        self.pos[self.pos < 0] = 0  # 小于0的全部给0
        self.pos[0] = min(self.pos[0], self.total_row - 1)  # 处理行数超界
        self.pos[1] = min(self.pos[1], self.total_col - 1)  # 处理列数超界

    def __is_die(self):
        """
        判断此处是不是死了，死了为True
        """
        return self.pos in self.die_pos

    def __is_win(self):
        """
        判断此处是不是赢了，赢了为True
        """
        return (self.pos == self.win_pos).all()