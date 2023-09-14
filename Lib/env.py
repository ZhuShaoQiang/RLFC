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
                 total_col: int=12, total_row: int=4, one_hot: bool=True) -> None:
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

        one_hot: step和reset返回的坐标是否是one_hot向量（但仍然会保留非one_hot的坐标）
        """
        self.total_step = 0  # 设置最大步数上限
        self.max_step = 1000  # 最大1K步
        self.win_pos = np.array([self.total_row-1, self.total_col-1], dtype=np.int8)
        self.die_pos = np.array([self.total_row-1, 1], dtype=np.int8)
        for c in range(2, self.total_col-1):
            self.die_pos = np.vstack((self.die_pos, [self.total_row-1, c]), dtype=np.int8)
        # print("win:", self.win_pos)
        # print("die:", self.die_pos)

        self.one_hot = one_hot
        if one_hot:
            self._one_hots = np.eye(self.total_col*self.total_row)  # 这是一个48*48的对角阵
            self.obs_space = self.total_col*self.total_row
        else:
            self.obs_space = 2

    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        初始化agent的位置，初始位置必在左下角的起点
        """
        self.total_step = 0
        self.pos = np.array([self.total_row-1, 0], dtype=np.int8)  # 位置不可能为负
        res_pos = self.__one_hot_pos() if self.one_hot else self.pos
        return res_pos, {"pos": self.pos}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行动作
        action: 0: 上, 1: 下, 2: 左, 3: 右
        如果在边界处执行位置到达了边界之外（不是掉下悬崖），那么位置不变
        返回值： 新位置，奖励，通关(成功为True), 死亡(死亡为True)，info信息
        """
        self.total_step += 1
        if action < 0 or action >= len(self.move):
            # 如果动作<0或者>=4，都是超界了
            raise f"[-] 期望动作范围是[0, 3], 分别代表 上 下 左 右，得到了动作: {action}"
        reward = self.step_reward
        self.pos = self.pos + self.move[action]  # 执行动作
        self.__process_exceedings()  # 处理超界坐标

        win = self.__is_win()
        die = self.__is_die()
        assert not (win and die), f"pos有问题，die和win同时发生了"  # 这个地方不能同时为True
        if win:
            reward = self.goal_reward
        else:
            if self.total_step >= self.max_step:
                # 一定步数走不完直接死
                die = True
        if die:
            reward = self.dead_reward

        res_pos = self.__one_hot_pos() if self.one_hot else self.pos
        return res_pos, reward, win, die, {"win": win, "die": die, "pos": self.pos, "reward": reward}
    
    def __one_hot_pos(self):
        """
        返回one_hot的坐标
        """
        n = int(
            (self.pos[0]*self.total_col + self.pos[1]).item()
        )
        return self._one_hots[n]

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
        # return self.pos in self.die_pos
        # FIXME: ndarray不能直接in，应该用下面的方法
        return np.any(np.all(self.pos == self.die_pos, axis=1))

    def __is_win(self):
        """
        判断此处是不是赢了，赢了为True
        """
        return (self.pos == self.win_pos).all()

class Adventure(BaseGrid):
    """
    自定义的实验的代码，也是一个网格世界，这个世界需要收集物品，收集的越多分数越多
    nrows
         0   1   2   3   4   5   6   ncols
       -----------------------------
    0  |   |   |   |   |   |   |   |
       -----------------------------
    1  |   | G6|   |   |   |   |   |
       -----------------------------
    2  |   | X |   |   |   |   |   |
       -----------------------------
    3  |   | X |   |   |   |   |   |
       -----------------------------
    4  |   | X |   |   |   |   |   |
       -----------------------------
    5  |   |   | G3|   |   |   |   |
       -----------------------------
    6  | * |   |   |   | G3|   | ^ |
       -----------------------------
    *: start point
    X: cliff, dead point
    G6: 分数点，6分
    G3: 分数点，3分
    ^: goal
    """
    def __init__(self, step_reward: float=-0.1, dead_reward: float=-10, goal_reward: float=10,
                 total_col: int=6, total_row: int=6, one_hot: bool=True) -> None:
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

        one_hot: step和reset返回的坐标是否是one_hot向量（但仍然会保留非one_hot的坐标）
        """
        self.total_step = 0  # 设置最大步数上限
        self.max_step = 1000  # 最大1K步
        # 这个不分赢不赢，就是到终点
        self.win_pos = np.array([self.total_row-1, self.total_col-1], dtype=np.int8)
        # 就是那两个陷阱的位置
        self.die_pos = np.array([[2, 1], [3, 1], [4, 1]], dtype=np.int8)

        # 创建一个vis数组
        self.vis = np.zeros((self.total_row, self.total_col))
        self.G3_pos = np.array([
            [5, 2],
            [6, 4],
        ])  # 3分
        self.G6_pos = np.array([
            [1, 1]
        ])  # 6分
        # print("win pos:", self.win_pos)
        # print("die pos:", self.die_pos)

        self.one_hot = one_hot  # 是否以OneHot标识坐标
        if one_hot:
            self._one_hots = np.eye(self.total_col*self.total_row)  # 这是一个48*48的对角阵
            self.obs_space = self.total_col*self.total_row
        else:
            self.obs_space = 2

    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        初始化agent的位置，初始位置必在左下角的起点
        初始化vis数组
        """
        self.vis = np.zeros((self.total_row, self.total_col))
        self.total_step = 0
        self.pos = np.array([self.total_row-1, 0], dtype=np.int8)  # 位置不可能为负
        res_pos = self.__one_hot_pos() if self.one_hot else self.pos
        return res_pos, {"pos": self.pos}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行动作
        action: 0: 上, 1: 下, 2: 左, 3: 右
        如果在边界处执行位置到达了边界之外（不是掉下悬崖），那么位置不变
        返回值： 新位置，奖励，通关(成功为True), 死亡(死亡为True)，info信息
        """
        self.total_step += 1
        if action < 0 or action >= len(self.move):
            # 如果动作<0或者>=4，都是超界了
            raise f"[-] 期望动作范围是[0, 3], 分别代表 上 下 左 右，得到了动作: {action}"
        reward = self.step_reward
        self.pos = self.pos + self.move[action]  # 执行动作
        self.__process_exceedings()  # 处理超界坐标

        win = self.__is_win()
        die = self.__is_die()
        assert not (win and die), f"pos有问题，die和win同时发生了"  # 这个地方不能同时为True

        G = self.__G_score()

        if win:
            reward = self.goal_reward
        else:
            if self.total_step >= self.max_step:
                # 一定步数走不完直接死
                die = True
            if G != 0:  # 只要不是0
                reward = G
            # 不赢的时候，就要看看有没有G分

        if die:
            reward = self.dead_reward

        res_pos = self.__one_hot_pos() if self.one_hot else self.pos
        return res_pos, reward, win, die, {"win": win, "die": die, "pos": self.pos, "reward": reward}
    
    def __one_hot_pos(self):
        """
        返回one_hot的坐标
        """
        n = int(
            (self.pos[0]*self.total_col + self.pos[1]).item()
        )
        return self._one_hots[n]

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
        死处有好多，是[[1, 2], [1, 3]]类型的
        """
        # return self.pos in self.die_pos
        # FIXME: ndarray不能直接in，应该用下面的方法
        return np.any(np.all(self.pos == self.die_pos, axis=1))

    def __is_win(self):
        """
        判断此处是不是赢了，赢了为True
        winpos只有一个，是[1, 2]形式的
        """
        return (self.pos == self.win_pos).all()
    
    def __G_score(self) -> int:
        """
        返回得到的G的得分
        没有时得到0
        得分处有好多，是[[1, 2], [1, 3]]类型的
        """
        if np.any(np.all(self.pos == self.G3_pos, axis=1)) and self.vis[self.pos[0], self.pos[1]] == 0:
            self.vis[self.pos[0], self.pos[1]] = 1  # 赋值为1
            return 3
        if np.any(np.all(self.pos == self.G6_pos, axis=1)) and self.vis[self.pos[0], self.pos[1]] == 0:
            self.vis[self.pos[0], self.pos[1]] = 1  # 赋值为1
            return 6
        return 0
