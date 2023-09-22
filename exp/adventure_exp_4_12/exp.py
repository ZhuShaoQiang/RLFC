# -*- coding: utf-8 -*-

"""
这个里面存放经验
这个是4 12的冒险环境
"""
import numpy as np

### 坐标的经验，下面初始化的时候，需要根据需要转化为onehot的形式。保留坐标形式是因为不知道使用者使用坐标形式还是onehot的形式
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
"""
adventured的环境代码，是4*12大小的
nrows
     0  1  2  3  4  5  6  7  8  9  10   11  ncols
    ---------------------------------------
0  |   |  |O |  |  |  |  |  |  |  |   |   |
    ---------------------------------------
1  |   |  |X |  |  |  |  |  |  |  |   |   |
    ---------------------------------------
2  |   |  |X |  | ^|  |  |  |  |  |   |   |
    ---------------------------------------
3  | * |  |  |  |  |  |  | ^|  |  |   | e |
    ---------------------------------------
*: start point
X: trap, die
e: goal, endpoint, win
O: G6
^: G3

"""

### 训练经验
__exps_pos = np.asarray([
    [
        [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11],
    ],
    [
        [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [3, 8],
    ],
    [
        [2, 4], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8],
    ],
    [
        [1, 3], [1, 4], [2, 4], [2, 5], [2, 6], [2, 7],
    ],
    [
        [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [2, 4],
    ],
    [
        [1, 1], [0, 1], [0, 2], [0, 3], [0, 4], [1, 4],
    ],
    [
        [3, 0], [2, 0], [1, 0], [0, 0], [0, 1], [0, 2],
    ],
    [
        [3, 0], [3, 1], [2, 1], [1, 1], [0, 1], [0, 2],
    ],
])

### 测试经验, 现在不是用测试经验
__test_exp_pos = np.asarray([
    [
        [1, 2],[1, 2],[1, 3],[1, 4],[2, 2],[2,2]
    ],
])

__onehots = None

def generate_exps(onehot: bool=True, total_col: int=None, total_row: int=None, **kwargs):
    """
    产生经验，一次产生一个片段中的内容
    onehot: 产生的坐标值是坐标还是Onehot
    **kwargs：用于接受多余的字典变量
    """
    if onehot:  # 如果需要产生one_hot的形式
        assert total_col != None or total_row != None, f"产生onehot类型的坐标序列经验的时候，需要在params中添加 total_col和total_row 参数"

        global __onehots
        __onehots = np.eye(total_col*total_row)
        # print(__onehots)
        def __to_onehot(pos):
            """
            把一个坐标变为onehot向量，利用__onehots
            """
            assert not __onehots is None, f"你没有初始化__onehots这个矩阵"
            n = pos[0]*total_col + pos[1]
            return __onehots[n]
        # print(__exps_pos)
        # 把坐标全都转为one_hot
        ### 训练集
        tmp = []
        for i in __exps_pos:
            # idx = np.random.choice(range(len(__exps_pos)))
            # i = __exps_pos[idx]
            i = list(
                    map(__to_onehot, i)
                )
            tmp.append(i)

        ### 测试集
        test_tmp = []
        for i in __test_exp_pos:
            # idx = np.random.choice(range(len(__exps_pos)))
            # i = __exps_pos[idx]
            i = list(
                    map(__to_onehot, i)
                )
            test_tmp.append(i)
        return np.asarray(tmp), np.asarray(test_tmp)
    return __exps_pos, __test_exp_pos

if __name__ == "__main__":
    """
    测试加载经验的函数的
    """
    pass