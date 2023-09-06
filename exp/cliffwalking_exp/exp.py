# -*- coding: utf-8 -*-

"""
这个里面存放经验
"""
import numpy as np

### 坐标的经验，下面初始化的时候，需要根据需要转化为onehot的形式。保留坐标形式是因为不知道使用者使用坐标形式还是onehot的形式
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
__exps_pos = np.asarray([
    [
        [0, 9],[1, 9],[1, 10],[1, 11],
    ],
    [
        [0, 10],[1, 10],[2, 10],[2, 11],
    ],
    [
        [0, 4],[0, 5],[0, 6],[1, 6],
    ],
    [
        [1, 5],[2, 5],[2, 6],[2, 7],
    ],
    [
        [0, 4],[0, 5],[1, 5],[2, 5],
    ],
    [
        [0, 3],[1, 3],[1, 4],[2, 4],
    ],
    [
        [0, 2],[1, 2],[1, 3],[2, 3],
    ],
    [
        [1, 7],[1, 8],[1, 9],[2, 9],
    ],
    [
        [0, 7],[1, 7],[2, 7],[2, 8],
    ],
    [
        [1, 4],[1, 5],[1, 6],[2, 6],
    ],
    [
        [1, 2],[2, 2],[2, 3],[2, 4],
    ],
    [
        [0, 11],[1, 11],[2, 11],[3,11],
    ],
    [
        [0, 11],[1, 11],[2, 11],[3,11],
    ],
    [
        [2, 9],[2, 10],[2, 11],[3, 11],
    ],
    [
        [3, 0],[2, 0],[2, 1],[2,2],
    ],
    [
        [0, 0],[0, 1],[0, 2],[0, 3],
    ],
    [
        [0, 0],[1, 0],[1, 1],[1, 2],
    ],
])

__test_exp_pos = np.asarray([
    [
        [0, 4],[1, 4],[2, 4],[2, 5],
    ],
    [
        [0, 7],[0, 8],[0, 9],[0, 10],
    ],
    [
        [0, 1],[0, 2],[0, 3],[1,3],
    ],
    [
        [1, 0],[2, 0],[2, 1],[2,2],
    ],
    [
        [0, 8],[1, 8],[2, 8],[2,9],
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
        assert total_col != None and total_row != None, f"产生onehot类型的坐标序列经验的时候，需要在params中添加 total_col和total_row 参数"
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