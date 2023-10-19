# -*- coding: utf-8 -*-

"""
这个文件用来测试自己的想法
"""

class A:
    def __init__(self) -> None:
        print("1")
        self.a = None
    def printa(self):
        print(f"a = {self.a}")

class B(A):
    def __init__(self) -> None:
        super().__init__()
        print("2")
        self.a = 2
        self.printa()
b = B()