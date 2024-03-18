class A:
    def __init__(self):
        print("Initializing A")

class B(A):
    def __init__(self):
        super().__init__()  # 调用父类 A 的 __init__ 方法
        print("Initializing B")

class C(B):
    def __init__(self):
        super().__init__()  # 调用父类 B 的 __init__ 方法
        print("Initializing C")

class D:
    def __init__(self):
        print("Initializing D")

class E(D, C):
    def __init__(self):
        super().__init__()  # 调用父类 C 的 __init__ 方法
        super(D, self).__init__()  # 调用父类 D 的 __init__ 方法
        print("Initializing E")

e = E()