import multiprocessing
import numpy as np
import ctypes
import time

class Test:
    def __init__(self):
        self.x = 0

    def fit(self, x):
        self.x = x

    def score(self, y):
        return self.x * y

    def wrapp(self, x, y, res, index):
        self.fit(x)
        print(id(self.x))
        time.sleep(np.random.rand() * 5)
        res[index] = self.score(y)

processes = []

shared_array_base = multiprocessing.Array(ctypes.c_double, 10)
res = np.ctypeslib.as_array(shared_array_base.get_obj())
res = res.reshape(10)

test = Test()
for i in range(10):
    p = multiprocessing.Process(target=test.wrapp, args=(i, 2, res, i))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

print(res)
