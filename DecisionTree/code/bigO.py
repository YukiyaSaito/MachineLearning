import numpy as np
import time

def func1(N):
    t1 = time.time()
    for i in range(N):
        print("Hello!")

    print("%f seconds" % (time.time()-t1))
        
def func2(N):
    t1 = time.time()

    x = np.zeros(N)
    x += 1000
    print("%f seconds" % (time.time()-t1))
    return x

def func3(N):
    t1 = time.time()
    x = np.zeros(1000)
    x = x * N
    print("%f seconds" % (time.time()-t1))
    return x


def func4(N):
    t1 = time.time()
    x = 0
    for i in range(N):
        for j in range(i,N):
            x += (i*j)

    print("%f seconds" % (time.time()-t1))
    return x

