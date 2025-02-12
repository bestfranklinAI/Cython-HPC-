import numpy as np
import random
from line_profiler import profile

@profile
def gauss_seidel(f):
    newf = f.copy()
    for i in range(1, newf.shape[0] - 1):
        for j in range(1, newf.shape[1] - 1):
            newf[i, j] = 0.25 * (newf[i, j + 1] + newf[i, j - 1] +
                                 newf[i + 1, j] + newf[i - 1, j])

    return newf

if __name__ == "__main__":
    N = 100
    x = np.array([[random.uniform(1, 100) for _ in range(N)] for _ in range(N)])
    for i in range(1000):
        x = gauss_seidel(x)