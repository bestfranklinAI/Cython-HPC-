# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

def cython_gauss_seidel(np.ndarray[double, ndim=2] f, int iterations=1000):
    """Cython-optimized Gauss-Seidel solver (Task 1.4)."""
    cdef int i, j, it, n = f.shape[0]
    cdef int m = f.shape[1]
    cdef np.ndarray[double, ndim=2] newf = f.copy()
    
    for it in range(iterations):
        for i in range(1, n-1):
            for j in range(1, m-1):
                newf[i, j] = 0.25 * (newf[i, j+1] + newf[i, j-1] +
                                      newf[i+1, j] + newf[i-1, j])
    return newf