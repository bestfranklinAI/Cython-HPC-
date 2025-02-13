import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int mandelbrot_cython(double complex c, int max_iter=100):
    cdef double complex z = 0
    cdef int n
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

@cython.boundscheck(False)
@cython.wraparound(False)
def mandelbrot_set_cython(int width, int height, double x_min, double x_max, double y_min, double y_max, int max_iter=100):
    cdef int i, j
    cdef double[:] x_vals = np.linspace(x_min, x_max, width)
    cdef double[:] y_vals = np.linspace(y_min, y_max, height)
    cdef int[:, :] image = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            image[i, j] = mandelbrot_cython(complex(x_vals[j], y_vals[i]), max_iter)

    return np.asarray(image)