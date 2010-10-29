#cython: boundscheck=False
"""
Mandelbrot Cython
-----------------
This code uses complex numbers requiring a recent version of Cythnon > 0.11.2
"""


from numpy import empty, zeros
cimport numpy as np

cdef int mandelbrot_escape(float complex c, int n):
    """ Mandelbrot set escape time algorithm in real and complex components
    """
    cdef float complex z
    cdef int i
    z = 0 # DP : Ian does initialise it to zero, we do initialise it to c
    for i in range(n):
        z = z*z + c
        if z.real*z.real + z.imag*z.imag > 4.0:  # DP :  was >=
           break
    else:
        i = 0 # DP : was returning -1 when not enough iterations ...
    return i

def generate_mandelbrot(np.ndarray[float, ndim=1] xs, np.ndarray[float, ndim=1] ys, int n):
    """ Generate a mandelbrot set """
    cdef unsigned int i,j
    cdef unsigned int N = len(xs)
    cdef unsigned int M = len(ys)
    cdef float complex z
    
    cdef np.ndarray[int, ndim=2] d = empty(dtype='i', shape=(M, N))
    for j in range(M):
        for i in range(N):
            z = xs[i] + ys[j]*1j
            d[j,i] = mandelbrot_escape(z, n)
    return d
