# Mandelbrot calculate using Serial numpy and faster numpy
# ian@ianozsvald.com 2010
# based on the GPU version:
# http://wiki.tiker.net/PyCuda/Examples/Mandelbrot

# Based on vegaseat's TKinter/numpy example code from 2006
# http://www.daniweb.com/code/snippet216851.html#
# with minor changes to move to numpy from the obsolete Numeric

import time
import sys

from nose.tools import assert_almost_equals, assert_true

#from tables import numexpr 

import numpy as nm
import numpy

# set width and height of window, more pixels take longer to calculate
# note that we calculate less than w*h pixels, this is just the window
# size, see xx and yy (far below, printed to console) for the actual
# array sizes
w = 400
h = 400

TESTS = -2.13, 0.77, -1.3, 1.3, 1000

def calculate_z_numpy_old(q, maxiter, z):
    # calculate z using numpy, this is the original
    # routine from vegaseat's URL
    # NOTE this routine was faster using a default of double-precision complex nbrs
    # rather than the current single precision
    output = nm.resize(nm.array(0,), q.shape) 
    for iter in range(maxiter):       
        z = z*z + q        
        done = nm.greater(abs(z), 2.0)
        q = nm.where(done,0+0j, q)
        z = nm.where(done,0+0j, z)
        output = nm.where(done, iter, output)
    return output

def calculate_z_serial_old(q, maxiter, z):
    # calculate z using pure python with numpy arrays
    # this routine unrolls calculate_z_numpy as an intermediate
    # step to the creation of calculate_z_gpu
    # it runs slower than calculate_z_numpy
    output = nm.resize(nm.array(0,), q.shape)
    for i in range(len(q)):
        if i % 100 == 0:
            # print out some progress info since it is so slow...
            print "%0.2f%% complete" % (1.0/len(q) * i * 100)
        for iter in range(maxiter):
            z[i] = z[i]*z[i] + q[i]
            if abs(z[i]) > 2.0:
                q[i] = 0+0j
                z[i] = 0+0j
                output[i] = iter
    return output    

def calculate_z_serial(q, maxiter, z):
    # calculate z using pure python with numpy arrays
    # this routine unrolls calculate_z_numpy as an intermediate
    # step to the creation of calculate_z_gpu
    # it runs slower than calculate_z_numpy
    output = nm.resize(nm.array(0,), q.shape)    
    for i in xrange(len(q)):
        if i % 100 == 0:
            # print out some progress info since it is so slow...
            print "%0.2f%% complete" % (1.0/len(q) * i * 100)
        for iter in range(maxiter):
            z[i] = z[i]*z[i] + q[i]
            if abs(z[i]) > 2.0:
                q[i] = 0+0j
                z[i] = 0+0j
                output[i] = iter
                break
    return output
     
    
def calculate_z_numpy(q, maxiter, z):
    # calculate z using numpy, this is the original
    # routine from vegaseat's URL
    # NOTE this routine was faster using a default of double-precision complex nbrs
    # rather than the current single precision
    output = nm.zeros(shape=q.shape, dtype=nm.int32)    
    for iter in range(maxiter):        
        z = z*z + q        
        done = nm.greater(nm.abs(z), 2.0)
        q = nm.where(done, 0+0j, q)
        z = nm.where(done, 0+0j, z)
        output = nm.where(done, iter, output)        
    return output    
    
    
def test_mandelbrot(fun, x1, x2, y1, y2, maxiter=300, raw=False):
        # force yy, q and z to use 32 bit floats rather than
        # the default 64 doubles for nm.complex for consistency with CUDA
        xx = nm.arange(x1, x2, (x2-x1)/w*2, dtype=nm.float32)
        yy = nm.arange(y2, y1, (y1-y2)/h*2, dtype=nm.float32) * 1j
        print "Calculation sizes: xx and yy are ", len(xx), len(yy)
        q = nm.ravel(xx+yy[:, nm.newaxis]).astype(nm.complex64)
        z = nm.zeros_like(q)         

        # with pyCUDA I use their high resolution timer,
        # here I'm using time.time() which should be good enough
        # on Win and Lin for comparisons
        time_start = time.time()
        output = fun(q, maxiter, z)
        time_end = time.time()
        secs = time_end - time_start
        print "Main took", secs
        if raw:
            return output
        else:
            return (output + (256*output) + (256**2)*output) * 8
        
def test_numpy():
    x1, x2, y1, y2, maxiter = TESTS
    fun = calculate_z_numpy
    test_mandelbrot(fun, x1, x2, y1, y2, maxiter)

def test_python():
    x1, x2, y1, y2, maxiter = TESTS
    fun = calculate_z_serial
    test_mandelbrot(fun, x1, x2, y1, y2, maxiter)    

def test_cython():    
    x1, x2, y1, y2, maxiter = TESTS
    try:
        import mandel    
        xx = nm.arange(x1, x2, (x2-x1)/w*2, dtype=nm.float32)
        yy = nm.arange(y2, y1, (y1-y2)/h*2, dtype=nm.float32)
        time_start = time.time()
        res = mandel.generate_mandelbrot(xx, yy, maxiter)
        time_end = time.time()
        cython_secs = time_end - time_start
        print "Cython took ", cython_secs
    
        fun = calculate_z_numpy
        x1, x2, y1, y2, maxiter = TESTS
        res2 = test_mandelbrot(fun, x1, x2, y1, y2, maxiter, True)    
        
        print "Debug info"
        print res.shape, res.dtype
        print res.ravel()[:100]
        print res2.shape, res2.dtype
        print res2[:100]    
        print abs(1.0 * nm.sum(res - res2.reshape((len(xx),len(yy)))) /  nm.sum(res)) * 100
        print nm.sum(res == -1)    
        assert_true(nm.allclose(res, res2.reshape((len(xx),len(yy)))))        
    except ImportError:
        print "You must compile the Cython extension"        
    
def test_compare_python():
    x1, x2, y1, y2, maxiter = TESTS
    fun = calculate_z_serial
    res = test_mandelbrot(fun, x1, x2, y1, y2, maxiter)    
    fun = calculate_z_serial_old
    res2 = test_mandelbrot(fun, x1, x2, y1, y2, maxiter)    
    assert_true(nm.allclose(res, res2))
    
def test_compare_numpy():
    x1, x2, y1, y2, maxiter = TESTS
    print "Old numpy"
    fun = calculate_z_numpy_old
    res2 = test_mandelbrot(fun, x1, x2, y1, y2, maxiter)        
    print "New numpy"
    fun = calculate_z_numpy
    res = test_mandelbrot(fun, x1, x2, y1, y2, maxiter)        
    assert_true(nm.allclose(res, res2, atol=0.1))

ROUTINES = { 'numpy':calculate_z_numpy,
             'python':calculate_z_serial }
               
# test the class
if __name__ == '__main__':
    if len(sys.argv) == 1:
        routines = ",".join(ROUTINES.keys())
        print "Usage: python mandelbrot.py", routines
        print "Where:"
        #print " gpu is a pure CUDA solution on the GPU"
        #print " gpuarray uses a numpy-like CUDA wrapper in Python on the GPU"
        print " numpy is a pure Numpy (C-based) solution on the CPU"
        print " python is a pure Python solution on the CPU with numpy arrays"
    else:
            
        if sys.argv[1] == 'python':
            try:                
                import psyco
                psyco.full()
                print "USING PSYCO"
            except ImportError, e:
                print "No psyco"

        
        if len(sys.argv) > 1:
            if sys.argv[1] not in ROUTINES.keys():
                show_instructions = True

        calculate_z = ROUTINES[sys.argv[1]]



        # Using a WinXP Intel Core2 Duo 2.66GHz CPU (1 CPU used)
        # with a 9800GT GPU I get the following timings (smaller is better).
        # With 200x200 problem with max iterations set at 300:
        # calculate_z_gpu: 0.03s 
        # calculate_z_serial: 8.7s
        # calculate_z_numpy: 0.3s
            
        #test_mandelbrot(calculate_z, *TESTS)
       
        test_cython()

