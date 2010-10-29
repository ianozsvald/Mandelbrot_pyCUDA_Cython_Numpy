# Mandelbrot calculate using GPU, Serial numpy and faster numpy
# Use to show the speed difference between CPU and GPU calculations
# ian@ianozsvald.com March 2010

# Based on vegaseat's TKinter/numpy example code from 2006
# http://www.daniweb.com/code/snippet216851.html#
# with minor changes to move to numpy from the obsolete Numeric

import sys
import numpy as nm

import Tkinter as tk
import Image          # PIL
import ImageTk        # PIL

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import pycuda.gpuarray as gpuarray

# You can choose a calculation routine below (calculate_z), uncomment
# one of the three lines to test the three variations
# Speed notes are listed in the same place

# set width and height of window, more pixels take longer to calculate
w = 400
h = 400


complex_gpu_sm_lightweight = SourceModule("""

extern "C++"
template<class T>
class gcomplex {

public:
	T real;
	T imag;

	__device__ gcomplex()
		: real(0.0),
		  imag(0.0)
	{
	};

	__device__ gcomplex(T r, T i)
		: real(r),
		  imag(i)
	{
	};

	__device__ bool operator==(const gcomplex& gcomplex2) {
		return real == gcomplex2.real && imag == gcomplex2.imag;
	}

	__device__ gcomplex operator+(const gcomplex& gcomplex2) {
		return gcomplex<T>(real + gcomplex2.real, imag + gcomplex2.imag);
	}

	__device__ gcomplex operator-(const gcomplex& gcomplex2) {
		return gcomplex<T>(real - gcomplex2.real, imag - gcomplex2.imag);
	}

	__device__ gcomplex operator*(const gcomplex& gcomplex2) {		
		T temp1 = (real * gcomplex2.real) - (imag * gcomplex2.imag);
		T temp2 = (imag * gcomplex2.real) + (real * gcomplex2.imag);
		return gcomplex<T>(temp1, temp2);
	}

	__device__ gcomplex operator/(const gcomplex& gcomplex2) {		
		T temp1 = (real * gcomplex2.real) + (imag * gcomplex2.imag);
		T temp2 = (imag * gcomplex2.real) - (real * gcomplex2.imag);
		T temp3 = (gcomplex2.real * gcomplex2.real) + (gcomplex2.imag * gcomplex2.imag);		
		return gcomplex<T>(temp1/temp3, temp2/temp3);
	}

};

extern "C++"
template <class T>
__device__ T abs(const gcomplex<T>& g) {
    return sqrt((g.real * g.real) + (g.imag * g.imag));
}

        __global__ void calc_gpu_sm(gcomplex<float> *z, gcomplex<float> *q, int *iteration, int maxiter) {
            const int i = blockDim.x * blockIdx.x + threadIdx.x;
            //const gcomplex<float> gc0;
            iteration[i] = 0;
            for (int n=0; n < maxiter; n++) {
                
                z[i] = (z[i]*z[i])+q[i]; 
                if (abs(z[i]) > 2.0f) {
                    iteration[i]=n; 
                    z[i] = gcomplex<float>(); 
                    q[i] = gcomplex<float>();
                    //z[i] = gc0;
                    //q[i] = gc0;
                }
            };            
        }

        """, keep=True)

calc_gpu_sm_lightweight = complex_gpu_sm_lightweight.get_function('calc_gpu_sm')
print 'complex_gpu_sm:'
print 'Registers', calc_gpu_sm_lightweight.num_regs
print 'Local mem', calc_gpu_sm_lightweight.local_size_bytes, 'bytes'
print 'Shared mem', calc_gpu_sm_lightweight.shared_size_bytes, 'bytes'

complex_gpu_sm = SourceModule("""
        #include <pycuda-complex.hpp>


        __global__ void calc_gpu_sm(pycuda::complex<float> *z, pycuda::complex<float> *q, int *iteration, int maxiter) {
            const int i = blockDim.x * blockIdx.x + threadIdx.x;
            //printf("%d %d %d %d\\n", gridDim.x, blockDim.x, blockIdx.x, threadIdx.x);
            iteration[i] = 0;
            for (int n=0; n < maxiter; n++) {
                z[i] = (z[i]*z[i])+q[i]; 
                if (abs(z[i]) > 2.0f) {
                    iteration[i]=n; 
                    z[i] = pycuda::complex<float>(); 
                    q[i] = pycuda::complex<float>();
                }
            };            
        }


        """, keep=True)

calc_gpu_sm = complex_gpu_sm.get_function('calc_gpu_sm')
print 'complex_gpu_sm:'
print 'Registers', calc_gpu_sm.num_regs
print 'Local mem', calc_gpu_sm.local_size_bytes, 'bytes'
print 'Shared mem', calc_gpu_sm.shared_size_bytes, 'bytes'


complex_gpu_sm_newindexing = SourceModule("""
/*
        // original newindexing code using original mandelbrot pycuda
        #include <pycuda-complex.hpp>

        __global__ void calc_gpu_sm(pycuda::complex<float> *z, pycuda::complex<float> *q, int *iteration, int maxiter, const int nbritems) {
            //const int i = blockDim.x * blockIdx.x + threadIdx.x;
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x * blockDim.x;
            unsigned cta_start = blockDim.x * blockIdx.x;

            for ( int i = cta_start + tid; i < nbritems; i += total_threads) {
                for (int n=0; n < maxiter; n++) {
                    z[i] = (z[i]*z[i])+q[i]; 
                    if (abs(z[i]) > 2.0f) {
                        iteration[i]=n; 
                        z[i] = pycuda::complex<float>(); 
                        q[i] = pycuda::complex<float>();
                    }
                };            
            }
        }
*/

extern "C++"
template<class T>
class gcomplex {

public:
	T real;
	T imag;

	__device__ gcomplex()
		: real(0.0),
		  imag(0.0)
	{
	};

	__device__ gcomplex(T r, T i)
		: real(r),
		  imag(i)
	{
	};

	__device__ bool operator==(const gcomplex& gcomplex2) {
		return real == gcomplex2.real && imag == gcomplex2.imag;
	}

	__device__ gcomplex operator+(const gcomplex& gcomplex2) {
		return gcomplex<T>(real + gcomplex2.real, imag + gcomplex2.imag);
	}

	__device__ gcomplex operator-(const gcomplex& gcomplex2) {
		return gcomplex<T>(real - gcomplex2.real, imag - gcomplex2.imag);
	}

	__device__ gcomplex operator*(const gcomplex& gcomplex2) {		
		T temp1 = (real * gcomplex2.real) - (imag * gcomplex2.imag);
		T temp2 = (imag * gcomplex2.real) + (real * gcomplex2.imag);
		return gcomplex<T>(temp1, temp2);
	}

	__device__ gcomplex operator/(const gcomplex& gcomplex2) {		
		T temp1 = (real * gcomplex2.real) + (imag * gcomplex2.imag);
		T temp2 = (imag * gcomplex2.real) - (real * gcomplex2.imag);
		T temp3 = (gcomplex2.real * gcomplex2.real) + (gcomplex2.imag * gcomplex2.imag);		
		return gcomplex<T>(temp1/temp3, temp2/temp3);
	}

};

extern "C++"
template <class T>
__device__ T abs(const gcomplex<T>& g) {
    return sqrt((g.real * g.real) + (g.imag * g.imag));
}

        __global__ void calc_gpu_sm(gcomplex<float> *z, gcomplex<float> *q, int *iteration, int maxiter, int nbritems) {
            //const int i = blockDim.x * blockIdx.x + threadIdx.x;
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x * blockDim.x;
            unsigned cta_start = blockDim.x * blockIdx.x;

            for ( int i = cta_start + tid; i < nbritems; i += total_threads) {            
                for (int n=0; n < maxiter; n++) {            
                    z[i] = (z[i]*z[i])+q[i]; 
                    if (abs(z[i]) > 2.0f) {
                        iteration[i]=n; 
                        z[i] = gcomplex<float>(); 
                        q[i] = gcomplex<float>();
                    }
                }
            };            
        }


        """, keep=True)

calc_gpu_sm_newindexing = complex_gpu_sm_newindexing.get_function('calc_gpu_sm')
print 'complex_gpu_sm:'
print 'Registers', calc_gpu_sm_newindexing.num_regs
print 'Local mem', calc_gpu_sm_newindexing.local_size_bytes, 'bytes'
print 'Shared mem', calc_gpu_sm_newindexing.shared_size_bytes, 'bytes'



#############


complex_gpu_sm_newindexing_double = SourceModule("""

extern "C++"
template<class T>
class gcomplex {

public:
	T real;
	T imag;

	__device__ gcomplex()
		: real(0.0),
		  imag(0.0)
	{
	};

	__device__ gcomplex(T r, T i)
		: real(r),
		  imag(i)
	{
	};

	__device__ bool operator==(const gcomplex& gcomplex2) {
		return real == gcomplex2.real && imag == gcomplex2.imag;
	}

	__device__ gcomplex operator+(const gcomplex& gcomplex2) {
		return gcomplex<T>(real + gcomplex2.real, imag + gcomplex2.imag);
	}

	__device__ gcomplex operator-(const gcomplex& gcomplex2) {
		return gcomplex<T>(real - gcomplex2.real, imag - gcomplex2.imag);
	}

	__device__ gcomplex operator*(const gcomplex& gcomplex2) {		
		T temp1 = (real * gcomplex2.real) - (imag * gcomplex2.imag);
		T temp2 = (imag * gcomplex2.real) + (real * gcomplex2.imag);
		return gcomplex<T>(temp1, temp2);
	}

	__device__ gcomplex operator/(const gcomplex& gcomplex2) {		
		T temp1 = (real * gcomplex2.real) + (imag * gcomplex2.imag);
		T temp2 = (imag * gcomplex2.real) - (real * gcomplex2.imag);
		T temp3 = (gcomplex2.real * gcomplex2.real) + (gcomplex2.imag * gcomplex2.imag);		
		return gcomplex<T>(temp1/temp3, temp2/temp3);
	}

};

extern "C++"
template <class T>
__device__ T abs(const gcomplex<T>& g) {
    return sqrt((g.real * g.real) + (g.imag * g.imag));
}

        __global__ void calc_gpu_sm(gcomplex<double> *z, gcomplex<double> *q, int *iteration, int maxiter, int nbritems) {
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x * blockDim.x;
            unsigned cta_start = blockDim.x * blockIdx.x;

            for ( int i = cta_start + tid; i < nbritems; i += total_threads) {            
                for (int n=0; n < maxiter; n++) {            
                    z[i] = (z[i]*z[i])+q[i]; 
                    if (abs(z[i]) > 2.0f) {
                        iteration[i]=n; 
                        z[i] = gcomplex<double>(); 
                        q[i] = gcomplex<double>();
                    }
                }
            };            
        }


        """, keep=True)

calc_gpu_sm_newindexing_double = complex_gpu_sm_newindexing_double.get_function('calc_gpu_sm')
print 'complex_gpu_sm:'
print 'Registers', calc_gpu_sm_newindexing_double.num_regs
print 'Local mem', calc_gpu_sm_newindexing_double.local_size_bytes, 'bytes'
print 'Shared mem', calc_gpu_sm_newindexing_double.shared_size_bytes, 'bytes'


#############



def calculate_z_gpu_sm(q, maxiter, z):
    output = nm.resize(nm.array(0,), q.shape)
    blocks = len(q)
    print "maxiter", maxiter
    print "len(q)", len(q)
    #calc_gpu_sm(drv.In(z), drv.In(q), drv.Out(output), numpy.int32(maxiter), grid=(blocks,1), block=(1,1,1))
    #calc_gpu_sm_lightweight(drv.In(z), drv.In(q), drv.Out(output), numpy.int32(maxiter), grid=(blocks,1), block=(1,1,1))
    calc_gpu_sm_newindexing(drv.In(z), drv.In(q), drv.InOut(output), numpy.int32(maxiter), numpy.int32(len(q)), grid=(480,1), block=(48,1,1))

    # double precision
    #z = z.astype(nm.complex128)
    #q = q.astype(nm.complex128)
    #output = output.astype(nm.complex128)
    #calc_gpu_sm_newindexing_double(drv.In(z), drv.In(q), drv.InOut(output), numpy.int32(maxiter), numpy.int32(len(q)), grid=(480,1), block=(512,1,1))

    return output

complex_gpu = ElementwiseKernel(
        """pycuda::complex<float> *z, pycuda::complex<float> *q, int *iteration, int maxiter""",
            """for (int n=0; n < maxiter; n++) {z[i] = (z[i]*z[i])+q[i]; if (abs(z[i]) > 2.00f) {iteration[i]=n; z[i] = pycuda::complex<float>(); q[i] = pycuda::complex<float>();};};""",
        "complex5",
        preamble="""#include <pycuda-complex.hpp>""",
        keep=True)

def calculate_z_gpu(q, maxiter, z):
    output = nm.resize(nm.array(0,), q.shape)
    q_gpu = gpuarray.to_gpu(q.astype(nm.complex64))
    z_gpu = gpuarray.to_gpu(z.astype(nm.complex64))
    iterations_gpu = gpuarray.to_gpu(output) 
    print "maxiter gpu", maxiter
    # the for loop and complex calculations are all done on the GPU
    # we bring the iterations_gpu array back to determine pixel colours later
    complex_gpu(z_gpu, q_gpu, iterations_gpu, maxiter)

    iterations = iterations_gpu.get()
    return iterations


def calculate_z_numpy_gpu(q, maxiter, z):
    """Calculate z using numpy on the GPU"""
    outputg = gpuarray.to_gpu(nm.resize(nm.array(0,), q.shape))
    zg = gpuarray.to_gpu(z.astype(nm.complex64))
    qg = gpuarray.to_gpu(q.astype(nm.complex64))
    # 2.0 as an array
    twosg = gpuarray.to_gpu(nm.array([2.0]*zg.size).astype(numpy.float32))
    # 0+0j as an array
    cmplx0sg = gpuarray.to_gpu(nm.array([0+0j]*zg.size).astype(nm.complex64))
    # for abs_zg > twosg result
    comparison_result = gpuarray.to_gpu(nm.array([False]*zg.size).astype(nm.bool))
    # we'll add 1 to iterg after each iteration
    iterg = gpuarray.to_gpu(nm.array([0]*zg.size).astype(nm.int32))
    
    for iter in range(maxiter):
        zg = zg*zg + qg

        # abs returns a complex (rather than a float) from the complex
        # input where the real component is the absolute value (which
        # looks like a bug) so I take the .real after abs()
        abs_zg = abs(zg).real
       
        comparison_result = abs_zg > twosg
        qg = gpuarray.if_positive(comparison_result, cmplx0sg, qg)
        zg = gpuarray.if_positive(comparison_result, cmplx0sg, zg)
        outputg = gpuarray.if_positive(comparison_result, iterg, outputg)
        iterg = iterg + 1
    output = outputg.get()
    return output


def calculate_z_numpy(q, maxiter, z):
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

def calculate_z_serial(q, maxiter, z):
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
 

show_instructions = False
if len(sys.argv) == 1:
    show_instructions = True

routine = {'gpuarray':calculate_z_numpy_gpu,
           'sourcemodule':calculate_z_gpu_sm,
            'gpu':calculate_z_gpu,
            'numpy':calculate_z_numpy,
            'python':calculate_z_serial}

if len(sys.argv) > 1:
    #if sys.argv[1] not in ['gpu', 'gpuarray', 'numpy', 'python']:
    if sys.argv[1] not in routine.keys():
        show_instructions = True

if show_instructions:
    routines = ",".join(routine.keys())
    print "Usage: python mandelbrot.py", routines
    print "Where:"
    print " gpu is a pure CUDA solution on the GPU"
    print " gpuarray uses a numpy-like CUDA wrapper in Python on the GPU"
    print " numpy is a pure Numpy (C-based) solution on the CPU"
    print " python is a pure Python solution on the CPU with numpy arrays"
    sys.exit(0)


calculate_z = routine[sys.argv[1]]

if sys.argv[1] == 'python':
    import psyco
    psyco.full()

# Using a WinXP Intel Core2 Duo 2.66GHz CPU (1 CPU used)
# with a 9800GT GPU I get the following timings (smaller is better).
# With 200x200 problem with max iterations set at 300:
# calculate_z_gpu: 0.03s 
# calculate_z_serial: 8.7s
# calculate_z_numpy: 0.3s


class Mandelbrot(object):
    def __init__(self):
        # create window
        self.root = tk.Tk()
        self.root.title("Mandelbrot Set")
        self.create_image()
        self.create_label()
        # start event loop
        self.root.mainloop()
               

    def draw(self, x1, x2, y1, y2, maxiter=300):
        # draw the Mandelbrot set, from numpy example
        xx = nm.arange(x1, x2, (x2-x1)/w*2)
        yy = nm.arange(y2, y1, (y1-y2)/h*2) * 1j
        print "xx and yy are ", len(xx), len(yy)
        # force yy, q and z to use 32 bit floats rather than
        # the default 64 doubles for nm.complex for consistency with CUDA
        yy = yy.astype(nm.complex64)
        q = nm.ravel(xx+yy[:, nm.newaxis]).astype(nm.complex64)
        z = nm.zeros(q.shape, nm.complex64)

        start_main = drv.Event()
        end_main  = drv.Event()
        start_main.record()
        output = calculate_z(q, maxiter, z)

        end_main.record()
        end_main.synchronize()
        secs = start_main.time_till(end_main)*1e-3
        print "Main took", secs

        output = (output + (256*output) + (256**2)*output) * 8
        # convert output to a string
        self.mandel = output.tostring()

    def create_image(self):
        """"
        create the image from the draw() string
        """
        self.im = Image.new("RGB", (w/2, h/2))
        # you can experiment with these x and y ranges
        self.draw(-2.13, 0.77, -1.3, 1.3, 1000)
        self.im.fromstring(self.mandel, "raw", "RGBX", 0, -1)

    def create_label(self):
        # put the image on a label widget
        self.image = ImageTk.PhotoImage(self.im)
        self.label = tk.Label(self.root, image=self.image)
        self.label.pack()

# test the class
if __name__ == '__main__':
    test = Mandelbrot()

