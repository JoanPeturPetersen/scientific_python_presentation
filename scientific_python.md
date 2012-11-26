Scientific Python
=================

This is still work in progress.

---

Background
==========

 * Worked on an industrial Ph. D. with DTU and GreenSteam.
 * Used python troughout the project
    * Data collection
    * Data processing
    * Modelling
    * Data presentation
    * Deployment

# Presenter Notes

Python can help you in all these steps from the beginning to the end, and it can
be done to a large part in and efficient and fast way.

---

Outline
=======

 * Native Python memory and speed
 * Numpy introduction (matrix manipulation)
 * Ipython and Matplotlib (visualization)

# Presenter Notes

We'll start of looking at how native python works memory and speed wise.
As you might already know most Python interpreters aren't very fast, so this
will affect the way you would implement algorithms and manage data processing
for scientific computing.

Then we'll look at a library for presenting matrices and matrix operations. This
is probably one of the most commonly used libraries in scientific python.

Finally we'll look at Matploblib, which is a nice 2d plotting library for Python.

Python is a high-level general-purpose interpreted programming language [http://en.wikipedia.org/wiki/Python_(programming_language)].

Python and speed (or lack thereof)
==================================

A very brute mean function:

    !python
    def calc_mean(list_matrix):
        """Returns the mean of a matrix given as a list of lists.
        """
        total = 0
        cnt = 0
        for lst in list_matrix:
            for number in lst:
                total += number
                cnt += 1
        return (1.*total) / cnt

# Presenter Notes

See more at http://scikit-learn.org/dev/developers/performance.html

---

Native Python
=============

---

Python and memory usage
=======================

Install memory profiler: `pip install -U memory_profiler`

Let's make a 1000x1000 demo matrix:

    !python
    @profile
    def make_matrix():
        matrix_2d = []
        for i in range(1000):
            repeat_me = range(1000)
            matrix_2d.append(repeat_me)
        return matrix_2d

    if __name__ == '__main__':
        make_matrix()

# Presenter Notes

If you want to run this without memory profiling, you can create a dummy
decorator:

def profile(fn):
    def wrapped(): return fn()
    return wrapped
---

Python and memory usage
=======================

Run the program using a memory profiler:

    >>> python -m memory_profiler test.py
    
    Line #    Mem usage    Increment   Line Contents
    ================================================
         2                             @profile
         3      6.03 MB      0.00 MB   def make_matrix():
         4      6.04 MB      0.00 MB       matrix_2d = []
         5     18.86 MB     12.83 MB       for i in xrange(1000):
         6     18.88 MB      0.01 MB           repeat_me = range(1000)
         7     18.88 MB      0.00 MB           matrix_2d.append(repeat_me)
         8     18.88 MB      0.00 MB       return matrix_2d

---

Python and memory usage
=======================

Use `sys.getsizeof(object)` to find the actual memory used by an object:

    >>> import sys
    >>> a = [1]
    >>> sys.getsizeof(a)
    40
    >>> sys.getsizeof(a[0])
    12
    >>> sys.getsizeof(range(1))
    40
    >>> sys.getsizeof(range(2))
    44
    >>> sys.getsizeof(range(3))
    48
    >>> sys.getsizeof(range(1000))
    4036

# Presenter Notes
`sys.getsizeof()` returns the size of memory footprint in bytes of objects.
In some python versions container objects it will return the size of only the container object.
The list contains references to the objects, and it can be hard to figure out the
memory consumption, because the interpreter tries to reuse references to objects.

Non-native Python code, such as C code, might also allocate memory without Python's knowledge.

---

Python and speed (or lack thereof)
==================================

    In [43]: %prun calc_mean(m)
             3 function calls in 0.159 seconds

       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    0.159    0.159    0.159    0.159 calc_mean.py:2(calc_mean)
            1    0.000    0.000    0.159    0.159 <string>:1(<module>)
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

---
IPython line profiler
=====================

`%prun` only times function calls.

line_profiler can report per line.

    %lprun -f calc_mean calc_mean(m)
    Timer unit: 1e-06 s

    File: calc_mean.py
    Function: calc_mean at line 2
    Total time: 4.27212 s

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
         2                                           def calc_mean(list_matrix):
         3                                               """Returns the mean of a matrix given as a list of lists.
         4                                               """
         5         1           11     11.0      0.0      total = 0
         6         1            1      1.0      0.0      cnt = 0
         7      1001         1497      1.5      0.0      for lst in list_matrix:
         8   1001000      1352026      1.4     31.6          for number in lst:
         9   1000000      1412441      1.4     33.1              total += number
        10   1000000      1506138      1.5     35.3              cnt += 1
        11         1            4      4.0      0.0      return (1.*total) / cnt


# Presenter Notes

http://scikit-learn.org/dev/developers/performance.html

line_profiler doc: http://packages.python.org/line_profiler/

---

Numpy introduction
==================

---

Libraries
=========

Some libaries that can make life easier:

* NumPy (matrix reprensentation, linear algebra, C/C++ code integration)
* SciPy (optimization, more linear algebra, special functions, FFT, ODE, ...)
* Scikit-learn (machine learning, supervised/unsupervised, model selection, ...)

Visualization:

* Matplotlib (nice 2d plots, some support for 3d plots, maps)
* Mayavi (3d plotting)
<!-- * ...
.comment * igraph (network plots) -->

Wrap:

* Fortran, C, C++, ...
* R

And then access to all the other python libraries: os, sys, wxPython, Django, flask, ...

---

NumPy
=====

Why should we use NumPy?

* NumPy arrays are more compact than python lists
* Large library of implemeted functions
* Computationally faster than using native python lists

Matlab users might find this useful: [http://www.scipy.org/NumPy_for_Matlab_Users](http://www.scipy.org/NumPy_for_Matlab_Users)

---

NumPy array
===========

An array in NumPy can be created this way:

    !python
    import numpy as np
    X = np.array([[1,2], [3,4]])

So there are kind of many brackets here. In Matlab it would just have been:

    >> X = [1,2; 3,4]

The NumPy matrix class helps a bit:

    >>> import numpy as np
    >>> X = np.matrix('1 2; 3 4')

If we have `from numpy import *` then we might write it as:

    >>> from numpy import *
    >>> X = matrix('1,2; 3,4')

# Presenter Notes

But normally I don't use the matrix class. One of the reasons being that I just
started with the array class, and this is also what most of the Numpy methods
will return.

It seems like Numpy is a kind of second grade citizen in Python. The notation
isn't really nice for it. But then again; Python is a general purpose language.

---

Numpy - references
==================

There might be some nasty surprices using NumPy arrays:

    >>> import numpy as np
    >>> X = np.array([[1,2],[3,4]])
    >>> Y = X
    >>> print X
    [[1 2]
     [3 4]]
    >>> print Y
    [[1 2]
     [3 4]]
    >>> X[0,0] = 0
    >>> print X
    [[0 2]
     [3 4]]
    >>> print Y
    [[0 2]
     [3 4]]    

Simple trick to avoid this: `Y = 1*X` (makes a copy)

---


Numpy - slicing arrays
======================

    >>> import numpy as np
    >>> X = np.array([[1,2],[3,4]])
    >>> print X
    [[1 2]
     [3 4]]
    >>> print X[0]
    [1 2]
    >>> print X[0,:]
    [1 2]
    >>> print X[:,0]
    [1 3]
---

Numpy - slicing arrays
======================

    >>> X = np.array(range(9)).reshape(3,3)
    >>> print X
    [[0, 1, 2],
     [3, 4, 5],
     [6, 7, 8]])
    >>> print X[1:, 1:]
    [[4 5]
     [7 8]]
    >>> print X[:1, :1]
    [[0]]

---

NumPy - slicing arrays
======================

There might be some nasty surprices using NumPy arrays:

    >>> import numpy as np
    >>> X = np.array([[1,2],[3,4]])
    >>> X.shape
    (2, 2)
    >>> X1 = X[:,0]
    >>> X1.shape
    (2,)

So it has become a one dimensional array.

    >>> Y = np.matrix('1,2; 3,4')
    >>> Y[:,0]
    matrix([[1],
            [3]])
    >>> Y1 = Y[:,0]
    >>> Y1.shape
    (2, 1)

---

NumPy - more slicing surprices
==============================

A slices is a reference to part of the original array.

    >>> a = eye(3)
    >>> print a
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    >>> b = a[:,1]
    >>> print b
    [ 0.  1.  0.]
    >>> b[1]=2
    >>> print a
    [[ 1.  0.  0.]
     [ 0.  2.  0.]
     [ 0.  0.  1.]]

---

Numpy - striding arrays
=======================

    >>> X = np.array(range(25)).reshape(5,5)
    >>> print X
    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]]
    >>> print X[:,::2]
    [[ 0  2  4]
     [ 5  7  9]
     [10 12 14]
     [15 17 19]
     [20 22 24]]
    >>> print X[:,1::2]
    [[ 1  3]
     [ 6  8]
     [11 13]
     [16 18]
     [21 23]]


# Presenter Notes

More info here: http://scipy-lectures.github.com/advanced/advanced_numpy/index.html
A video here: https://www.youtube.com/watch?v=7vcjjN9eNvs

---


Numpy - working with indices
============================

    >>> X = np.array(range(25)).reshape(5,5)
    >>> print X
    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]]
    >>> B = X % 3 == 0
    >>> print B
    [[ True False False  True False]
     [False  True False False  True]
     [False False  True False False]
     [ True False False  True False]
     [False  True False False  True]]
    >>> X[B] = 0
    >>> print X
    [[ 0  1  2  0  4]
     [ 5  0  7  8  0]
     [10 11  0 13 14]
     [ 0 16 17  0 19]
     [20  0 22 23  0]]


---

Ipython and Matplotlib
======================

---
IPython
=======

IPython is an enhanced Python shell.

* Code completion and highlighting
* Profiling code
* Command history

Install: Enthought, Sage, (Pythonxy)


---

IPython magic functions
=======================

* `%hist`: Show command history
* `%edit`: Open editor and execute code after closing
* `%prun`: Profile a method call

* _variable_`?`: Show type and docstring
* _variable_`??`: Show the code


---

IPython help
============

    >>> import numpy as np
    >>> np.mean?
    Base Class: <type 'function'>
    String Form:<function mean at 0x12f6570>
    Namespace:  Interactive
    File:       /Library/Frameworks/Python.framework/Versions/7.2/lib/python2.7/site-packages/numpy/core/fromnumeric.py
    Definition: np.mean(a, axis=None, dtype=None, out=None)
    Docstring:
    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : int, optional
        Axis along which the means are computed. The default is to compute
        the mean of the flattened array.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
    ...

---

IPython profiling
=================


    %prun
---

IPython speed optimization
==========================

Generally:

 * Only optimize code that needs to be.
 * Has someone already optimized this problem?
 * Is there a better way of doing it? Another algorithm?

Try to avoid python loops, especially nested loops, and write the code in such a
way that it uses numpy, scipy, and similar functions instead.

Cython?

---

Matplotlib
==========

Matplotlib's gallery is a good place to get inspiration:

[http://matplotlib.org/gallery.html](http://matplotlib.org/gallery.html)

IPython in pylab mode
---------------------

Special support for Matplotlib.pylab plots using `-pylab` (`--pylab` since IPython version 0.12).

Multi-threaded handling of the figures in the background.

---

Scikit learn
============


---

Application examples
====================

---


Summary
=======

With python you have

* access to a large library of well tested methods for linear algebra, optimization, machine learning, and so on.

---


Resources
=========

* http://www.scipy.org/Tentative_NumPy_Tutorial

