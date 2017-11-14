"""
Recopilación de funciones para el análisis de señales en python
"""



# Primero detectamos si estamos en el proceso de instalación de la 
# librería. La herramienta que utilizamos para generar el paquete
# crea la variable __LABUGR_SETUP__
try:
    __LABUGR_SETUP__
except NameError:
    __LABUGR_SETUP__ = False

# Si no estamos en el setup hay que comprobar que no se intenta 
# importar desde el directorio fuente
if not __LABUGR_SETUP__:
    try:
        from labugr.__config__ import show as show_config
    except ImportError:
        msg = """Error importando labugr: labugr no se puede importar
        mientras estés en el directorio fuente. Por favor, ejecute 
        Python desde otro directorio."""
        raise ImportError(msg)


import numpy as np
import matplotlib.pyplot as plt

from numpy import pi, e, inf

#Creacion de arrays
from numpy import (
	array, zeros, ones, eye, full, asarray, copy, fromfunction,
	arange, linspace, logspace, geomspace, tri)

#Propiedades del array
from numpy import (
	shape, ndim, size, dtype, real, imag, angle, conj, diag, tril ,triu)

#Funciones trigonométricas (son igual que las de math pero se pueden utilizar en arrays)
from numpy import (
	sin, cos, tan, arcsin, arccos, arctan, hypot, sinh, cosh,
	tanh, arcsinh, arccosh, sinc, arctanh, deg2rad, rad2deg)

#Manipulación de arrays
from numpy import (
    reshape, transpose, concatenate, stack, delete, insert,
    unique, roll, sort, where)
from numpy import array_split as split

#Funciones matemáticas (abs() y pow() estan incluido)
from numpy import(
    floor, ceil, exp, log, log10, log2, sqrt, sum, gradient, cross, maximum,
    minimum, abs, dot)
from numpy import around as round 

#Funciones para graficar
from matplotlib.pyplot import (
    axis, title, grid, xlabel, ylabel, xscale, yscale, xlim, ylim, xticks,
    yticks, legend)
from matplotlib.pyplot import (
    plot, subplot, figure, bar, barh, stem, step, subplot, show, close,
    ion, ioff, semilogy, semilogx, axvline, hlines, vlines)
    
#Algebra lineal
from numpy.linalg import (
    matrix_power, det, solve, inv)

#Random
from numpy.random import (
    rand, randn, randint)

#Iteracion
from numpy import nditer

#Funciones lógicas. All y any substituyen a las built_in para 
#que sean compatibles con numpy arrays (use allclose to compare allrays)
from numpy import (
    all, any, logical_and, logical_or, allclose, isclose, array_equal)

from .fftpack import (
    fft, ifft, fftn, ifftn, dct, idct, dst, idst, diff, hilbert,
    ihilbert, fftshift, ifftshift, fftfreq)


from .signal import *
from .doc.ayuda import ayuda
from .integrate import *
from .testing import test_all
from .systems import *
from .filters import *
from .audio import *


excluded = ['signal', 'testing', 'doc','dependencies', 'systems',
'signal', 'fftpack', 'audio', 'filters', 'integrate' ,'excluded',
'test', 'show_config']

__all__ = [s for s in dir() if not ((s in excluded)or s.startswith('_'))]

