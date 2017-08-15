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

# # Si no estamos en el setup hay que comprobar que no se intenta 
# # importar desde el directorio fuente
# if not __LABUGR_SETUP__:
#     try:
#         from labugr.__config__ import show as show_config
#     except ImportError:
#         msg = """Error importando labugr: labugr no se puede importar
#         mientras estés en el directorio fuente. Por favor, ejecute 
#         Python desde otro directorio."""
#         raise ImportError(msg)


from .signal import *
from .doc.ayuda import ayuda

#import numpy as np
import matplotlib as plt

from numpy import arange, pi

#Creacion de arrays
from numpy import (
	array, zeros, ones, eye, full, asarray, copy, fromfunction,
	arange, linspace, logspace, geomspace, tri)

#Guardando y cargando arrays
from numpy import (
	save, load, fromfile)

#Propiedades del array
from numpy import (
	shape, ndim, size, dtype, real, imag, diag, tril ,triu)

#Funciones trigonométricas (son igual que las de math pero se pueden utilizar en arrays)
from numpy import (
	sin, cos, tan, arcsin, arccos, arctan, hypot, sinh, cosh,
	tanh, arcsinh, arccosh, arctanh, deg2rad, rad2deg)

#Manipulación de arrays
from numpy import (
    reshape, transpose, concatenate, stack, delete, insert,
    unique, roll, where, )
#En where, el primer array contiene el indice de la fila y el segundo el de las columnas
from numpy import array_split as split


#Algebra lineal
#from numpy.linalg import 

#See masked arrays, y allclose para eliminar e-12

#Funciones matemáticas (abs() y pow() estan incluido)
from math import (
    floor, ceil, gcd, isfinite, isinf, isnan, exp, e, log,
    sqrt, )

#sdjvnd
from numpy import nditer

from . import fftpack

from labugr.testing.utils import PytestTester
test = PytestTester(__name__)
del PytestTester

excluidos = ['respuestaF', 'signal', 'spectral', 'testing', 'windows',
            'doc','waveforms','dependencias']

__all__ = [s for s in dir() if not ((s in excluidos)or s.startswith('_'))]

