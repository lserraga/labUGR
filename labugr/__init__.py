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


from .scipy.waveforms import *
from .scipy.respuestaF import *
from .scipy.windows import *
from .scipy.spectral import *
from .doc.ayuda import ayuda

import numpy as np
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

#Funciones trigonométricas
from numpy import (
	sin, cos, tan, arcsin, arccos, arctan, hypot, sinh, cosh,
	tanh, arcsinh, arccosh, arctanh, deg2rad, rad2deg)

#Algebra lineal
#from numpy.linalg import 

from . import fftpack

__all__ = [s for s in dir() if not (s=='doc' or s=='waveforms' or s=='dependencias' or s.startswith('_'))]
