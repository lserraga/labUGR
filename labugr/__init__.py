"""
Recopilación de funciones para el análisis de señales en python
"""

from .scipy.waveforms import *
from .scipy.respuestaF import *
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

__all__ = [s for s in dir() if not (s=='doc' or s=='waveforms' or s=='scipy' or s=='dependencias' or s.startswith('_'))]
