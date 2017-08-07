from .scipy.waveforms import *
from .scipy.respuestaF import *

import numpy as np
import matplotlib as plt

from numpy import arange, pi

#Creacion de arrays
from numpy import (
	array, zeros, ones, eye, full, asarray, copy, fromfunction,
	arrange, linspace, logspace, geomspace, tri)

#Guardando y cargando arrays
from numpy import (
	save, load, fromfile)

#Propiedades del array
from numpy import (
	shape, ndim, size, dtype, real, imag, diag, tril ,triu)

__all__ = [s for s in dir() if not (s=='waveforms' or s=='dependencias' or s.startswith('_'))]
