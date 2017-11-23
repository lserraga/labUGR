from labugr import *

def dft(senal):
	"""
	Funcion para calcular la transformada discreta de una secuencia de valores
	"""
	N = len(senal) # numero de muestras
	resultado = zeros(N, dtype=np.complex_) # Inicializando el resultado como 
	# un vector de numeros complejos (sin dtype la parte imaginaria se perderia)

	# Ecuacion principal
	for k in range(0, N):
		for n in range(0, N):
			resultado[k] += senal[n]*exp(-2j*pi*k*n/N)

	# Estableciendo la precision del resultado (8 decimales)
	return round(resultado, 8)


def dft2(senal):
	"""
	Funcion para calcular la transformada discreta de una secuencia de valores a
	traves del calculo matricial
	"""
	N = len(senal) # numero de muestras
	k = arange(N) # vector de k
	n = arange(N) # vector de n

	expon = exp(-2j*pi * np.outer(n, k) / N) # parte exponencial
	# outer es equivalente a los dos bucles for

	# producto vectorial de la senal de entrada y la exponencial
	resultado = dot(senal, expon)

	# Estableciendo la precision del resultado (8 decimales)
	return round(resultado, 8)


