import os
import difflib
from labugr import windows

__all__ = ['ayuda']

def __nombre_funciones():
	"""
	Devuelve una lista de las funciones en el directorio de trabajo
	"""
	lista = os.listdir()

	for i in range(0, len(lista)):
		lista[i] = lista[i].split('-')[0]

	return lista


def ayuda(funcion):
	"""
	Muestra en pantalla la documentación en español de una función.

	Carga la documentación desde el directorio doc/funciones.

	Parámetros
    ----------
    funcion : string o función
    	El nombre de la función como string o como objeto en si 

    Ejemplos
    --------
    >>> ayuda(arange)
    >>> ayuda('arange')
	"""
	directorio_previo = os.getcwd()
	os.chdir(os.path.join(os.path.dirname(__file__),'funciones'))

	lista = __nombre_funciones()
	
	#Si la función es pasada como objeto, obtener su nombre como str
	if not isinstance(funcion,str):
		funcion = funcion.__name__
		if funcion in windows.__all__:
			funcion = "windows." + funcion
	
	if not funcion in lista:
		parecida = difflib.get_close_matches(funcion, lista)
		print("""
		ERROR: {} no se reconoce como función
		La función más parecida es {}""".format(funcion, parecida))
	else:
		archivo = "{}-es.txt".format(funcion)
		with open(archivo) as f:
			print (f.read())
	os.chdir(directorio_previo)
	return