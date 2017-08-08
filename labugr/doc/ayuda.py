import os
import difflib

def __nombre_funciones():
	lista = os.listdir()

	for i in range(0, len(lista)):
		lista[i] = os.path.splitext(lista[i])[0]

	return lista


def ayuda(funcion):
	directorio_previo = os.getcwd()
	os.chdir(os.path.join(os.path.dirname(__file__),'funciones'))

	lista = __nombre_funciones()
	
	if not funcion in lista:
		parecida = difflib.get_close_matches(funcion, lista)
		print("""
		ERROR: {} no se reconoce como función
		La función más parecida es {}""".format(funcion, parecida))
	else:
		archivo = "{}.txt".format(funcion)
		with open(archivo) as f:
			print (f.read())
	os.chdir(directorio_previo)
	return