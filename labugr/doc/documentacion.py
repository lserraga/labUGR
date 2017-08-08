"""
Recoge la documentación en inglés de las funciones dentro de labugr
para más adelante ser traducidas
"""

from labugr import *

#Guardando el nombre de todas las funciones
funciones = dir()
#Eliminando los parámetros del script de la lista
funciones = list(filter(lambda a: not a.startswith('__'), funciones))

import os

#Guardando el directorio del script podemos asegurarnos que la
#documentación se guarda siempre en la misma carpeta
directorio = os.path.dirname(__file__)

aux = 0
for funcion in funciones:
	#Cada documentación se guarda como "nombre_función".txt
	nombre_doc = '{}/funciones/{}.txt'.format(directorio, funcion)

	#Si el archivo ya existe, no sobreescribirlo
	if not os.path.isfile(nombre_doc):

		with open(nombre_doc, 'w') as f:
			#En __doc__ encontramos la documentació en inglés
			try:
				texto_ingles = globals()[funcion].__doc__
				f.write(texto_ingles)
				aux += 1
			except:
				print("Error cargando la documentación de {}"
					.format(funcion))

print("La documentación de {} funciones ha sido actualizada".format(aux))
