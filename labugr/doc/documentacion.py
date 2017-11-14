"""
Recoge la documentación en inglés de las funciones dentro de labugr
para más adelante ser traducidas
"""

from labugr import *

#Guardando el nombre de todas las funciones
funciones = dir()

#Funciones dentro de windows.
from labugr import windows
ventanas = ['windows.' + ventana for ventana in windows.__all__]
funciones += ventanas

#Eliminando los parámetros del script de la lista
funciones = list(filter(lambda a: not a.startswith('__'), funciones))

import os
import sys
from os.path import isfile 
from labugr.signal.windows import *

#Cambiando directorio de trabajo al directorio del script
os.chdir(sys.path[0])

#Crear el directorio funciones si este no existe
if not os.path.exists("doc-ESP"):
	os.makedirs("doc-ESP")

aux = 0
for funcion in funciones:
	#Cada documentación se guarda como "nombre_función".txt
	nombre_doc = os.path.join('doc-ESP','{}.txt'.format(funcion))
	nombre_doc_trad = os.path.join('doc-ESP','{}-es.txt'.format(funcion))

	#Si el archivo ya existe o su traduccion, no sobreescribirlo
	if not isfile(nombre_doc) and not isfile(nombre_doc_trad):

		with open(nombre_doc, 'w') as f:
			#En __doc__ encontramos la documentació en inglés
			try:
				spl = funcion.split('.')
				if len(spl) == 2:
					texto_ingles = globals()[spl[1]].__doc__
				else:
					texto_ingles = globals()[funcion].__doc__
				f.write(texto_ingles)
				aux += 1
			except:
				print("Error cargando la documentación de {}"
					.format(funcion))

print("La documentación de {} funciones ha sido actualizada".format(aux))

