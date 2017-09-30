"""
Utilizando la libreria googletrans, que utiliza la API de google translator,
este script traduce la documentacion de las funciones a espanol
"""
import os
from os.path import join
import sys
from googletrans import Translator

#Inicializando el traductor
translator = Translator()

#Cambiando directorio de trabajo al directorio del script
os.chdir(sys.path[0])
aux = 0
#Cargando la lista de archivos
files = os.listdir('funciones')

for file in files:

	if file.endswith("-es.txt"):
		continue

	with open(join('funciones',file), 'r') as f:
			try:
				spl = file.split('.')

				#Para las funciones de windows
				if len(spl) == 3:
					file_es = "{}.{}-es.txt".format(spl[0], spl[1])
				else:
					file_es = "{}-es.txt".format(spl[0])

				#Enviando mas de 4500 Bytes produce errores al traducir
				data = f.read(4500)
				with open(join('funciones', file_es), 'a') as f2:
					while len(data)>0:
						traduction = translator.translate(data, src='en', 
														 dest='es').text
						f2.write(traduction)
						data = f.read(4500)
				aux += 1
			except Exception as ex:
				print("Error traduciendo la documentación de {}"
					.format(file))
				print(ex)
				continue

	#Eliminando el archivo en ingles
	os.remove(join('funciones',file))

print ("Traducidas la documentación de {} funciones".format(aux))