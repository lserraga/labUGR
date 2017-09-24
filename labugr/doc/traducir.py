import os
from os.path import join
import sys
from googletrans import Translator

translator = Translator()

#Cambiando directorio de trabajo al directorio del script
os.chdir(sys.path[0])
aux = 0
files = os.listdir('funciones')

for file in files:

	if file.endswith("-es.txt"):
		continue

	with open(join('funciones',file), 'r') as f:
			try:
				file_es = "{}-es.txt".format(file.split('.')[0])

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

	os.remove(join('funciones',file))

print ("Traducidas la documentación de {} funciones".format(aux))