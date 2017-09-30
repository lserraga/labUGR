import pkgutil
from importlib import import_module
import labugr
from labugr.audio.__init__ import test_portaudio, test_backend
from sys import platform
__all__=['test_all']

def test_all():
	"""
	Testing all modules.

	Returns
	-------
	passed: int
		0 if there are major errors, 1 if everything is ok and 2 if there are
		non-critical errors.
	"""
	#Submódulos sin tests
	no_tests = ['dependencies', 'doc', 'testing']
	passed = True
	print("\n******** Iniciando tests para labugr ********\n")

	#Iterar a través de ls submódulos de labugr
	for importer, submodule, ispkg in pkgutil.iter_modules(labugr.__path__):
		#Determinamos si el submódulo es correcto
		if (ispkg and not submodule in no_tests):

			print("Corriendo tests para el submodulódulo {}".format(submodule))
			#Lo importamos
			module = import_module ("labugr.{}".format(submodule))

			#Corremos los tests
			if(module.test(extra_argv=["-qqq"])==True):
				print("\tOK")
			else:
				print("\tERROR")
				passed = False

	if not passed:
		print("""\nAlgunos de los submódulos no se han instalado correctamente.
	Por favor, desinstale labugr, numpy y matplotlib y resinstale labugr
	(pip install labugr)""")

	elif not test_portaudio():
		warning_msg = """\nLibreria instalada correctamente.\
	\n\n****Para poder reproducir archivos de audio por favor instale Portaudio****\
	\n****Portaudio es una libreria multiplataforma para audio I/O ****\n\n"""

		if platform == "linux" or platform == "linux2":
			warning_msg += """Para instalar portaudio en Linux utilice este comando:\
		\n\tsudo apt-get install portaudio19-dev"""

		elif platform == "darwin":
			warning_msg += """Para instalar portaudio en macOS utilice este comando:\
		\nEs necesario tener instalado Homebrew\
		\n\tbrew install portaudio"""

		# Para windows es bastante complicado instalar portaudio por ello, los binarios
		# estan incluidos en el paquete pyaudio.
		print(warning_msg)

	elif not test_backend():
		warning_msg = """\nLibreria instalada correctamente.\
	\n\n****No se ha encontrado ningun backend de audio en el sistema. Es necesario para\
	\n****utilizar archivos de audio comprimidos (mp3, mp4,...)****\n\n"""
		if platform == "linux" or platform == "linux2":
			warning_msg += """Se recomienda instalar ffmpeg en Linux\
			\n\tsudo apt-get install ffmpeg"""

		elif platform == "darwin":
			warning_msg += """CoreAudio no funciona correcamente en su MAC"""

		else:
			warning_msg += """Se recomienda instalar ffmpeg en windows\
			\n\tDescargar el archivo correspondiente a tu sistema (https://ffmpeg.zeranoe.com/builds/)\
			\n\tAnadir la carpeta ffmpeg-20170921-183fd30-win64-static\bin al PATH de Windows """
		# Para mac OS, CORE audio tendria que estar instalado por defecto
		print(warning_msg)
	else:
		print("\nTodos los tests han resultado satisfactorios")