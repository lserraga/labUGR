import pkgutil
from importlib import import_module
import labugr

__all__=['test_all']

def test_all():
	#Submódulos sin tests
	no_tests = ['dependencias', 'doc', 'testing']

	passed = True

	#Iterar a través de ls submódulos de labugr
	for importer, submodule, ispkg in pkgutil.iter_modules(labugr.__path__):
		#Determinamos si el submódulo es correcto
		if (ispkg and not submodule in no_tests):

			print("\nCorriendo tests para el submodulódulo {}".format(submodule))
			#Lo importamos
			module = import_module ("labugr.{}".format(submodule))

			#Corremos los tests
			if(module.test(extra_argv=["-qqq"])==True):
				print("\tOK")
			else:
				print("\tERROR")
				passed = False

	if passed:
		print("\nTodos los tests han resultado satisfactorios")
	else:
		print("""\nAlgunos de los submódulos no se han instalado correctamente.
	Por favor, desinstale labugr, numpy y matplotlib y resinstale labugr
	(pip install labugr)""")