import sys
import os
import subprocess
from platform import architecture
import builtins

#Comprobamos que la version de python es la correcta
if sys.version_info[:2] < (3, 4):
    raise RuntimeError("Requerido python 3.4, 3.5 o 3.6")


#Para poder determinar si estamos instalando labUGR
builtins.__LABUGR_SETUP__ = True

#Directorio de trabajo
directorio = os.path.abspath(os.path.dirname(__file__))

#Descripcion completa del paquete
with open(os.path.join(directorio, 'README.md')) as f:
    long_description = f.read()


#Funcion para determinar si python es 32 o 64 bits
def get_bitness():
    bits, _ = architecture()
    return '32' if bits == '32bit' else '64' if bits == '64bit' else None

#Directorio atlas necesario para compilar desde el código fuente en windows
atlas_compil = """[atlas]
include_dirs = {dir}\\atlas-builds\\{version}\\include
library_dirs = {dir}\\atlas-builds\\{version}\\lib
atlas_libs = numpy-atlas
lapack_libs = numpy-atlas
"""

#Con site.cfg podemos especificar compiladores
with open(os.path.join(directorio,'site.cfg'),'w') as f:
    #Solo queremos utilizar site.cfg cuando estamos en windows
    if os.name == 'nt':
        if (get_bitness()=='32'):
            atlas_compil=atlas_compil.format(version="atlas-3.10.1-sse2-32",
                                             dir=directorio)
        else:
            atlas_compil=atlas_compil.format(version="atlas-3.11.38-sse2-64",
                                             dir=directorio)
        f.write(atlas_compil)
    #Si no es windows, cuando with cierra el archivo, este se queda en blanco

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('labugr')

    return config

def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd, 'scripts', 'cythonize.py'),
                         'labugr'],
                        cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def setup_package():

    # Modulos externos requeridos
    install_requires = ['numpy>=1.8.2','mpmath', 'matplotlib']

    # En MAC y linux pyaudio produce errores que paran la instalacion de 
    # labugr si el sistema no tiene instalado portaudio
    if os.name == 'nt':
        install_requires.append('pyaudio')

    
    metadata = dict(
        name='labugr',
        version='1.1',
        author='Luis Serra Garcia',
        author_email='lsgarcia@correo.ugr.es',
        url='http://github.com/lserraga/labUGR',
        package_data={'':['README.md']},
        description="Laboratorio de señales UGR",
        license='MIT',
        keywords='signal analysis',
        long_description=long_description,
        install_requires=install_requires,
        #packages=packages,
        platforms = ["Windows", "Linux", "MacOS"],
        # Dependencias extra. Cython para la instalación desde código 
        # fuente y pytest-nose para testing 
        # $ pip install .[test]
        # $ pip install labugr[test]
        extras_require={
            'dev': ['cython'],
            'test': ['pytest', 'nose'],
        },
        python_requires='>=3.4.*',
    )

    from numpy.distutils.core import setup
    cwd = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
        generate_cython()
    metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
