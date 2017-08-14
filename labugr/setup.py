from __future__ import division, print_function, absolute_import

import sys


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('labugr',parent_package,top_path)
    config.add_subpackage('testing')
    config.add_subpackage('fftpack')
    config.add_subpackage('signal')
    config.add_subpackage('signal.tools')
    config.add_subpackage('doc') 
    config.add_subpackage('dependencias')
    config.add_data_dir('doc')#Importamos la documentacion de las funciones para poder encontrarlas con ayuda()

    config.make_config_py() #con esto generamos __config__ que nos sirve para comprobar que no se importa desde el source dir
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
