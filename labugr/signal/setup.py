from __future__ import division, print_function, absolute_import

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('signal', parent_package, top_path)

    config.add_data_dir('tests') # Como tests no es considerado como un
    							 # package, hay que añadirlo como 
    							 # directorio de datos
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
