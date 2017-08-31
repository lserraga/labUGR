from __future__ import division, print_function, absolute_import

numpy_nodepr_api = dict(define_macros=[("NPY_NO_DEPRECATED_API",
                                            "NPY_1_9_API_VERSION")])

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('signal', parent_package, top_path)

    config.add_data_dir('tests') # Como tests no es considerado como un
    							 # package, hay que anadirlo como 
    							 # directorio de datos
    config.add_subpackage('tools')

    config.add_extension('sigtools',
                         sources=['src/sigtoolsmodule.c', 'src/firfilter.c',
                                  'src/medianfilter.c', 'src/lfilter.c.src',
                                  'src/correlate_nd.c.src'],
                         depends=['src/sigtools.h'],
                         include_dirs=['.'],
                         **numpy_nodepr_api)
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
