from __future__ import division, print_function, absolute_import

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('systems', parent_package, top_path)

    fitpack_src = [join('fitpack', '*.f')]
    config.add_library('fitpack', sources=fitpack_src)
    config.add_extension('dfitpack',
                     sources=['src/fitpack.pyf'],
                     libraries=['fitpack'],
                     depends=fitpack_src,
                     )

    config.add_data_dir('tests') 
    config.add_subpackage('filters')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
