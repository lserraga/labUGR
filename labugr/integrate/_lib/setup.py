from __future__ import division, print_function, absolute_import

import os


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('_lib', parent_package, top_path)

    include_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
    depends = [os.path.join(include_dir, 'ccallback.h')]

    config.add_extension("_ccallback_c",
                         sources=["_ccallback_c.c"],
                         depends=depends,
                         include_dirs=[include_dir])
    
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
