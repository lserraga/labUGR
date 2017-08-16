#!/usr/bin/env python
from __future__ import division, print_function

def configuration(parent_package='',top_path=None):
    from labugr.distutils.misc_util import Configuration
    config = Configuration('distutils', parent_package, top_path)
    config.add_subpackage('command')
    config.add_subpackage('fcompiler')
    config.add_data_files('mingw/gfortran_vs2003_hack.c')
    config.add_data_files('site.cfg')
    config.make_config_py()
    return config

if __name__ == '__main__':
    from labugr.distutils.core import setup
    setup(configuration=configuration)
