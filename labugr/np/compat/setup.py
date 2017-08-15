#!/usr/bin/env python
from __future__ import division, print_function


def configuration(parent_package='',top_path=None):
    from labugr.np.distutils.misc_util import Configuration
    config = Configuration('compat', parent_package, top_path)
    return config

if __name__ == '__main__':
    from labugr.np.distutils.core import setup
    setup(configuration=configuration)
