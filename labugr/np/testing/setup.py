#!/usr/bin/env python
from __future__ import division, print_function


def configuration(parent_package='',top_path=None):
    from labugr.np.distutils.misc_util import Configuration
    config = Configuration('testing', parent_package, top_path)

    config.add_subpackage('nose_tools')
    config.add_data_dir('tests')
    return config

if __name__ == '__main__':
    from labugr.np.distutils.core import setup
    setup(maintainer="NumPy Developers",
          maintainer_email="numpy-dev@labugr.np.org",
          description="NumPy test module",
          url="http://www.labugr.np.org",
          license="NumPy License (BSD Style)",
          configuration=configuration,
          )
