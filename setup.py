from setuptools import setup
import sys, os
if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version 2.7 or >= 3.4 required.")



def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('labugr')

    return config



def setup_package():

    # Figure out whether to add ``*_requires = ['numpy']``.
    # We don't want to do that unconditionally, because we risk updating
    # an installed numpy which fails too often.  Just if it's not installed, we
    # may give it a try.  See gh-3379.
    try:
        import numpy, matplotlib
    except ImportError:  # We do not have numpy installed
        build_requires = ['numpy>=1.8.2','matplotlib>=2.0.2']
    else:
        # If we're building a wheel, assume there already exist numpy wheels
        # for this platform, so it is safe to add numpy to build requirements.
        # See gh-5184.
        build_requires = (['numpy>=1.8.2','matplotlib>=2.0.2'] if 'bdist_wheel' in sys.argv[1:]
                          else [])

    packages = ['labugr.testing']
    
    metadata = dict(
        name='labugr',
        version='0.1.0.3',
        author='Luis Serra Garcia',
        author_email='lsgarcia@correo.ugr.es',
        url='http://github.com/lserraga/labugr',
        package_data={'':['README.md']},
        scripts=['scripts/remove_build.sh'],
        description='Laboratorio de señales UGR.',
        long_description="open('README.md').read()",
        setup_requires=build_requires,
        install_requires=build_requires,
        packages=packages,
        include_package_data=True,
        python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
    )

    from numpy.distutils.core import setup
    metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
