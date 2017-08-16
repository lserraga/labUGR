import sys
from os import path
if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version 2.7 or >= 3.4 required.")

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.txt')) as f:
    long_description = f.read()


# # Always prefer setuptools over distutils
# from setuptools import setup, find_packages
# from os import path

# here = path.abspath(path.dirname(__file__))

# # Get the long description from the README file
# with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
#     long_description = f.read()

# setup(
#     name='labugr',

#     # Versions should comply with PEP440.  For a discussion on single-sourcing
#     # the version across setup.py and the project code, see
#     # https://packaging.python.org/en/latest/single_source_version.html
#     version='0.1.0.8',

#     description='Laboratorio de señales UGR.',
#     long_description=long_description,

#     # The project's main homepage.
#     url='https://github.com/lserraga/labUGR',

#     # Author details
#     author='Luis Serra Garcia',
#     author_email='lsgarcia@correo.ugr.es',

#     # Choose your license
#     license='',

#     # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
#     classifiers=[
#         # How mature is this project? Common values are
#         #   3 - Alpha
#         #   4 - Beta
#         #   5 - Production/Stable
#         'Development Status :: 3 - Alpha',

#         # Indicate who your project is intended for
#         'Intended Audience :: Developers',
#         'Topic :: Software Development :: Build Tools',

#         # Pick your license as you wish (should match "license" above)
#         'License :: OSI Approved :: MIT License',

#         # Specify the Python versions you support here. In particular, ensure
#         # that you indicate whether you support Python 2, Python 3 or both.
#         'Programming Language :: Python :: 2',
#         'Programming Language :: Python :: 2.7',
#         'Programming Language :: Python :: 3',
#         'Programming Language :: Python :: 3.3',
#         'Programming Language :: Python :: 3.4',
#         'Programming Language :: Python :: 3.5',
#     ],

#     # What does your project relate to?
#     keywords='sample setuptools development',

#     # You can just specify the packages manually here if your project is
#     # simple. Or you can use find_packages().
#     packages=find_packages(exclude=['src', 'tools', 'tests','funciones']),

#     # Alternatively, if you want to distribute just a my_module.py, uncomment
#     # this:
#     #   py_modules=["my_module"],

#     # List run-time dependencies here.  These will be installed by pip when
#     # your project is installed. For an analysis of "install_requires" vs pip's
#     # requirements files see:
#     # https://packaging.python.org/en/latest/requirements.html
#     install_requires=['numpy>=1.8.2'],

#     # List additional groups of dependencies here (e.g. development
#     # dependencies). You can install these using the following syntax,
#     # for example:
#     # $ pip install -e .[dev,test]
#     extras_require={
#         'dev': ['check-manifest'],
#         'test': ['coverage'],
#     },

#     # If there are data files included in your packages that need to be
#     # installed, specify them here.  If using Python 2.6 or less, then these
#     # have to be included in MANIFEST.in as well.
#     package_data={
#         '':['README.rst'],
#     },
#     scripts=['scripts/remove_build.sh'],

#     include_package_data=True,
#     ignore_setup_xxx_py=True,
#     assume_default_configuration=True,
#     delegate_options_to_subpackages=True,
#     quiet=True,
#     python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*'
# )

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
        import numpy
    except ImportError:  # We do not have numpy installed
        build_requires = ['numpy>=1.8.2']
    else:
        # If we're building a wheel, assume there already exist numpy wheels
        # for this platform, so it is safe to add numpy to build requirements.
        # See gh-5184.
        build_requires = (['numpy>=1.8.2'] if 'bdist_wheel' in sys.argv[1:]
                          else [])

    packages = ['labugr.testing']
    
    metadata = dict(
        name='labugr',
        version='0.1.0.8',
        author='Luis Serra Garcia',
        author_email='lsgarcia@correo.ugr.es',
        url='http://github.com/lserraga/labUGR',
        package_data={'':['README.txt']},
        scripts=['scripts/remove_build.sh'],
        description="Laboratorio de señales UGR",
        long_description=long_description,
        setup_requires=build_requires,
        install_requires=build_requires,
        packages=packages,
        include_package_data=True,
        python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*'
    )

    from numpy.distutils.core import setup
    metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
