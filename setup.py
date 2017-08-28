import sys
import os
import subprocess
if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version 2.7 or >= 3.4 required.")

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

#Para poder determinar si estamos instalando labUGR
builtins.__LABUGR_SETUP__ = True

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.txt')) as f:
    long_description = f.read()


atlas_compil = """[atlas]
include_dirs = c:\\Users\\LuikS\\Desktop\\labugr\\atlas-builds\\atlas-3.10.1-sse2-32\\include
library_dirs = c:\\Users\\LuikS\\Desktop\\labugr\\atlas-builds\\atlas-3.10.1-sse2-32\\lib
atlas_libs = numpy-atlas
lapack_libs = numpy-atlas
"""
# Solo queremos utilizar site.cfg cuando estamos en windows
with open(os.path.join(here,'site.cfg'),'w') as f:
    if os.name == 'nt':
        f.write(atlas_compil)



def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('labugr')

    return config

def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd, 'cythonize.py'),
                         'labugr'],
                        cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


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
        version='0.1.2',
        author='Luis Serra Garcia',
        author_email='lsgarcia@correo.ugr.es',
        url='http://github.com/lserraga/labUGR',
        package_data={'':['README.txt']},
        scripts=['scripts/remove_build.sh'],
        description="Laboratorio de señales UGR",
        license='',
        keywords='signal analysis',
        long_description=long_description,
        install_requires=build_requires,
        packages=packages,
        platforms = ["Windows", "Linux"],
        python_requires='!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
    )

    from numpy.distutils.core import setup
    cwd = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
        # Generate Cython sources, unless building from source release
        generate_cython()
    metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
