from distutils.core import setup
import sys

if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version 2.7 or >= 3.4 required.")

setup(
    name='LabUGR',
    version='0.1.0',
    author='Luis Serra GArcia',
    author_email='lsgarcia@correo.ugr.es',
    packages=['labugr'],
    scripts=[],
    url='http://github.com/lserraga/labugr',
    license='LICENSE.txt',
    description='LAboratorio de señales UGR.',
    long_description=open('README.md').read()
)

