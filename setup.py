from setuptools import setup
import sys

if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version 2.7 or >= 3.4 required.")

setup(
    name='LabUGR',
    version='0.1.0',
    author='Luis Serra Garcia',
    author_email='lsgarcia@correo.ugr.es',
    packages=['labugr', 'labugr.scipy', 'labugr.dependencias', 'labugr.doc','labugr.fftpack'],
    scripts=[],
    url='http://github.com/lserraga/labugr',
    license='LICENSE.txt',
    description='Laboratorio de señales UGR.',
    long_description=open('README.md').read(),
    install_requires=['numpy>=1.13.1','matplotlib>=2.0.2']
)

