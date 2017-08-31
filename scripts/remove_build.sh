#!/bin/bash

#Eliminar los archivos creados con pip3 install -e o setup.py build
#Utilizar cuando se desarrolla en python 3.5

#Archivos y carpetas de la build
sudo rm -r labugr.egg-info

sudo rm -r build

sudo rm -r dist

sudo rm labugr/__config__.py 

#Archivos C generados dinamicamente
sudo rm labugr/fftpack/src/dct.c 

sudo rm labugr/fftpack/src/dst.c 

sudo rm labugr/fftpack/convolvemodule.c 

sudo rm labugr/fftpack/_fftpackmodule.c

sudo rm labugr/integrate/_lib/_ccallbac_c.c

sudo rm labugr/integrate/_lib/_ccallbac_c.c

sudo rm labugr/signal/src/lfilter.c

sudo rm labugr/signal/src/correlate_nd.c

#Archivo generado por distutils
sudo rm MANIFEST

#Archivo generado por cythonyze
sudo rm cythonize.dat

#Eliminado los links en la distribucion de python
directorio=$(pwd)

cd -P /usr/local/lib/python3.5/dist-packages

sudo sed -i '/proyecto/d' ./easy-install.pth

sudo rm labugr.egg-link

cd -P $directorio

echo Archivos de instalacion eliminados
