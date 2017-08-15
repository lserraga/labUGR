#!/bin/bash

#Eliminar los archivos creados con pip3 install -e

sudo rm -r labugr.egg-info

sudo rm -r build

sudo rm -r dist

sudo rm labugr/__config__.py 

sudo rm labugr/fftpack/src/dct.c 

sudo rm labugr/fftpack/src/dst.c 

sudo rm labugr/fftpack/convolvemodule.c 

sudo rm labugr/fftpack/_fftpackmodule.c

directorio=$(pwd)

cd -P /usr/local/lib/python3.5/dist-packages

sudo sed -i '/proyecto/d' ./easy-install.pth

sudo rm labugr.egg-link

cd -P $directorio

echo Archivos de instalacion eliminados
