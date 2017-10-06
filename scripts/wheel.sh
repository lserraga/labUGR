#Versiones de Python para las que queremos crear las wheels
pythonV='cp34-cp34m cp35-cp35m cp36-cp36m'

#Decargamos el paquete de github
git clone https://github.com/lserraga/labugr.git

#Instalamos la libreria atlas para la compilacion de labugr
yum install -y atlas-devel

#Para cada version de python instalamos numpy, cython y creamos una wheel
for version in $pythonV
do
	ENV=opt/python/$version/bin
	"${ENV}/pip" install numpy cython
	"${ENV}/pip" wheel --no-deps labugr/ -w wheelhouse/
done

#Para cada wheel creada utilizamos auditwheel para comprobar que esta
#cumple con los estandares de manylinux y establecer las etiquetas 
#adecuadas para que pip pueda utilizar la wheel
for whl in wheelhouse/*.whl
do
	auditwheel repair $whl -w wheelhouseOK/
done

#La contrasena de PyPi es el primer parametro all llamar al script
TWINE_PASSWORD=$1
#Instalamos twine y subimos las wheels creadas a pipy
"${ENV}/pip" install twine
"${ENV}/twine" upload wheelhouseOK/*.whl -u lserraga -p "${TWINE_PASSWORD}"