#Versiones de Python para las que queremos crear las wheels
pythonV='cp34-cp34m cp35-cp35m cp36-cp36m'
#pythonV='cp35-cp35m'

#Comprobamos que las credenciales estan en pipyPass.txt
if [ ! -f /labugr/scripts/pipyPass.txt ]; then
    echo "Credenciales para PiPy no se encuentran en scripts/pipyPass.txt"
    exit
fi

#Instalamos la libreria atlas para la compilacion de labugr
yum install -y atlas-devel

#Para cada version de python instalamos numpy y creamos una wheel
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

#Cargamos la contraseña de pipy desde un archivo txt
TWINE_PASSWORD=$(</labugr/scripts/pipyPass.txt)
#Instalamos twine y subimos las wheels creadas a pipy
"${ENV}/pip" install twine
"${ENV}/twine" upload wheelhouseOK/*.whl -u lserraga -p "${TWINE_PASSWORD}"