# Por si se llama desde otra carpeta cambiar el directorio
# a labugr
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/..

py_versions=('3.4.7' '3.5.4' '3.6.2')
aux=4

for py in ${py_versions[@]};do
	# Activando la version de python correspondiente
	pyenv global $py
	ENV="${PYENV_ROOT}/versions/${py}/bin"

	# Instalando requisitos y generando la wheel
	"${ENV}/pip" install wheel numpy cython
	"${ENV}/pip" wheel --no-deps . -w wheelhouse/

	# Instalando la libreria y corriendo los tests
	"${ENV}/pip" install pytest nose
	wheel=$(find . -name "*cp3${aux}*")
	"${ENV}/pip" install $wheel
	"${ENV}/python" -c 'import labugr; labugr.test_all()'
	let "aux+=1"
done

# Anadiendo las dependencias externas a la wheel
pyenv global system
pip3 install delocate
for whl in wheelhouse/*.whl
do
	delocate-wheel $whl
done

# Cargamos la contraseña de pipy desde un archivo txt
TWINE_PASSWORD=$(<scripts/pipyPass.txt)
pip3 install twine
# Subiendo las wheels a pypi
twine upload wheelhouse/*.whl -u lserraga -p "${TWINE_PASSWORD}"