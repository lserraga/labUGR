# Por si se llama desde otra carpeta cambiar el directorio
# a labugr
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/..

py_versions=('3.4.7' '3.5.4' '3.6.2')
aux=4

for py in ${py_versions[@]};do
	pyenv global $py
	ENV="${PYENV_ROOT}/versions/${py}/bin"
	"${ENV}/pip" install wheel numpy cython
	# "${ENV}/pip" wheel --no-deps . -w wheelhouse/
	"${ENV}/pip" install pytest nose
	wheel=$(find . -name "*cp3${aux}*")
	"${ENV}/pip" install $wheel
	"${ENV}/python" -c 'import labugr; labugr.test_all()'
	let "aux+=1"
done

"${ENV}/pip" install delocate

for whl in wheelhouse/*.whl
do
	"${ENV}/delocate-wheel" $whl
done

#Cargamos la contraseña de pipy desde un archivo txt
TWINE_PASSWORD=$(</labugr/scripts/pipyPass.txt)
"${ENV}/pip" install twine
"${ENV}/twine" upload wheelhouse/*.whl -u lserraga -p "${TWINE_PASSWORD}"