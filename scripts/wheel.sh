pythonV='cp34-cp34m cp35-cp35m cp36-cp36m'
for version in $pythonV
do
	ENV=opt/python/$version/bin
	"${ENV}/pip" install numpy
	"${ENV}/pip" wheel labugr/ -w wheelhouse/
done
for whl in wheelhouse/*.whl
do
	auditwheel repair $whl -w wheelhouseOK/
done

"${ENV}/pip" install twine
"${ENV}/twine" upload wheelhouseOK/*.whl -u lserraga -p Bebopjazz95