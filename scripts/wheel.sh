ENV=opt/python/cp35-cp35m/bin
"${ENV}/pip" install numpy
"${ENV}/pip" wheel labugr/ -w wheelhouse/
"${ENV}/pip" install twine
auditwheel repair wheelhouse/* -w wheelhouseOK/
"${ENV}/twine" upload wheelhouseOK/*.whl -u lserraga -p Bebopjazz95