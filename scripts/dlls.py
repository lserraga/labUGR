import os
from zipfile import ZipFile
from delocate import wheeltools
from hashlib import sha256

"""Script para copiar los dlls utilizados por la libreria a la wheel. Genera las entradas que 
deben ser anadidas a RECORD"""

mingw_path = r'C:\Program Files (x86)\mingw-w64\i686-4.8.1-posix-dwarf-rt_v3-rev2\mingw32\bin'
dlls = []
wheel_path = r'labugr-1.0.0-cp36-cp36m-win32.whl'
wheel_path2 = r'C:\Users\LuikS\Desktop\labugr\scripts\labugr-1.0.0-cp36-cp36m-win32.whl'


for file in os.listdir(mingw_path):
	if file.endswith(".dll"):
		file_path = os.path.join(mingw_path, file)
		dlls.append(file_path)
		with open(file_path, 'r') as f:
			data = f.read()
		 	hash_val = sha256(data.encode('utf-8'))
		print ('labugr/dll_libs/'+file+',sha256='+','+str(os.path.getsize(file_path)))

print(os.getcwd())
for dll in dlls:
	with wheeltools.InWheel(wheel_path, wheel_path):
			shutil.copy2(lib_path, pjoin('labugr','dll_libs'))

