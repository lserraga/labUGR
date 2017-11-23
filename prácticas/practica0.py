from labugr import *
import timeit

# Pimer ejercicio

A = array ([(1, 3, 5, 8), (2, 6, 5, 3), (4, 1, 9, 7), (1, 8, 0, 2)])

B = array ([(1, 9, 5, 8), (12, 5, 5, 9), (4, 2, 9, 74), (0, 6, 0, 3)])

print ("La matriz A:\n %s" % A)
print ("La matriz B:\n %s" % B)

print ("La segunda fila de B:\n %s" % B[1, :])
print ("La cuarta columna de B:\n %s" % B[:, [3]])
print ("Submatriz de B:\n %s" % B[0:2, 0:2])
print ("3*A: \n%s \nA-7: \n%s" % (3 * A, A - 7))
print ("A por B traspuesta:\n %s" % np.dot (A, B.transpose ()))
print ("Inversa de A:\n %s" % np.linalg.inv (A))
print ("Inversa de B:\n %s" % np.linalg.inv (B))

# Segundo ejercicio
c = linspace(0, pi, 20)
d = arange(0, 10, 10 / 1000.)  # Sin en punto python los considera enteros y el resultado es 0

print(d.size)

# Tercer ejercicio
z = rand(250)

print("Componentes mayores que 0.9: \n%s" % z[z > 0.9])
print("Componentes menores o igual que 0.15: \n%s" % z[z <= 0.15])


# Carto ejercicio
def cuadradoNaturales(N):

    # Primer metodo
    def metodo1():
        resultado_metodo1 = zeros(N, dtype=int)
        for i in resultado_metodo1:
            resultado_metodo1[i] = i**2
        return resultado_metodo1

    # Segundo metodo
    def metodo2():
        numeros = arange (0, N)
        resultado_metodo2 = numeros**2
        return resultado_metodo2

    t1 = timeit.timeit(metodo1,  number=500000)
    t2 = timeit.timeit(metodo2,  number=500000)

    return t1, t2


tiempos = cuadradoNaturales(15)

print ("El primer metodo tarda %.2f segundos y el segundo %.2f segundos" % (tiempos[0], tiempos[1]))

input("Pulse cualquier tecla para cerrar...")
