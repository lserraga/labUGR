from labugr import *

#Ejercicio 2.1
#Generando la ecuacion de la impedancia combinada
z1 = ((70+60j)*(40-25j))/((70+60j)+(40-25j))
#Obteniendo el modulo
modulo = abs(z1)
#Obteniendo la fase
fase = angle(z1)
#Obteniendo la parte real y la parte imaginaria
preal = z1.real
pimag = z1.imag
#Obteniendo la representacion en forma polar
formaPolar = (abs(x), angle(x))

#Definiendo el voltaje
faseDec = deg2rad(15)
v1 = 12*cos(faseDec) + 12j*sin(faseDec)

#Obteniendo la corriente
i1 = v1/z1
print('Parte real: {}\nParte imaginaria {}\nModulo: {}\nFase: {}'.format(i1.real,
    i1.imag, abs(i1), np.angle(i1)))


#Ejercicio 2.2

#Vector de frecuencias de 200 elementos entre 0 y 2000 Hz
f = linspace (0, 2000, 200)
#Ecuacion de la respuesta
H = (2*pi*300) / (2j*pi*f + 2*pi*300)


figure()
#Parte real
subplot(2, 2, 1)
title('Parte real')
plot(f, H.real, 'o')
#Parte imaginaria
subplot(2, 2, 2)
title('Parte imaginaria')
plot(f, H.imag, 'o')
#Modulo
subplot(2, 2, 3)
title('Modulo')
plot(f, 20*np.log10(abs(H)), 'o')
xlabel('f(Hz)')
ylabel('dB')
#Fase 
subplot(2, 2, 4)
title('Fase')
plot(f, np.angle(H, deg=True), 'o')
xlabel('f(Hz)')
ylabel('Grados decimales')

show()