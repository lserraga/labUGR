from labugr import *

# Creacionde la senal cuadrada
Fs = 1000
f = 200
T = 1/f
t = linspace(0, 1, 1*Fs)
h = sin(2*pi*f*t) + rand(len(t))/2.5-0.2

# Calculo de la densidad espectral de potencia
frecs, DEP = periodogram(h, Fs)

# Calculo del espectro de potencia
frecs2, EP = periodogram(h, Fs, scaling='spectrum')


# Grafica
figure(2)
subplot(2,1,1)
plt.semilogy(frecs, DEP)
ylim([1e-7, 1e2])
title("Densidad espectral de potencia")
plt.ylabel('Amplitud(V**2/Hz)')
subplot(2,1,2)
plt.semilogy(frecs2, EP)
ylim([1e-7, 1e2])
title("Espectro de potencia")
plt.xlabel('f(Hz)')
plt.ylabel('Amplitud(V**2)')

show()


