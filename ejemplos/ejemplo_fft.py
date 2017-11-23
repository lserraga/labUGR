from labugr import *

Fs = 500
F = 4
T = 1/F

#Vector de tiempos
t = linspace(-0.5, 0.5, 1*Fs)

#Senal cuadrada
h = square(2*pi*F*(t+T/4))

#Transformada
H = abs(fft(h))

#Vector de frecuencias centrado en 0
f = np.fft.fftfreq(len(t), 1/Fs)

#Otra opcion para centrarla sin utilizar fftfreq
# f = linspace(-Fs/2, Fs/2, len(t))
# H0 = np.fft.fftshift(H)


#Senal cuadrada
figure()
title('x(t)')
xlabel('t')
ylabel('Amplitud')
plot(t, h)

#Transformada 
figure()
title('X(f)')
xlabel('f')
ylabel('Amplitud')
plot(H)

#Transformada centrada en 0
figure()
title('X(f) centrada en 0')
xlabel('f')
ylabel('Amplitud')
plot(f, H)

#Transformada, solo espectro positivo
figure()
title('X(f) parte positiva del espectro')
xlabel('f')
ylabel('Amplitud')
plot(H[:250])

show()