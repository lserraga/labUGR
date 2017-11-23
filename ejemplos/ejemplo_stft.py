from labugr import *

Fs = 500
f = 4
T = 1/f
t = linspace(0, 1, 1*Fs)
h = square(2*pi*f*(t+1/4)) + rand(len(t))/2.5-0.2

# Transformada
frec, tiempo, Zxx = stft(h, Fs)
print(len(frec), len(tiempo), shape(Zxx)) #129 5 (129, 5) 

# Comprobando que cumple COLA
print(check_COLA(windows.hann(256), 256, 128)) # True

# Comprobamos que se puede reconstruir la senal 
tiempo_rec, reconstruida = istft(Zxx, Fs)
print(allclose(h, reconstruida[:1*Fs])) # True

# Representando los intervalos
figure(1)
subplot(2,2,1)
plot(t, h)
title("h(t) entre %.2f y %.2f" % (tiempo[0], tiempo[1]))
xlim(tiempo[0], tiempo[1])
subplot(2,2,2)
plot(t, h)
title("h(t) entre %.2f y %.2f" % (tiempo[1], tiempo[2]))
xlim(tiempo[1], tiempo[2])
subplot(2,2,3)
plot(t, h)
title("h(t) entre %.2f y %.2f" % (tiempo[2], tiempo[3]))
xlabel('t')
xlim(tiempo[2], tiempo[3])
subplot(2,2,4)
plot(t, h)
title("h(t) entre %.2f y %.2f" % (tiempo[3], tiempo[4]))
xlabel('t')
xlim(tiempo[3], tiempo[4])

# Representando los epectros de los intervalos
Zxx = Zxx.T
figure(2)
subplot(2,2,1)
title("Intervalo I")
plot(frec, Zxx[0])
subplot(2,2,2)
title("Intervalo II")
plot(frec, Zxx[1])
subplot(2,2,3)
title("Intervalo II")
xlabel('f')
plot(frec, Zxx[2])
subplot(2,2,4)
title("Intervalo IV")
xlabel('f')
plot(frec, Zxx[3])

show()
