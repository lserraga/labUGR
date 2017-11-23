from labugr import *
from urlib.request import urlretrieve

# Descarga del archivo 
link = "http://soundbible.com/grab.php?id=1272&type=mp3"
nombre = "leon.mp3"
urlretrieve(link, nombre)

# Cargamos el audio
fs, datos = read(nombre) # fs=44100Hz, datos=vector con los valores
duracion = len(datos)/fs  # 2.2197278911564626 segundos
print('Frecuencia de muestreo del audio: {}Hz'.format(fs))
print('Duracion del audio: {} segundos'.format(round(duracion, 2)))

# Separacion de los canales de audio
canal1 = datos[:, 0] # valores del canal izquierdo
canal2 = datos[:, 1] # valores del canal derecho

# Vector de tiempos
t = linspace(0, duracion, len(datos))


figure()
title("Senal de audio")
plot(t, canal1, label="Canal I")
plot(t, canal2, label="Canal II")
plt.legend()
xlabel("t(s)")
ylabel("Amplitud")

# Periodograma
frecs, DEP = periodogram(canal1, fs)
_, EP = periodogram(canal1, fs, scaling='spectrum')
_, DEP2 = periodogram(canal2, fs)
_, EP2 = periodogram(canal2, fs, scaling='spectrum')


figure()
subplot(2,1,1)
plt.semilogy(frecs, DEP, label="Canal I")
plt.semilogy(frecs, DEP2, label="Canal II")
plt.legend()
ylim([1e-8, 1e8])
title("Densidad espectral de potencia")
plt.ylabel('Amplitud(V**2/Hz)')
subplot(2,1,2)
plt.semilogy(frecs, EP, label="Canal I")
plt.semilogy(frecs, EP2, label="Canal II")
plt.legend()
ylim([1e-8, 1e8])
title("Espectro de potencia")
plt.xlabel('f(Hz)')
plt.ylabel('Amplitud(V**2)')




# Espectrograma de la senal con 4 segmentos de tiempo
muestras_ventana = int(len(datos)/4)
frecuencias, tiempos, magnitud = spectrogram(canal1, fs, mode='complex', nperseg=muestras_ventana)
_, _, magnitud2 = spectrogram(canal1, fs, mode='complex', nperseg=muestras_ventana)


figure()
plt.suptitle("Espectrograma (respuesta en frecuencia)")
subplot(2,2,1)
plot(frecuencias, magnitud.T[0].real, label="Canal I")
plot(frecuencias, magnitud2.T[0].real, label="Canal II")
title("Intervalo I: {:03.2f}-{:03.2f} segundos".format(0, tiempos[1]))
plt.legend()
xlim(0, 1500)
subplot(2,2,2)
plot(frecuencias, magnitud.T[1].real, label="Canal I")
plot(frecuencias, magnitud2.T[1].real, label="Canal II")
title("Intervalo II: {:03.2f}-{:03.2f} segundos".format(tiempos[1], tiempos[2]))
plt.legend()
xlim(0, 1500)
subplot(2,2,3)
plot(frecuencias, magnitud.T[2].real, label="Canal I")
plot(frecuencias, magnitud2.T[2].real, label="Canal II")
title("Intervalo III: {:03.2f}-{:03.2f} segundos".format(tiempos[2], tiempos[3]))
plt.legend()
xlim(0, 1500)
plt.xlabel('f(Hz)')
subplot(2,2,4)
plot(frecuencias, magnitud.T[3].real, label="Canal I")
plot(frecuencias, magnitud2.T[3].real, label="Canal II")
title("Intervalo IV: {:03.2f}-{:03.2f} segundos".format(tiempos[3], duracion))
plt.legend()
xlim(0, 1500)
plt.xlabel('f(Hz)')
show()
