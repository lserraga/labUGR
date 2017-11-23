from labugr import *

# Vector de tiempos
fs = 1000
t = linspace(-1, 1, 2*fs)
N = len(t)

# Senales cuadrada y triangular utilizando una ventana y padding
cuadrada = 0.5 * get_window('boxcar', int(N/2))
cuadrada = np.lib.pad(cuadrada, (int(N/4), int(N/4)), 'constant', constant_values=(0, 0))
triangular = get_window('triang', int(N/2))
triangular = np.lib.pad(triangular, (int(N/4), int(N/4)), 'constant', constant_values=(0, 0))

# Convolucion
convolucion = convolve(cuadrada, triangular)/sum(cuadrada)

figure(0)
plt.suptitle("Convolucion")
subplot(2, 2, 1)
title('h(t)')
ylim(0,1)
plot(cuadrada)
xlabel('t(ms)')
subplot(2, 2, 2)
title('g(t)')
plot(triangular)
xlabel('t(ms)')
subplot(2, 1, 2)
title('h(t)*g(t)')
xlabel('t(ms)')
plot(convolucion)

show()
