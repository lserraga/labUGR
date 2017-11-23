from labugr import *

# Parametros del filtro
K = 28
fc = 1e3
wc = 2*pi*fc
Q = 40

# Creacion del sistema
numerador = [K*1, 0] 
denominador = [Q/wc, 1, Q*wc]
sistema = TransferFunction(numerador, denominador)

# Rrepresentacion en el plano Z
#zplane(sistema)

# Respuesta impulsiva
t_imp, y_imp = impulse(sistema)

# Respuesta al escalon 
t_esc, y_esc = step(sistema)

# Figura para la respuesta impulsiva y al escalon
figure(1)
subplot(2, 1, 1)
title('Respuesta impulsiva')
ylabel('Magnitud')
grid()
plot(t_imp, y_imp)
subplot(2, 1, 2)
title('Respuesta al escalon')
ylabel('Magnitud')
plot(t_esc, y_esc)
grid()
xlabel('t(s)')



# Representacion en frecuencia
w_frecs, magn_frecs = freqresp(sistema)

# Figura para la respuesta en frecuencia
figure(2)
plt.suptitle("Respuesta en frecuencia")
subplot(2, 1, 1)
xlim([1e2, 1e5]) # Los valores interesantes se encuentran en este intervalo
title('Parte real')
ylabel('Magnitud')
plt.semilogx(w_frecs, magn_frecs.real, 'r') # Eje horizontal en escala logaritmica
grid(True, which="both")
plt.axvline(wc) # Linea vertical representando la frecuencia de corte
subplot(2, 1, 2)
xlim([1e2, 1e5])
title('Parte imaginaria')
ylabel('Magnitud')
plt.semilogx(w_frecs, magn_frecs.imag, 'r')
plt.axvline(wc)
xlabel('w(rad/s)')
grid(True, which="both")


# Diagrama de Bode
w_bode, magn_bode, fase_bode = bode(sistema)

figure(3)
plt.suptitle("Diagrama de Bode")
subplot(2, 1, 1)
title('Magnitud')
ylabel('Magnitud (dB)')
plt.semilogx(w_bode, magn_bode, 'r')
xlim([1e2, 1e5])
grid(True, which="both")
plt.axvline(wc)
subplot(2, 1, 2)
title('Fase')
ylabel('Fase (grados decimales)')
plt.semilogx(w_bode, fase_bode, 'r')
plt.axvline(wc)
xlim([1e2, 1e5])
xlabel('w(rad/s)')
grid(True, which="both")


# Ejemplo 5.3



show()
