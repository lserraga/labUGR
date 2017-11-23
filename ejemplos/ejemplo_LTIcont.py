from labugr import *

# Parametros del circuito
L = 0.1
C = 200e-6
R = 50

# Frecuencia de corte
w0 = 1/sqrt(L*C)

# Sistema definido por la funcion de transferencia
num = 1/(L*C)
den = [1, R/L, 1/(L*C)]
H = TransferFunction(num , den)

# Polos del sistema
print('Polos del sistema: {}'.format(abs(H.poles)))

# Respuesta impulsiva
t_imp, y_imp = impulse(H)

# Respuesta al escalon 
t_esc, y_esc = step(H)

# Representacion en frecuencia
w_frecs, magn_frecs = freqresp(H)

# Diagrama de Bode
w_bode, magn_bode, fase_bode = bode(H)

# Senal de entrada
t = linspace(0, 2, 2000)
f1 = 50
f2 = 400
x = sin(2*pi*f1*t) + sin(2*pi*f2*t)

# Simulacion del sistema
t_sim, y_sim, ev = lsim(H, x, t)



# Respuesta impulsiva y escalon
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



# Respuesta en frecuencia
figure(2)
plt.suptitle("Respuesta en frecuencia")
subplot(2, 1, 1)
title('Parte real')
ylabel('Magnitud')
plt.semilogx(w_frecs, magn_frecs.real, 'r')
grid(True, which="both")
plt.axvline(w0)
plt.text(w0,min(magn_frecs.real)-0.15,'w0')
subplot(2, 1, 2)
title('Parte imaginaria')
ylabel('Magnitud')
plt.semilogx(w_frecs, magn_frecs.imag, 'r')
plt.axvline(w0)
plt.text(w0,min(magn_frecs.imag)-0.1,'w0')
xlabel('w(rad/s)')
grid(True, which="both")


# Diagrama de Bode
figure(3)
plt.suptitle("Diagrama de Bode")
subplot(2, 1, 1)
title('Magnitud')
ylabel('Magnitud (dB)')
plt.semilogx(w_bode, magn_bode, 'r')
grid(True, which="both")
plt.axvline(w0)
plt.text(w0,min(magn_bode)-10,'w0')
subplot(2, 1, 2)
title('Fase')
ylabel('Fase (grados decimales)')
plt.semilogx(w_bode, fase_bode, 'r')
plt.axvline(w0)
plt.text(w0,min(fase_bode)-20,'w0')
xlabel('w(rad/s)')
grid(True, which="both")


#Simulacion del sistema
figure(4)
plt.suptitle("Simulacion del sistema")
subplot(2, 1, 1)
title('Entrada')
ylabel('Magnitud')
plot(t, x)
xlim(1, 1.1)
grid()
subplot(2, 1, 2)
title('Salida')
ylabel('Magnitud')
plot(t, y_sim)
xlim(1, 1.1)
xlabel('t(s)')
grid()
show()