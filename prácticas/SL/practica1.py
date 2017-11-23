from labugr import *

# APARTADO 1.1
def senal1_x(t):
    resultado = zeros(0)
    for i in range(0, t.size):
        if -1 < t[i] < 0:
            resultado.append(t[i])
        else:
            resultado.append(0)
    return resultado

def senal_x(t):
    tamano = t.size
    resultado = zeros(tamano)
    for i in range(0, tamano):
        if -1 < t[i] < 0:
            resultado[i] = t[i]
    return resultado


FS = 1e4 #Frecuencia de Muestreo
dt = 1/FS #Periodo de muestreo (tamano de paso del vector discreto de tiempos)
t = arange(-3, 3, dt)
y1 = senal_x(t)
y2 = senal_x(-t)
y3 = senal_x(-3-t)
y4 = senal_x(t/3)
yEven = (senal_x(t)+senal_x(-t))/2
yOdd = (senal_x(t)-senal_x(-t))/2
ySum = yEven + yOdd

# Dibujando graficas
fig1=figure(1)
fig1.canvas.set_window_title("Apartado 1.1.2")
plot(t, y1)
axis([-1.5, 1.5, -1.5, 1.5])
ylabel('x(t)')
xlabel('t')
title('x(t)')

fig2=figure(2)
fig2.canvas.set_window_title("Apartado 1.1.3")
subplot(221)
ylabel('x(t)')
title('x(t)')
xlabel('t')
plot(t, y1)
subplot(222)
ylabel('x(-t)')
title('x(-t)')
xlabel('t')
plot(t, y2)
subplot(223)
title('x(-3-t)')
xlabel('t')
ylabel('x(-3-t)')
plot(t, y3)
subplot(224)
title('x(t/3)')
xlabel('t')
ylabel('x(t/3)')
plot(t, y4)

fig3=figure(3)
fig3.canvas.set_window_title("Apartado 1.1.4")
subplot(141)
ylabel('x(t)')
title('x(t)')
xlabel('t')
plot(t, y1)
subplot(142)
ylabel('x(-t)')
title('x(-t)')
xlabel('t')
plot(t, yEven)
subplot(143)
title('x(-3-t)')
xlabel('t')
ylabel('x(-3-t)')
plot(t, yOdd)
subplot(144)
title('x(t/3)')
xlabel('t')
ylabel('x(t/3)')
plot(t, ySum)



# APARTADO 1.2

Fs = 1000
dt = 1./Fs
t = arange(0, 20, dt)

fig4=figure(4)
fig4.canvas.set_window_title("Apartado 1.2.1")

# Senal 1
x1 = 0.5 * sin(t)
f1 = 1/(2*pi)
T1 = 1/f1

subplot(221)
plot(t, x1, label='line 1', linewidth=3)
plot([T1, T1], [-2, 2], 'r-', lw=2)
grid()
xlabel('t')
ylabel('x_1(t)')
axis([0, 15, -2, 2])
title('x_1(t), T_1 = ' + str(round(T1, 4)), fontweight='bold')

# Senal 2
x2 = 1.5*cos(pi/2*t + 2)
f2 = 1/4.
T2 = 1/f2

subplot(222)
plot(t, x2)
plot([T2, T2], [-2, 2], 'r-', lw=2)
grid()
xlabel('t')
ylabel('x_2(t)')
axis([0, 15, -2, 2])
title('x_2(t), T_2 = ' + str(round(T2, 4)), fontweight='bold')

# Senal 3
x3 = 2*cos(3*pi/2*t + 2)
f3 = 3/4.
T3 = 1/f3

subplot(223)
plot(t, x3)
plot([T3, T3], [-3, 3], 'r-', lw=2)
grid()
xlabel('t')
ylabel('x_3(t)')
axis([0, 15, -3, 3])
title('x_3(t), T_3 = ' + str(round(T3, 4)), fontweight='bold')


# Senal 4
x4 = 3*sin(pi*t) + 1.5*sin(pi/4*t)
f4 = 1/8.
T4 = 1/f4

subplot(224)
plot(t, x4)
plot([T4, T4], [-5, 5], 'r-', lw=2)
grid()
xlabel('t')
ylabel('x_4(t)')
axis([0, 15, -5, 5])
title('x_4(t), T_4 = ' + str(round(T4, 4)), fontweight='bold')


#APARTADO 1.2.2

fig5=figure(5)
fig5.canvas.set_window_title("Apartado 1.2.2")
# Senal 1
subplot(221)
plot(t, x1+x2)
plot([T1, T1], [-2, 2], 'r-', lw=2)
grid()
title('x_1(t), T_1 = ' + str(round(T1, 4)), fontweight='bold')

# Senal 2
x6 = x2+x3
subplot(222)
plot(t, x6)
plot([T2, T2], [min(x6), max(x6)], 'r-', lw=2)
grid()
xlabel('t')
ylabel('x_2(t)')
axis([0, max(t), min(x6), max(x6)])
title('x_2(t), T_2 = ' + str(round(T2, 4)), fontweight='bold')

# Senal 3
subplot(223)
plot(t, x2+x4)
plot([T3, T3], [-3, 3], 'r-', lw=2)
grid()
xlabel('t')
ylabel('x_3(t)')
title('x_3(t), T_3 = ' + str(round(T3, 4)), fontweight='bold')


#APARTADO 1.3.1
def ustep(t):
    tamano=t.size
    resultado = zeros(tamano)
    for i in range(0, tamano):
        if t[i]>0:
            resultado[i] = 1
    return resultado

#APARTADO 1.3.2
def delta(t):
    resultado = zeros(t.size)
    resultado[np.where(abs(t)<1e-11)] = 1 # Son valores discretos por lo que no se puede hacer == 0
    return resultado

#APARTADO 1.3.3
def rampa(t):
    tamano=t.size
    resultado = zeros(tamano)
    for i in range(0, tamano):
        if t[i]>0:
            resultado[i] = t[i]
    return resultado

# Apartado 1.3.4
Fs = 1000. # Frecuencia de muestreo
dt = 1/Fs # Periodo de muestreo
t = arange(-3, 3, dt) # Vector de tiempos
t1, t2 = -1, 1
y = ustep(t-t1) - ustep(t-t2)

fig6=figure(6)
fig6.canvas.set_window_title("Apartado 1.3.4")
# Senal 1
x1 = ustep(t)
subplot(221)
plt.step(t, x1)
grid()
xlabel('t')
axis([-5, 5, 0, 1.1])
title('Escalon unitario, u(t)', fontweight='bold')

# Senal 2
x2 = delta(t)

subplot(222)
plot(t, x2)
grid()
xlabel('t')
axis([-5, 5, 0, 1.1])
title('Impulso unitario, (t)', fontweight='bold')

# Senal 3
x3 = rampa(t)

subplot(223)
plot(t, x3)
grid()
xlabel('t')
axis([-5, 5, 0, 1.1])
title('Senal rampa, r(t)', fontweight='bold')


# Senal 4
subplot(224)
plot(t, y)
grid()
xlabel('t')
axis([-5, 5, 0, 1.1])
title('Senal rectangular, rect(t,t1,t2)', fontweight='bold')

plt.tight_layout()
show()