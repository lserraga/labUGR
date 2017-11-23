from labugr import *

# Generarcion de la senal de entrada
fs = 1000
f_nyq = fs/2
t = linspace(0, 3, fs*3)
f1 = 10
f2 = 450
x = sin(2*pi*f1*t) + sin(2*pi*f2*t)

# Filtro paso alta butterworth
num, den = butter(5, 300/f_nyq, btype='highpass')

w, h = freqz(num, den)
figure()
plt.plot(w*f_nyq/(pi), abs(h))
#
zi = lfilter_zi(num, den)

lfilter_sin_zi = lfilter(num, den, x)
lfilter_con_zi, _ = lfilter(num, den, x, zi=zi*x[0])

figure()
plot(t[2.98<t],x[2.98<t])
plot(t[2.98<t],lfilter_sin_zi[2.98<t], 'r')
plot(t[2.98<t],lfilter_con_zi[2.98<t], 'g')
show()