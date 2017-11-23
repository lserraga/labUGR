from labugr import *

wc = pi/4 #frecuencia de corte normalizada
M=20 #numero de muestras
n = arange(-M,M) #vector de muestras
h = wc/pi * sinc(wc*(n)/pi) #respuesta impulsiva ideal del filtro acotada 
							#entre -M y M
n = n + M #Retardo para tener un filtro causal

w, H = freqz(h, 1, whole=True) # dominio en frecuencia del filtro
H = np.fft.fftshift(H) # centrar la respuesta en frecuncia en 0
w = w-pi #Centramos el vector de frecuencias en 0


figure()

#Representamos la respuesta impulsiva del filtro
subplot(3,1,1)
stem(n, h)
xlabel("n(muestras)")
ylabel("h(n)")
grid()

#Representamos el espectro en frecuencia
subplot(3,1,2)
plot(w,abs(H))
xlim([-pi/2, pi/2])
plt.vlines([-wc,wc],0,1.2,color='g',lw=2.,linestyle='--')
xlabel("w(muestras/s)")
ylabel("|H(w)|")
grid()

#Representamos el espectro en frecuencia end dB
subplot(3,1,3)
plot(w,20*np.log10(abs(H)))
xlim([-pi/2,pi/2])
ylim([-40, 10])
plt.vlines([-wc,wc],10,-40,color='g',lw=2.,linestyle='--')
xlabel("w(muestras/s)")
ylabel("|H(w)|(dB)")
grid()

plt.tight_layout()

show()