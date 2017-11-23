from labugr import *

# Parametros
fs = 1000
f_nyq = fs/2
fc = 300
wc = 2*pi*fc
coefs = 37

# Parametros para firwin2 y firls
frecs = linspace(0, f_nyq)
ganancia = zeros(len(frecs))
ganancia[frecs>f_nyq/2] = 1

frecs = (0, 100, 200, 300, 400, 500)
ganancia = (0, 0, 0, 1 ,1, 1)

# Creacion de filtros
por_ventanas = firwin(coefs, fc, nyq=f_nyq, pass_zero=False) 
por_muestreo = firwin2(coefs, frecs, ganancia, nyq=f_nyq) 
por_rizado = firwin2(coefs, frecs, ganancia, nyq=f_nyq) 


for filtro in (por_ventanas, por_muestreo, por_rizado):
	w, h = freqz(filtro)
	plt.semilogy(w*f_nyq/pi, abs(h))

show()


import matplotlib.pyplot as plt
fig, axs = plt.subplots(2)
fs = 10.0  # Hz
desired = (0, 0, 1, 1, 0, 0)
for bi, bands in enumerate(((0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 4.5, 5))):
    fir_firls = firls(73, bands, desired, nyq=fs/2)
    fir_firwin2 = firwin2(73, bands, desired, nyq=fs/2)
    hs = list()
    ax = axs[bi]
    for fir in (fir_firls, fir_firwin2):
        freq, response = freqz(fir)
        hs.append(ax.semilogy(0.5*fs*freq/np.pi, np.abs(response))[0])
    for band, gains in zip(zip(bands[::2], bands[1::2]),
                           zip(desired[::2], desired[1::2])):
        ax.semilogy(band, np.maximum(gains, 1e-7), 'k--', linewidth=2)
    if bi == 0:
        ax.legend(hs, ('firls', 'firwin2'),
                  loc='lower center', frameon=False)
    else:
        ax.set_xlabel('Frequency (Hz)')
    ax.grid(True)
    ax.set(title='Band-pass %d-%d Hz' % bands[2:4], ylabel='Magnitude')

fig.tight_layout()
plt.show()