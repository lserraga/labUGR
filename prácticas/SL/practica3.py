from labugr import *
from matplotlib.widgets import Slider



FS = 1e2 #Frecuencia de Muestreo
dt = 1/FS #Periodo de muestreo (tamano de paso del vector discreto de tiempos)
t = arange(-8, 8, dt)

T=4
#Creando la funcion
x1 = lambda t: 0.5*square(2*pi/T*(t-T/4))+0.5
x = lambda t: sawtooth(2 * pi /T * (t-T/2))
x2 = lambda t: 0.5*square(2*pi/4*(t-4/4))+0.5 + sawtooth(2 * pi / 4 * (t-4/2))

#Funcion para obtener los primeros n+1 coeficientes a de x
def aCof(y,T0,n):
    resultado=zeros(n+1)
    #Valor de a0
    resultado[0]=(quad(y,-T0/2, T0/2)[0])*(2/T0)
    #Definimos la funcion que deseamos integrar
    def z(t):
        return y(t) * cos(2 * pi * i * (1 / T0) * t)
    #Calculo del resto de coeficientes
    for i in range(1,n+1):
        resultado[i]=(quad(z,-T0/2, T0/2)[0])*(2/T0)
    return resultado

#Funcion para obtener los coeficientes b de x
def bCof(y,T0,n):
    resultado = zeros(n+1)
    #Valor de b0 es 0
    # Definimos la funcion que deseamos integrar
    def z(t):
        return y(t) * sin(2 * pi * i * (1 / T0) * t)
    # Calculo del resto de coeficientes
    for i in range(1,n+1):
        resultado[i]=(quad(z,-T0/2, T0/2)[0])*(2/T0)
    return resultado

#Funcion para obtener el desarrollo en serie de fourier de x
#Devuelve 3 valores: los coeficientes de a, de b y el desarrollo en serie
def serieFourier(x,T,n,t):
    resultado = zeros(len(t))
    #Calculamos los coeficientes a y b
    an = aCof(x,T,n)
    ab = bCof(x, T, n)
    #Para cada elemento del vector de tiempos calcular la suma de los coeficientes
    for i in range(0, len(t)):
        resultado[i] += an[0] / 2
        for i2 in range(1,n+1):
            resultado[i] += an[i2]*cos(2*pi*i2/T*t[i])+ab[i2]*sin(2*pi*i2/T*t[i])
    return (resultado,an,ab)

#Funcion para obtener los coeficientes complejos de x a partir
#de los coeficientes a y b
def cCoef(a,b):
    #Positive part
    c=(a-1j*b)/2
    #Adding the negative part
    for i in range(1,len(a)):
        c=insert(c,0,(a[i]+1j*b[i])/2)
    return c


n=1
N=arange(0,n+1)
z,aS,bS=serieFourier(x,T,n,t)
fig=figure(1)
subplot(311)
plot(t,x(t))
ylabel("Funcion")
aux,=plot(t, z)
plt.legend(("Funcion original","Serie de Fourier"))
armonicos = Slider(plt.axes([0.25, 0.01, 0.65, 0.03]), 'Armonicos',1, 40,
                   valfmt='%1.0f',valinit=1,fill=False)
xlabel('t')

cMod=subplot(313)
stem(arange(-n,n+1), abs(cCoef(aS,bS)),'r--',markerfmt='ro',basefmt='k--')
ylabel("Modulo Coefcientes C")
xlabel('n')

coefs=subplot(312)
stem(arange(0,n+1), aS,'r--','Emptied',markerfmt='ro',basefmt='k--')
stem(arange(0,n+1),bS,'b--',markerfmt='bo',basefmt='k--')
plt.legend(("a Coefs","b Coefs"))




def update(value):
    global n,cMod,coefs
    #The values in the slider will go from 0 to 50 on an exponential scale
    arm = int(exp(armonicos.val/10))
    armonicos.valtext.set_text(arm)
    if(arm==n):
        return
    nuevaSerie,nuevoA,nuevoB=serieFourier(x,T,arm,t)
    aux.set_ydata(nuevaSerie)
    cMod.remove()
    coefs.remove()
    del cMod,coefs
    cMod = subplot(313)
    stem(arange(-arm, arm + 1), abs(cCoef(nuevoA, nuevoB)), 'r--', markerfmt='ro', basefmt='k--')
    ylabel("Modulo Coefcientes C")
    xlabel('n')
    coefs = subplot(312)
    stem(arange(0, arm + 1), nuevoA, 'r--', 'Emptied', markerfmt='ro', basefmt='k--')
    stem(arange(0, arm+1), nuevoB, 'b--', markerfmt='bo', basefmt='k--')
    plt.legend(("a Coefs", "b Coefs"))
    fig.canvas.draw_idle()
    n = arm
armonicos.on_changed(update)
show()