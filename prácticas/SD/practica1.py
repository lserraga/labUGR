from labugr import *

L = 30
nn = arange(-15, 15)


x3 = zeros(30)
x3[nn>-8] = 1
x3[nn>0] = 0.5*nn[nn>0] -1


x4 = zeros(30)
x4[nn<=0] = exp(nn[nn<=0])
x4[nn>0] = 1 - nn[nn>0]/10

x5 = ones(30)
x5[nn>0] = np.log10(nn[nn>0])


x6 = ones(30)
x6[nn>-6] = 4 + nn[nn>-6]
x6[nn>0] = 4 - nn[nn>0]/5


figure()
subplot(2, 2, 1)
title('x3')
ylabel('Amplitud')
stem(nn, x3)
subplot(2, 2, 2)
title('x4')
xlabel("muestras(n)")
stem(nn, x4)
subplot(2, 2, 3)
title('x5')
ylabel('Amplitud')
xlabel("muestras(n)")
stem(nn, x5)
subplot(2, 2, 4)
title('x6')
xlabel("muestras(n)")
stem(nn, x6)


x7 = sawtooth(pi/4*nn, width=0.5)
x8 = square(pi/2*nn, duty=0.25)

figure()
subplot(2, 1, 1)
title('sawtooth')
ylabel('Amplitud')
stem(nn, x7)
subplot(2, 1, 2)
title('square')
ylabel('Amplitud')
xlabel("muestras(n)")
stem(nn, x8)


show()