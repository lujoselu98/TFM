Partimos de unos datos de FHR que consisten en 538 curvas coon 21620 puntos (poco más de 90 minutos a 4Hz).

Sobre estos datos vamos a calcular la DFT de cada curva, en el proceso perdemos resolución pues solo calculamos el valor
de la transformada de Fourier para las frecuencias menores del límite de Nyquist (2Hz). La implementación es NaN save
en el sentido de que los quita de la suma del cálculo y luego está ponderada por el número de puntos que no son nulos para 
tener magnitudes comparables. Por último la recortamos a frecuencias menores a 0.1Hz y quitamos el primer punto pues este no es mas 
que la "parte no periódica" (media) de la curva. Por lo que finalmente tenemos 538 curvas de 240 puntos.  


![Data desc by class](../Plots/fft_Data/Desc_plot.png)
![Data correlation class](../Plots/fft_Data/Class_correlation.png)
![Class Distribution](../Plots/cdcor_Data/Class_Distribution.png)
