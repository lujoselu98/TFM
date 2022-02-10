Partimos de unos datos de FHR y UC que consisten en 538 curvas con 21620 puntos (poco más de 90 minutos a 4Hz).

Primero para que las correlaciones cruzadas no se distorsionen mucho sustituimos las partes constantes muy largas (más de 5 minutos)
por NaN que no se tienen en cuenta para el cálculo.

Sobre estos datos calculamos la distancia de correlación cruzada con lag. Para ello desplazamos la UC un lag en el tiempo y
calculamos la distancia de correlación (NaN save implementación propia) con la FHR sin desplazar, variando el lag obtenemos la función de
distancia de correlación para cada curva que luego usaremos en el clasificador.


![Data desc by class](../Plots/cdcor_Data/Desc_plot.png)
![Data correlation class](../Plots/cdcor_Data/Class_correlation.png)
![Class Distribution](../Plots/cdcor_Data/Class_Distribution.png)
