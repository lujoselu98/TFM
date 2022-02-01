Partimos de unos datos de FHR y UC que consisten en 538 curvas con 9600 puntos (40 minutos a 4Hz).
Sobre estos datos calculamos la distancia de correlación cruzada con lag. Para ello desplazamos la UC un lag en el tiempo y
calculamos la distancia de correlación (NaN save implementación propia) con la FHR sin desplazar, variando el lag obtenemos la función de
distancia de correlación para cada curva que luego usaremos en el clasificador.


![Data desc by class](../Plots/cdcor_Data/Desc_plot.png)
![Data correlation class](../Plots/cdcor_Data/Class_correlation.png)
![Class Distribution](../Plots/cdcor_Data/Class_Distribution.png)
