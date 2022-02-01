Partimos de unos datos de FHR y UC que consisten en 538 curvas con 9600 puntos (40 minutos a 4Hz). 
Sobre estos datos calculamos la correlación cruzada con lag. Para ello desplazamos la UC un lag en el tiempo y
calculamos la correlación cruzada (NaN save pandas) con la FHR sin desplazar, variando el lag obtenemos la función de
correlación cruzada para cada curva que luego usaremos en el clasificador.


![Data desc by class](../Plots/cc_data/Desc_plot.png)
![Data correlation class](../Plots/cc_data/Class_correlation.png)
![Class Distribution](../Plots/cc_data/Class_Distribution.png)

