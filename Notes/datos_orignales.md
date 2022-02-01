## Datos originales
Originalmente hay 552 patrones de longitud 21620 para la frecuencia fetal y las contracciones uterinas [(552, 21620)]
![UC Dismissed](..\Plots\Original_Data\Means_plots.png)

Sobre estos datos usamos el siguiente criterio para detectar curvas de UC que no van a tener la suficiente información
como para poder tener poder predictivo. Sumamos dos cantidades:

- La cantidad absoluta en número de puntos de la parte constante distinta de cero y  de  `np.nan`. Para este cálculo consideramos
solo las partes constantes con una duración superior a 5 segundos (20 puntos a 4Hz)
- La cantidad absoluta en número de puntos de ceros en la curva, independientemente de si son consecutivos.

Ordenamos las curvas respecto a este criterio y retiramos el 5% superior, lo que corresponde a 14 curvas (redondeando).
Las curvas eliminadas son [1104, 1119, 1130, 1134, 1149, 1155, 1158, 1186, 1188, 1258, 1327, 1376, 1451, 1477], siendo 6
de clase cero 8 de clase 1.
