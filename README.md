# Particion-de-datos-de-un-dataset
Mediante 5 tecnicas de partición se observa su desempeño al dividir el dataset en entrenamiento y pruebas esto mediante 3 datasets basicos los cuales se aplican de la siguente forma:

1. Tecnica 1 y 5 con el dataset 1
2. Tecnica 2 y 4 con el dataset 2
3. Tecnica 3 con el dataset 3

> [!NOTE]
> Esto se basa a la entrada del usuario definiendo el porcetanje destinado a Practica y el restante para test.

## _Resultados Obtenidos_

#### Resultados Por datasets
![Desempeño 270](https://github.com/JuanSalvi/Particion-de-datos-de-un-dataset/assets/91103822/a0a7e629-d7fd-4356-a0d3-a810a541501b)
![Desempeño 250](https://github.com/JuanSalvi/Particion-de-datos-de-un-dataset/assets/91103822/9d52705e-2931-46a0-bd3c-a262d19b10db)
![Desempeño 210](https://github.com/JuanSalvi/Particion-de-datos-de-un-dataset/assets/91103822/b5fe2873-202f-4261-a6a2-517b2e02c90e)

#### Resultados Globales!
![Desempeño TODAS](https://github.com/JuanSalvi/Particion-de-datos-de-un-dataset/assets/91103822/00aa5e01-8394-4c88-8a94-5975aaf171d0)

> [!IMPORTANT]
> Estos métodos se ajustan mucho diferentes necesidades como lo es **__train_test_split_random__** siendo útil cuando necesitamos una división rápida y aleatoria de datos sin embargo si deseamos mantener una distribución proporcional de las clases en ambos conjuntos es mejor optar por otra como **train_test_split_stratified** hablando de rapidez y uso proporcional.
> <br> <br>
> Cuando queremos evaluar el rendimiento del modelo de manera precisa los métodos de validación cruzada como **k_fold_cross_validation y stratified_k_fold_cross_validation** son fundamentales, cosa que vimos antes de manera grafica teniendo alta precisión y con ello una constante de iteraciones de entrenamiento. Y como restante el **random_sampling** es útil cuando necesitamos realizar nuestro muestreo aleatorio simple del conjunto de datos ayudándonos a crear conjuntos de “test y train” cuando no necesitamos una evaluación exhaustiva del modelo.
