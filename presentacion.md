# Guion de presentacion - Examen Final Predictiva

Duracion objetivo: hasta 15 minutos.

Idea general: contar una historia de negocio simple. Primero se define el problema de churn, despues se muestra que el dataset tiene senal, luego se explica como se evaluaron los modelos y finalmente se presenta CatBoost como herramienta para priorizar acciones de retencion.

El orden este mantenelo:
Diapo1: Presentacion del tema (titulo, nombre: Matias Goldschmidt y examen final - Analtica Predictiva)
Diapo 2: Caso de negocio: Incluir la cantidad de clientes en el dataset, el porcentaje de churn. Incluir el problema de negocio, que se busca solucionaar con el modelo predictivo y cuanto ma scaro es perder un cliente a retenerlo.
Diapo3: Dataset: Incluir  cantidad total de columnas, cantidad de variables predictoras, la limpieza realizada y la division de las variables predictoras en demograficas, servicios, contrato y facturacion. Para cada division aclarar el impacto en el churn.
Diapo 4: EDA: Hacer un grafico en el cual se muestra como afectan las principales categorias de cada variable importante en el churn. Poner al costado alguna conclusion importante del EDA.
Diapo5: Porque se elige el PR-AUC como metrica pra el modelo y el F2 para el umbral. Hablar de que es un DS desbalanceado.
Diapo6: Validacion. Explicar train y test y como funciona, y como se uso en este caso.
Diapo7: Modelo Baseline: Mostrar el dummy y la regresion logistica. Poner que variables se usaron y las metricas que devolvio el baseline. Acalrar porque se usa la regresion logistica como baseline.
Diapo8: Mostrar los modelos probados en u grafico. EJE X: PR-AUC promedio en CV y eje Y cada modelo. A la derecha del grafico poner alguna conclusion de los resultados.
Diapo 9: Hablar del modelo final. 
Diapo 10: Hablar del modelo final
En estas dos diapositivas inclui los hiperparametros del modelo final, las metricas, la matriz de confusion, el grafico para la eleccion del umbral y alguna cosa mas que consideres.
Diapo 11: Modelo final: Importancia de variables. Poner el grfico shap de importancia de variables. Poner algunas cosas obtenidas de ese grafico y su relacion con lo que se observaba en el EDA.
Diapo 12: Limitaciones y posibles mejoras. Completalo con las que observes.
Diapo 13: Conclusion: Hace una conclusion que este vinculada al caso de negocio. Habla un poco del modelo ganador y de como este ayuda a "solucionar" el problema de negocio.

Extra: Si realice feature engeenierng, agrega una diapo extra donde creas que debe ir y conta como fue hecho, que varibales cree y si fueron importantes o no.
