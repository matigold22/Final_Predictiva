# Guion de presentacion - Examen Final Predictiva

Duracion objetivo: hasta 15 minutos.

Idea general: contar una historia de negocio simple. Primero se define el problema de churn, despues se muestra que el dataset tiene senal, luego se explica como se evaluaron los modelos y finalmente se presenta CatBoost como herramienta para priorizar acciones de retencion.

Orden final de 15 diapositivas:

1. Portada: tema, materia, instancia de examen y autor.
2. Caso de negocio: problema, objetivo predictivo, tamaño de la base y churn.
3. Dataset: 19 predictoras en cuatro familias y limpieza relevante.
4. Feature engineering: variables creadas y composición de las principales.
5. EDA: asociación de categorías relevantes con la tasa de churn.
6. Evaluación: PR-AUC como métrica principal y F2 para elegir el umbral.
7. Validación: split 80/20 estratificado, validación cruzada de 5 folds y control de leakage.
8. Baseline: Dummy y regresión logística bajo el mismo pipeline.
9. Selección: diez modelos comparados por PR-AUC promedio en CV de 5 folds.
10. Modelo final: CatBoost, hiperparámetros, métricas y matriz de confusión.
11. Umbral: relación entre recall, precisión y F2; corte final en 0,31.
12. Interpretación: importancia global SHAP y relación con el EDA.
13. Lectura de negocio: segmentación por score y ejemplos de intervención.
14. Limitaciones y mejoras posibles.
15. Conclusión vinculada con el caso de negocio.
