# Final Predictiva - Churn Telco

Repositorio individual para el examen final de Analisis Predictivo. El objetivo es predecir churn en clientes Telco y generar una entrega reproducible: entrenamiento del modelo final, aplicacion sobre test, resultados y presentacion.

## Estructura

```text
EDA/
  TP1_Analisis_Predictivo_1Q2026.ipynb
  TP1_Analisis_Predictivo_1Q2026 (2).ipynb
  TP1 - Churn en clientes de Telco (3).pdf
  WA_Fn-UseC_-Telco-Customer-Churn.csv
  README_TP1.md

datos/
  raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
  processed/

notebooks/
  01_train_final_model.ipynb
  02_apply_final_model.ipynb

modelos/
  final_model.joblib
  resultados/

presentacion/
  final/

README.md
notas.md
```

## Que hay en cada carpeta

`EDA/` conserva el material del primer trabajo tal cual: notebooks exploratorias, PDF y dataset usado en ese analisis.

`datos/` contiene los datos usados por los notebooks finales. El archivo principal es:

```text
datos/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

`notebooks/` contiene los dos archivos que pide la consigna:

```text
notebooks/01_train_final_model.ipynb
notebooks/02_apply_final_model.ipynb
```

`modelos/` contiene el modelo final guardado y los resultados exportados:

```text
modelos/final_model.joblib
modelos/resultados/model_metrics.json
modelos/resultados/model_selection_cv_results.csv
modelos/resultados/hyperparameter_search_results.csv
modelos/resultados/threshold_search_results.csv
modelos/resultados/test_predictions.csv
modelos/resultados/test_predictions_from_saved_model.csv
```

`presentacion/` contiene la presentacion final y los graficos usados para armarla.

## Modelo final

Modelo seleccionado: `CatBoost`.

Metrica principal de seleccion de modelos: `PR-AUC`, adecuada para clases desbalanceadas.

Metrica para seleccion de umbral: `F2-score`, priorizando recall sin ignorar precision.

Umbral final: `0.31`.

Metricas principales en test:

```text
PR-AUC:    0.664
Recall:    90.9%
Precision: 42.4%
F2:        0.740
```

## Como reproducir

1. Ejecutar `notebooks/01_train_final_model.ipynb`.
   - Entrena modelos candidatos.
   - Selecciona el modelo final.
   - Guarda `modelos/final_model.joblib`.
   - Exporta metricas y predicciones en `modelos/resultados/`.

2. Ejecutar `notebooks/02_apply_final_model.ipynb`.
   - Carga el modelo guardado.
   - Reconstruye las features.
   - Replica las predicciones del test interno.

## Notas

Las limitaciones, mejoras posibles y conclusiones para defender el trabajo estan en:

```text
notas.md
```
