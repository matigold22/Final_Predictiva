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

scripts/
  train_final_model.py
  apply_final_model.py

modelos/
  final_model.joblib
  final_model_metadata.json
  resultados/

presentacion/
  final/

README.md
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

`scripts/` agrega versiones minimas y reproducibles que cumplen de forma
estricta con la consigna, sin quitar las notebooks completas de analisis:

```text
scripts/train_final_model.py
scripts/apply_final_model.py
```

Las versiones exactas de las dependencias estan fijadas en `requirements.txt`.

`modelos/` contiene el modelo final guardado y los resultados exportados:

```text
modelos/final_model.joblib
modelos/final_model_metadata.json
modelos/resultados/model_metrics.json
modelos/resultados/model_selection_cv_results.csv
modelos/resultados/hyperparameter_search_results.csv
modelos/resultados/threshold_search_results.csv
modelos/resultados/test_predictions.csv
modelos/resultados/test_predictions_from_saved_model.csv
```

`presentacion/` contiene la presentacion final y los graficos usados para armarla:

```text
presentacion/final/Presentacion_Final_Churn_Redesign.pptx
presentacion/final/Presentacion_Final_Churn_Redesign.pdf
```

## Modelo final

Modelo seleccionado: `CatBoost`.

Metrica principal de seleccion de modelos: `PR-AUC`, adecuada para clases desbalanceadas.

Validacion de modelos: validacion cruzada estratificada de `5 folds` sobre train.

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

Opcion minima recomendada para la entrega:

```text
pip install -r requirements.txt
python scripts/train_final_model.py
python scripts/apply_final_model.py
```

El primer script entrena exclusivamente el CatBoost final con los
hiperparametros seleccionados, determina el umbral F2 dentro de train y guarda
el modelo y su metadata. El segundo carga esos artefactos y genera el archivo de
predicciones del test reproducible.

Opcion completa de analisis:

1. Ejecutar `notebooks/01_train_final_model.ipynb`.
   - Entrena modelos candidatos.
   - Selecciona el modelo final.
   - Guarda `modelos/final_model.joblib`.
   - Exporta metricas y predicciones en `modelos/resultados/`.

2. Ejecutar `notebooks/02_apply_final_model.ipynb`.
   - Carga el modelo guardado.
   - Reconstruye las features.
   - Replica las predicciones del test interno.
