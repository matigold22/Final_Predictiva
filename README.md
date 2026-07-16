# Telco Churn - Examen Final Predictivo

Repositorio individual para desarrollar un modelo predictivo de churn en clientes de telecomunicaciones.

## Objetivo

Predecir si un cliente tiene riesgo de abandonar el servicio (`Churn`) a partir de sus caracteristicas demograficas, contractuales, de facturacion y servicios contratados.

El caso de negocio es priorizar acciones de retencion sobre clientes con mayor probabilidad de churn.

## Dataset

Dataset base: IBM Telco Customer Churn.

Archivo de entrada:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Estructura del repositorio

```text
data/
  raw/              Datos originales
  processed/        Datos procesados si hace falta generarlos
legacy/             Material del TP anterior usado como referencia
models/             Modelo final entrenado
notebooks/          Notebooks finales reproducibles
outputs/            Predicciones y metricas exportadas
presentation/
  legacy/           Presentacion anterior como insumo
  final/            Presentacion final del examen
```

## Entregables esperados

La consigna pide:

1. Presentacion final en PDF o PPT formato ITBA.
2. Notebook o script que entrene y guarde el modelo final.
3. Notebook o script que aplique el modelo final a los datos de test y replique el archivo enviado.

## Plan de desarrollo

1. Definir metrica de evaluacion y particion de datos.
2. Construir un baseline.
3. Comparar modelos candidatos.
4. Seleccionar y describir el modelo final.
5. Guardar el modelo final de forma reproducible.
6. Generar predicciones con el modelo guardado.
7. Armar la presentacion final con las secciones pedidas.

## Nota

Los archivos dentro de `legacy/` y `presentation/legacy/` son insumos del trabajo anterior. La entrega final debe quedar en `notebooks/`, `models/`, `outputs/` y `presentation/final/`.
