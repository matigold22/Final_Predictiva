# Guion de presentacion - Examen Final Predictiva

Duracion objetivo: hasta 15 minutos.

Idea general: contar una historia de negocio simple. Primero se define el problema de churn, despues se muestra que el dataset tiene senal, luego se explica como se evaluaron los modelos y finalmente se presenta CatBoost como herramienta para priorizar acciones de retencion.

## Diapositiva 1 - Portada

**Objetivo:** presentar el tema y anticipar el resultado principal.

**Contenido:**
- Titulo: Prediccion de churn en clientes Telco.
- Dataset IBM Telco Customer Churn.
- Modelo final: CatBoost.
- Metricas principales: PR-AUC, recall, precision y F2.

**Mensaje oral:**  
El trabajo busca transformar el problema de perdida de clientes en un ranking accionable de riesgo de churn.

## Diapositiva 2 - Caso de negocio

**Objetivo:** dejar claro que se resuelve un problema comercial, no solo tecnico.

**Contenido:**
- Target: `Churn = Yes`.
- Base: 7.043 clientes.
- Churners: 1.869 clientes.
- Tasa de churn: 26,5%.
- Objetivo: estimar probabilidad de churn por cliente.
- Dato de negocio: adquirir un cliente nuevo suele ser bastante mas caro que retener uno existente.

**Mensaje oral:**  
El modelo no decide automaticamente una promocion. Su funcion es priorizar clientes para que negocio pueda actuar antes de que abandonen.

## Diapositiva 3 - Dataset

**Objetivo:** explicar la estructura de datos y la limpieza realizada.

**Contenido:**
- Total de columnas originales: 21.
- Variables predictoras: 19, excluyendo `customerID` y `Churn`.
- Target: 1 variable.
- Grupos de variables:
  - Demograficas: genero, senior, pareja, dependientes.
  - Servicios: telefono, internet, seguridad, backup, proteccion, soporte, streaming.
  - Contrato: tenure y tipo de contrato.
  - Facturacion: paperless, metodo de pago, MonthlyCharges, TotalCharges.
- Limpieza: `TotalCharges` venia como texto; 11 valores vacios se imputaron como 0 porque correspondian a clientes con `tenure = 0`.

**Mensaje oral:**  
Las variables mas fuertes vienen del vinculo contractual y de los servicios contratados; las demograficas aportan menos senal.

## Diapositiva 4 - EDA

**Objetivo:** mostrar que la tasa de churn cambia segun caracteristicas observables.

**Contenido visual:**
- Grafico de tasa de churn por categoria.
- Linea de referencia con churn promedio.

**Variables a destacar:**
- Contrato mensual.
- Internet por fibra optica.
- Ausencia de OnlineSecurity.
- Ausencia de TechSupport.
- Metodo de pago electronico.

**Mensaje oral:**  
El churn no esta distribuido al azar. Hay segmentos con tasas mucho mayores a la media, lo que justifica entrenar un modelo predictivo.

## Diapositiva 5 - Evaluacion: metricas

**Objetivo:** justificar PR-AUC y F2.

**Contenido:**
- Clase positiva minoritaria: 26,5%.
- Accuracy puede ser enganosa.
- PR-AUC se usa para comparar modelos porque mide la calidad del ranking en clases desbalanceadas.
- F2 se usa para elegir el umbral porque prioriza recall, sin sacar precision de la formula.

**Mensaje oral:**  
Primero elijo el modelo que mejor ordena clientes por riesgo. Despues elijo el umbral operativo segun la prioridad de detectar churners.

## Diapositiva 6 - Evaluacion: validacion

**Objetivo:** demostrar que el procedimiento evita leakage y sobreajuste evidente.

**Contenido:**
- Train/test split 80/20 estratificado.
- Train: 5.634 clientes.
- Test: 1.409 clientes.
- Validacion cruzada de 3 folds dentro de train.
- Seed fija: 42.
- Preprocesamiento dentro de pipeline.
- Test usado una sola vez al final.

**Mensaje oral:**  
La seleccion de modelo, hiperparametros y umbral se hace sin mirar el test final. Eso permite reportar una evaluacion mas honesta.

## Diapositiva 7 - Baseline

**Objetivo:** mostrar contra que se compara el modelo final.

**Contenido:**
- Dummy classifier como piso sin informacion.
- Regresion logistica como baseline real.
- La logistica usa el mismo pipeline y las mismas variables que el resto.
- Numero de variables predictoras originales: 19, mas features derivadas generadas de forma reproducible.
- PR-AUC de Dummy: aproximadamente 0,265.
- PR-AUC de Logistic Regression: aproximadamente 0,663.

**Mensaje oral:**  
La regresion logistica es importante porque es simple, interpretable y deja una vara alta. Si un modelo complejo no la supera, no se justifica.

## Diapositiva 8 - Seleccion de modelos

**Objetivo:** comparar varias alternativas de forma clara.

**Contenido visual:**
- Grafico horizontal con PR-AUC en eje X y modelos en eje Y.
- Incluir 10 modelos:
  - Dummy.
  - Logistic Regression.
  - Decision Tree.
  - Random Forest.
  - Gradient Boosting.
  - Extra Trees.
  - AdaBoost.
  - Gaussian Naive Bayes.
  - LightGBM.
  - CatBoost.

**Conclusiones a mostrar:**
- CatBoost queda primero por PR-AUC.
- La diferencia con Gradient Boosting y Logistic Regression es chica.
- Gradient Boosting logra buena precision pero menor recall.
- LightGBM fue probado, pero en esta grilla no supero a CatBoost.

**Mensaje oral:**  
La seleccion no se basa en el nombre del algoritmo, sino en comparar todos bajo la misma metrica y validacion.

## Diapositiva 9 - Modelo final

**Objetivo:** presentar performance final y umbral operativo.

**Contenido:**
- Modelo final: CatBoost.
- Umbral: 0,31.
- PR-AUC test: 0,664.
- Recall test: 90,9%.
- Precision test: 42,4%.
- F2 test: 0,740.
- Porcentaje accionado: 56,8% del test.

**Contenido visual:**
- Curva precision/recall/F2 por umbral.
- Matriz de confusion en test.

**Mensaje oral:**  
El umbral elegido captura la mayoria de los churners, aceptando falsos positivos. En un negocio real, ese corte se ajustaria con costos y capacidad comercial.

## Diapositiva 10 - Interpretabilidad

**Objetivo:** conectar el modelo con el EDA y hacerlo defendible.

**Contenido visual:**
- Permutation importance.
- SHAP global.

**Variables a destacar:**
- Contrato mensual.
- Tipo de contrato.
- Tenure.
- Tenure group.
- TotalCharges.
- OnlineSecurity.
- TechSupport.
- Fibra optica.
- Metodo de pago.

**Mensaje oral:**  
Las variables importantes coinciden con lo visto en EDA. Esto da coherencia al modelo, aunque no implica causalidad.

## Diapositiva 11 - Limitaciones y posibles mejoras

**Objetivo:** mostrar criterio critico y realismo.

**Limitaciones:**
- No hay dimension temporal real.
- No hay costos reales de contacto, oferta o perdida de cliente.
- Precision moderada: muchas acciones no corresponderian a churn real.
- Algunas variables originales y derivadas son redundantes.
- No hay datos de reclamos, satisfaccion, competencia o uso real del servicio.

**Posibles mejoras:**
- Validar con meses futuros.
- Elegir umbral por valor esperado economico.
- Agregar NPS, tickets, reclamos, promociones y cambios de plan.
- Calibrar probabilidades.
- Probar uplift modeling para contactar clientes persuadibles.

**Mensaje oral:**  
La mejora mas importante no seria solamente probar otro algoritmo, sino incorporar informacion temporal y costos reales para tomar mejores decisiones.

## Diapositiva 12 - Conclusiones

**Objetivo:** cerrar con una conclusion orientada a negocio.

**Contenido:**
- Mejor modelo: CatBoost.
- PR-AUC test: 0,664.
- Recall: 90,9%.
- Precision: 42,4%.
- El modelo genera un ranking de riesgo de churn.
- Accion sugerida: priorizar clientes de alto score para acciones de retencion.

**Mensaje oral:**  
El modelo permite pasar de una cartera completa a una lista priorizada de clientes con mayor riesgo. Antes de llevarlo a produccion, habria que definir costo de contacto, valor de cliente y capacidad del equipo comercial.

## Cierre oral sugerido

En resumen, el trabajo muestra que existe senal predictiva clara en variables contractuales, de antiguedad, servicios y facturacion. CatBoost fue el mejor modelo bajo PR-AUC, y con un umbral optimizado por F2 logra alto recall. La utilidad principal no es reemplazar la decision comercial, sino ordenar la cartera para enfocar acciones de retencion donde tienen mas probabilidad de ser necesarias.
