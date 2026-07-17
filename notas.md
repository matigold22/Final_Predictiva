# Notas para presentacion

## Limitaciones y posibles mejoras

El modelo fue desarrollado con una particion interna del dataset original, ya que no se cuenta con un conjunto de test externo provisto por la consigna. Esto permite evaluar de manera reproducible, pero una mejora importante seria validar el desempeno con datos de otro periodo o con una muestra realmente futura para medir estabilidad temporal.

La variable objetivo `Churn` esta desbalanceada: la mayoria de los clientes no abandona el servicio. Por eso se uso PR-AUC para seleccionar modelos, ya que evalua mejor la capacidad de ordenar casos positivos cuando hay desbalance. Luego se optimizo el umbral con F2-score, una metrica que da mas peso al recall sin ignorar la precision. Esta decision es razonable para un problema de retencion, porque suele ser mas costoso no detectar a un cliente que se va que contactar a un cliente que finalmente no se iba. De todos modos, el umbral elegido podria ajustarse mejor si la empresa definiera costos reales de falsos positivos y falsos negativos.

El feature engineering agrega variables utiles para el negocio, como antiguedad agrupada, indicadores de fibra optica, contrato mensual, servicios contratados y relacion entre gasto mensual y gasto historico. Sin embargo, algunas variables nuevas son redundantes con variables originales. Por ejemplo, `MonthlyCharges`, `avg_charge_per_tenure`, `has_fiber_optic` y conteos de servicios estan correlacionados. Esto no genera leakage, pero puede introducir ruido interpretativo: la importancia de una variable puede repartirse entre varias variables similares.

No se detecto leakage directo: `customerID` no se usa como variable predictora, no hay clientes repetidos entre train y test, el preprocesamiento esta dentro de un pipeline y el umbral se elige en una validacion interna, no sobre el test final. Con CatBoost la brecha train-test es baja a moderada, por lo que no aparecen senales fuertes de sobreajuste en esta evaluacion. De todos modos, una mejora posible seria validar el modelo con datos futuros para confirmar estabilidad temporal.

Como posibles mejoras, con mas tiempo o recursos se podria:

- Validar el modelo con datos de meses posteriores.
- Ajustar el umbral usando una matriz de costos de negocio.
- Simplificar variables redundantes para mejorar interpretabilidad.
- Probar calibracion de probabilidades para usar el score como ranking comercial.
- Incorporar datos adicionales, como reclamos, contactos con soporte, cambios de plan, promociones recibidas o historial de pagos.
- Medir el impacto economico esperado de distintas estrategias de retencion.

## Conclusiones

El analisis permite transformar un problema de negocio, la perdida de clientes, en un modelo predictivo reproducible. Se parte del dataset Telco Churn, se definen variables relevantes, se construyen features orientadas al negocio y se comparan modelos contra un baseline.

La metrica principal para comparar modelos fue PR-AUC porque el objetivo es ordenar correctamente a los clientes con mayor riesgo dentro de un problema desbalanceado. Se probaron modelos lineales, arboles, Random Forest, Gradient Boosting, LightGBM y CatBoost, todos con validacion cruzada y busqueda de hiperparametros acotada. Bajo este criterio, el modelo final seleccionado fue CatBoost. Luego se optimizo el umbral con F2-score y se obtuvo un umbral de 0.22. Esta configuracion logra un recall alto, detectando la mayor parte de los clientes que efectivamente abandonan, aunque acepta una mayor cantidad de falsos positivos.

Las variables mas relevantes para el modelo estan asociadas al tipo de contrato, contrato mensual, antiguedad del cliente, cargos acumulados, fibra optica, seguridad online y soporte tecnico. Estas variables no deben interpretarse como causas directas del churn, sino como senales predictivas que ayudan a identificar perfiles de riesgo.

En terminos de negocio, el modelo puede servir como herramienta de priorizacion: permite ordenar clientes segun probabilidad de abandono y enfocar acciones de retencion sobre los casos con mayor riesgo. La principal recomendacion es usarlo como apoyo a la decision, complementandolo con conocimiento comercial y validacion futura antes de llevarlo a produccion.
