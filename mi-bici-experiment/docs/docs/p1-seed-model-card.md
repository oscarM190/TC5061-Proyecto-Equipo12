# Ficha del modelo P1 de demanda de MiBici

## Detalles del modelo

| Campo | Valor |
| --- | --- |
| Nombre | `mibici-equipo12-p1-demanda` |
| Arquitectura | Pipeline de scikit-learn con preprocesamiento y un regresor LightGBM (`LGBMRegressor`) de árboles potenciados por gradiente. |
| Autoría | Equipo 12 del curso TC5061 Operaciones de Aprendizaje Automático, Tec de Monterrey. |
| Fecha de desarrollo | LightGBM se incorporó a P1 el 23 de octubre de 2026. La corrida documentada se ejecutó el 14 de noviembre de 2026. |
| Versión | `v1` en el registro local de MLflow, registrada originalmente con el alias `champion`. El alias puede apuntar a versiones posteriores. |
| Experimento y corrida | `mibici-equipo12`, corrida `P1-lightgbm-personalizada` (`8a967659f0674c4dbf3539d7f11e83b3`). |

Esta ficha documenta el modelo original de P1. Docker Compose usa por defecto la versión que tenga el alias `champion`. Si el alias falta, `model-seed` entrena una variante de demostración con los mismos hiperparámetros y una muestra de 500 filas de septiembre de 2026. La registra con la etiqueta `origen=seed-demo` en el experimento `mibici-equipo12-serving`, corrida `P1-seed-demo`. [Configuración de Docker Compose](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/docker-compose.yml#L25) · [Script de arranque](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/mi-bici-experiment/scripts/seed_model.py#L58) · [CSV de demostración](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/serving/seed/p1_muestra.csv)

## Uso previsto

Estimar cuántos viajes saldrán de una estación de MiBici en un día, a partir del historial reciente de esa estación y de variables de calendario. El modelo original está dirigido al operador de MiBici. La variante de demostración permite al equipo del proyecto probar la integración entre la API, el registro de MLflow y el trabajador asíncrono. [Descripción del proyecto](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/README.md#preguntas-del-proyecto)

## Datos de entrenamiento

El conjunto histórico de P1 abarca del **1 de octubre de 2025 al 30 de septiembre de 2026**. Se construyó con los registros abiertos de viajes de MiBici en el Área Metropolitana de Guadalajara. Cada fila representa una estación en un día. La variable objetivo `trips` es el número de salidas registradas ese día. [Ventana de datos de P1](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/mi-bici-experiment/mi_bici_experiment/config.py#L24) · [Portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/)

La preparación descarta registros sin estación o con fecha inválida, agrega las salidas por estación y día y completa con ceros los días sin viajes a partir de la primera actividad de cada estación. Calcula los rezagos de uno y siete días y la media de los siete días previos. Después de descartar las filas con historial insuficiente, el ajuste de la `v1` usa las primeras **92.345 observaciones** de 389 estaciones, del **8 de octubre de 2025 al 14 de junio de 2026**. [Construcción de variables](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/mi-bici-experiment/mi_bici_experiment/features.py#L16) · [Preparación y partición de los datos](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/mi-bici-experiment/mi_bici_experiment/modeling/modules/p1_regression/models.py#L86)

## Datos de evaluación

La partición original es cronológica y se hace por fecha, con proporciones aproximadas de 70 % para ajuste, 15 % para validación y 15 % para prueba. La validación cubre del 15 de junio al 7 de agosto de 2026 (22.883 observaciones) y la prueba del 8 de agosto al 30 de septiembre de 2026 (25.175 observaciones). Los rezagos se calculan con las salidas observadas antes de cada día. [Partición cronológica](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/mi-bici-experiment/mi_bici_experiment/modeling/modules/p1_regression/models.py#L93)

La comparación posterior de versiones usó 14.509 observaciones de 469 estaciones en octubre de 2026. El archivo generado es `mi-bici-experiment/data/processed/p1_2026_10.parquet` y usa los últimos días de septiembre como historial para construir los primeros rezagos de octubre. [Preparación del conjunto de octubre](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/mi-bici-experiment/mi_bici_experiment/modeling/monitor.py#L142) · [Instrucciones de evaluación](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/README.md#ab-testing-de-p1)

## Métricas de desempeño

| Métrica de la `v1` | Validación | Prueba | Evaluación posterior en octubre de 2026 |
| --- | ---: | ---: | ---: |
| Error absoluto medio (MAE) | 5,84 | 6,27 | 8,01 |
| Raíz del error cuadrático medio (RMSE) | 10,09 | 11,44 | 15,80 |

Todas las cifras se expresan en viajes por estación y día. Los resultados de validación y prueba se tomaron de la corrida original en MLflow. Los de octubre proceden de la comparación posterior contra el `challenger`. En una evaluación local separada sobre el mismo conjunto de octubre, la variante de demostración obtuvo un MAE de 12,03 y un RMSE de 28,42. Las evaluaciones aplican el mismo límite inferior de cero que la API. [Evaluación de P1](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/mi-bici-experiment/mi_bici_experiment/modeling/modules/p1_regression/models.py#L124) · [Regla de predicción de la API](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/mi-bici-experiment/mi_bici_experiment/api.py#L113)

## Configuración del entrenamiento

| Parámetro | Valor |
| --- | ---: |
| Configuración | Hiperparámetros de `modelo_3` |
| Número de árboles (`n_estimators`) | 400 |
| Tasa de aprendizaje (`learning_rate`) | 0,05 |
| Máximo de hojas por árbol (`num_leaves`) | 31 |
| Semilla (`random_state`) | 42 |

El pipeline ajusta `StandardScaler` sobre los tres indicadores de historial y `is_weekend`. Usa `OneHotEncoder(handle_unknown="ignore")` para `origen_id`, `dayofweek` y `month`. La corrida original aplica los hiperparámetros indicados al conjunto de ajuste y conserva los periodos de validación y prueba para medir el error. Cuando se comparan varias configuraciones, el flujo de P1 selecciona la de menor MAE de validación. [Hiperparámetros de P1](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/mi-bici-experiment/mi_bici_experiment/config.py#L24) · [Pipeline de regresión](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/mi-bici-experiment/mi_bici_experiment/modeling/modules/p1_regression/models.py#L104) · [Comandos para reproducir P1](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/README.md#comandos-para-reproducir-p1)

El artefacto de la `v1` registra Python 3.14.4, LightGBM 4.7.0, scikit-learn 1.9.1 y MLflow 3.17.2.

## Limitaciones

- Los cambios estacionales, los festivos, los eventos especiales y los cierres pueden alterar los patrones de viajes. La variante de demostración tiene una cobertura especialmente limitada porque se ajusta con una muestra de un solo mes.
- El modelo depende de siete días de historial. Su cobertura es limitada en estaciones nuevas o con datos recientes incompletos. El codificador admite estaciones y meses desconocidos, pero estas categorías carecen de un efecto aprendido propio.
- Los rezagos observados respaldan el pronóstico diario. Pronosticar varios días seguidos requeriría obtener o estimar los rezagos futuros y evaluar ese escenario por separado.

## Consideraciones éticas

- Las salidas observadas pueden subestimar la demanda de las estaciones con escasez de bicicletas o interrupciones del servicio. Usar esas predicciones como único criterio de redistribución podría perpetuar diferencias de disponibilidad entre zonas.
- Los errores publicados son globales. Un análisis por zona, volumen de viajes y antigüedad de la estación ayudaría a detectar diferencias de desempeño antes de usar el modelo en decisiones operativas.
- Los datos se agregan por estación y día. El pipeline utiliza conteos, identificadores de estación y variables de calendario, sin incorporar identificadores ni atributos demográficos de las personas usuarias.

## Requisitos operativos

La API recibe un objeto JSON por estación y día en `POST /predict`. El servicio asíncrono usa los mismos campos en `POST /predict/async` y requiere Redis y un trabajador RQ.

| Campo | Tipo y restricción | Significado |
| --- | --- | --- |
| `trips_lag_1` | Número ≥ 0 | Viajes del día anterior. |
| `trips_lag_7` | Número ≥ 0 | Viajes de hace siete días. |
| `trips_roll_mean_7` | Número ≥ 0 | Media diaria de los siete días anteriores. |
| `is_weekend` | `0` o `1` | `1` para sábado o domingo. |
| `origen_id` | Entero > 0 | Identificador de la estación de origen. |
| `dayofweek` | Entero de `0` a `6` | Día de la semana, de lunes (`0`) a domingo (`6`). |
| `month` | Entero de `1` a `12` | Mes del día pronosticado. |

La API rechaza campos adicionales. La respuesta síncrona contiene `prediction`, una estimación decimal no negativa, y `model`, la URI del modelo. [Esquema de entrada](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/mi-bici-experiment/mi_bici_experiment/api.py#L48) · [Endpoints de predicción](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/mi-bici-experiment/mi_bici_experiment/api.py#L138)

El despliegue requiere Docker con Compose v2. La imagen de la API usa Python 3.14, LightGBM con `libgomp1`, scikit-learn, pandas, MLflow y FastAPI. MLflow guarda el pipeline con el formato `skops` y conserva el registro y los artefactos en el volumen `mlflow-store` por defecto. La API tiene un límite predeterminado de 2 CPU y `1g` de memoria. Los puertos predeterminados son 8000 para la API y 5000 para MLflow. [Imagen de la API](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/Containerfile) · [Dependencias de Python](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/mi-bici-experiment/requirements.txt) · [Configuración de servicios](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/docker-compose.yml)

## Mantenimiento

- Revisar mensualmente la cobertura de estaciones, la calidad de las variables de entrada y los cambios en su distribución. Cuando estén disponibles las salidas observadas, calcular MAE y RMSE y desglosarlos por estación o zona, periodo y volumen de viajes.
- Si se detecta deterioro sostenido o cambia la red de estaciones, entrenar un candidato con más meses de historial y usar un periodo posterior para evaluar su desempeño. Registrar la versión y sus métricas en MLflow con el alias `challenger`.
- Comparar el candidato y el `champion` en las mismas filas antes de promoverlo. Tras cambiar el alias, reiniciar la API y el trabajador para que carguen la versión promovida. [Monitoreo y comparación de P1](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/README.md#ab-testing-de-p1) · [Arranque con Docker Compose](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/blob/73689dbca20174bd3d370cf0179df893732b1aff/README.md#levantar-el-sistema)
