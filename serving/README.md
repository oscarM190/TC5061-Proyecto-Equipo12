# API de MiBici: P3

Servicio de inferencia del modelo de agrupación de estaciones. Conserva el repositorio, modelos,
pruebas y documentación de las actividades anteriores. Motor principal: **Docker**;
el mismo `Containerfile` puede utilizarse con Podman.

## Modelo y estrategia de despliegue

Se usa **el modelo existente del Model Registry**, no un modelo entrenado al iniciar la API:

- Nombre: `mibici-equipo12-p3-grupos`.
- Versión exportada en esta copia local: `1`, alias `champion` en el Registry consultado.
- Corrida de origen: `3c32c7b78cb64a4eb4eedce691c3f015`.
- Pipeline: StandardScaler → PCA(2) → K-Means de 3 grupos.

Elegimos **descargar el artefacto del Registry y empaquetarlo en la imagen**. `MLFLOW_TRACKING_URI`
se utiliza durante la exportación, en el host. Durante la ejecución del contenedor se usa
`MODEL_PATH=/app/artifacts/p3`; no hacen falta red, credenciales ni acceso al Registry.
El script resuelve un alias a una versión numérica antes de descargar y registra nombre,
versión, `run_id`, URI y hashes SHA-256 en `deployment.json`. El arranque comprueba los hashes,
la firma y el linaje. Los hashes detectan cambios respecto del manifiesto; no son una firma
criptográfica de un proveedor externo.

La inferencia es síncrona: una petición HTTP contiene de 1 a 1000 perfiles agregados y devuelve
sus grupos en el mismo orden. Esto sirve a la planeación de estaciones y permite pequeños lotes.
La API no recibe viajes crudos ni estima el inventario instantáneo de bicicletas.
El escalado y la reducción de dimensión forman parte del modelo; el cliente no debe repetirlos.

## 1. Preparar el entorno

Ejecuta todos los comandos de esta guía desde la **raíz del repositorio**.
Requisitos: Python 3.14.7, Docker con servicio disponible (o Podman), y acceso al Registry y sus
artefactos para la descarga inicial. Para construir por primera vez se necesita acceso al
registro de imágenes y a los paquetes Python.

```bash
python3.14 -m venv .venv-api
source .venv-api/bin/activate
python -m pip install -r requirements-api-dev.txt
```

PowerShell: `py -3.14 -m venv .venv-api` y `.\.venv-api\Scripts\Activate.ps1`.
Este entorno es independiente del entrenamiento y de Deepchecks.
`requirements-api.txt` fija las dependencias directas del servicio. El `Containerfile` fija
la versión de Python. Las dependencias transitivas y el digest de la imagen base todavía no
están bloqueados; para una entrega posterior con reconstrucción binaria exacta, generar un
lock y fijar el digest después de validar el build.

## 2. Recuperar el modelo del Registry

Con el servidor de MLflow disponible en el host:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m serving.export_model --name mibici-equipo12-p3-grupos --version 1
```

PowerShell: `$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'` y el mismo comando Python.
También puede configurarse una URI SQLite absoluta si el Registry es local y los artefactos
siguen disponibles en sus ubicaciones registradas. El servidor debe apuntar a la base existente;
arrancar otro servidor con una base vacía no recupera los modelos anteriores.

Alternativa por alias (no combinar con `--version`):

```bash
python -m serving.export_model --name mibici-equipo12-p3-grupos --alias champion
```

Esto crea `artifacts/p3/`. Si ya existe, el script se detiene para no sustituir una versión
silenciosamente. Para una nueva versión, exporta primero con `--output artifacts/p3-candidato`,
comprueba el manifiesto y reemplaza de forma deliberada la carpeta usada por el build.
No se almacenan credenciales ni la URI del servidor en el manifiesto de despliegue.
Los artefactos se excluyen de Git: cada integrante debe descargarlos antes de construir.
No deben agregarse modelos serializados al historial para resolver un problema de acceso.

En esta carpeta local ya está descargada la versión 1. La exportación se comprobó usando
una copia temporal de la base del Registry existente, sin cambiar sus aliases ni sus corridas.

## 3. Construir y ejecutar

Docker:

```bash
docker build -f Containerfile -t mibici-p3:1 .
docker run --rm --name mibici-p3 \
  --read-only --tmpfs /tmp:rw,nosuid,size=128m \
  --cap-drop ALL --security-opt no-new-privileges \
  -p 127.0.0.1:8000:8000 mibici-p3:1
```

Podman:

```bash
podman build -f Containerfile -t localhost/mibici-p3:1 .
podman run --rm --name mibici-p3 \
  --read-only --tmpfs /tmp:rw,nosuid,size=128m \
  --cap-drop ALL --security-opt no-new-privileges \
  -p 127.0.0.1:8000:8000 localhost/mibici-p3:1
```

En PowerShell coloca cada comando en una sola línea o usa el acento grave como continuación.
El puerto queda disponible solo en el host. El proceso del contenedor usa UID 10001 y no requiere
privilegios. `.dockerignore` limita el contexto a la API, dependencias y modelo; excluye datos,
notebooks, bases de MLflow, `.git` y entornos virtuales.
El contenedor carga el modelo una sola vez y falla al iniciar si faltan archivos o la firma es
incompatible. No sustituye el modelo por predicciones ficticias.

## 4. Consumir la API

```bash
curl --fail http://127.0.0.1:8000/healthz
curl --fail http://127.0.0.1:8000/model
curl --fail http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/p3-request.json
```

En Windows utiliza `curl.exe` para evitar el alias de PowerShell.
También puedes abrir `http://127.0.0.1:8000/docs`, desplegar `POST /predict`, pulsar **Try it out**,
pegar el JSON de `examples/p3-request.json` y pulsar **Execute**.

Petición de un perfil:

```json
{"instances":[{"prop_fin_semana":0.207,"log_viajes_dia":2.4248027257,"balance":-0.122}]}
```

La respuesta tiene esta estructura (las etiquetas dependen de la versión del modelo):

```json
{
  "model": {
    "name": "mibici-equipo12-p3-grupos",
    "version": "1",
    "run_id": "3c32c7b78cb64a4eb4eedce691c3f015",
    "uri": "models:/mibici-equipo12-p3-grupos/1"
  },
  "predictions": [0]
}
```

Consulta [`examples/p3-response.json`](../examples/p3-response.json) para la salida comprobada
con los tres perfiles del archivo de petición.

| Campo | Tipo y rango | Construcción |
|---|---|---|
| `prop_fin_semana` | Número finito entre 0 y 1 | Salidas en sábado/domingo divididas entre todas las salidas del periodo |
| `log_viajes_dia` | Número finito ≥ 0 | `log1p(salidas / días)`; los días se calculan según `features.resumen_uso` |
| `balance` | Número finito entre −1 y 1 | `(llegadas − salidas) / (llegadas + salidas)` |

Usa el mismo periodo y las mismas reglas de agregación de `perfil_estaciones` para construir perfiles.
El entrenamiento aplica un mínimo de 300 salidas; la API recibe las tres variables finales y no puede
comprobar ese conteo. Los números de grupo son identificadores del modelo, no categorías ordinales;
pueden cambiar al registrar otro entrenamiento. Las descripciones urbanas son interpretaciones,
no etiquetas de referencia verificadas por la API.

Se rechazan con HTTP 422 campos faltantes, desconocidos, texto en lugar de números, booleanos,
nulos, NaN/infinito, rangos incorrectos y lotes vacíos o mayores a 1000 perfiles. Por ejemplo:

```bash
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' -d '{"instances":[]}'
```

Los errores internos de inferencia devuelven HTTP 500 con un mensaje genérico; los detalles quedan
en los logs. `/healthz` solo responde cuando terminó la carga del modelo.

## 5. Pruebas y evidencia

```bash
python -m pytest serving/tests -v
```

Las pruebas cubren el contrato, entradas inválidas, salida del modelo, carga única, manifiestos
alterados, firma, linaje y resolución de alias. La prueba de integración utiliza el modelo real
si existe `artifacts/p3`; en CI se omite explícitamente cuando el Registry no está disponible.
Se conserva el workflow anterior y se añade `.github/workflows/api.yml` para estas pruebas.
Ningún workflow declara un build del modelo real si no dispone del artefacto del Registry.

Para construir, ejecutar y probar **un contenedor real**, con limpieza automática al terminar:

```bash
bash scripts/test_container.sh
# O con Podman:
CONTAINER_ENGINE=podman IMAGE_NAME=localhost/mibici-p3:1 bash scripts/test_container.sh
```

El script guarda build, ID del contenedor, versiones instaladas, estado del servicio, petición,
respuesta, error 422 y logs en `evidence/container/`. Además compara las predicciones HTTP con
las del artefacto dentro del contenedor. Sale con error si el motor, el build o una verificación fallan.

Para añadir la **captura real de Swagger UI respondiendo desde el contenedor**:

```bash
npm install --prefix tools/evidence playwright
cd tools/evidence
npx playwright install chromium
cd ../..
CAPTURE_SCREENSHOT=1 bash scripts/test_container.sh
```

La captura quedará en `evidence/container/api-container.png`. El navegador necesita acceso a los
recursos de Swagger UI de su CDN. Como alternativa, ejecuta el contenedor manualmente, envía la
petición desde `/docs` y guarda una captura en esa misma ruta. Incluye el código 200 y el JSON de
respuesta. Una captura de pruebas locales sin contenedor no sustituye esta evidencia.

Para detener un contenedor iniciado manualmente: `docker stop mibici-p3` o `podman stop mibici-p3`.

## Colaboración y entrega

La evaluación requiere contribuciones reales de los cuatro integrantes. Propuesta de revisión:
Jordan: integración de P3; Oscar: contrato/API; Julián: contenedor y ejecución; Eduardo: validaciones,
documentación y captura. Esta distribución es una propuesta, no evidencia de trabajo ya realizado.
Cada integrante debe hacer sus cambios o revisión y registrar su contribución con su propia identidad.

Integrar mediante PR en el repositorio existente `TC5061-Proyecto-Equipo12`, comprobar que el profesor
pueda acceder y entregar su enlace en Canvas. Esta copia local no ha sido publicada ni enviada a Canvas.
La consigna recibida usa «Actividad 8» en la rúbrica; la carpeta solicitada es `catividad 7`.

## Referencias

- [FastAPI en contenedores](https://fastapi.tiangolo.com/deployment/docker/).
- [Flujos de trabajo del Model Registry de MLflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/).
- [Construcción de imágenes con Podman](https://docs.podman.io/en/latest/markdown/podman-build.1.html).
