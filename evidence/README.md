# Estado de verificación y evidencias

Fecha: 5 de noviembre de 2026. Este archivo distingue las comprobaciones realizadas y documenta
la evidencia generada en el entorno de pruebas con Docker y en pruebas locales.

**Comprobado y completado:**

- Exportación real de `models:/mibici-equipo12-p3-grupos/1` desde una copia temporal del Registry
  SQLite existente. Se mantuvieron sus artefactos originales, corrida y versión.
- Carga del pipeline real y predicciones `[0, 2, 1]` para `examples/p3-request.json`.
- 48 pruebas del servicio, incluida integración con el artefacto real. Resultado en
  [`local/api-tests.xml`](local/api-tests.xml).
- Peticiones ASGI dentro del proceso: `/healthz` 200, `/predict` 200, lote vacío 422.
  Petición y respuestas reales en [`local/api-probe.json`](local/api-probe.json).
- Ejecución conjunta de las pruebas anteriores y nuevas: ver [`local/pytest.txt`](local/pytest.txt)
  y [`local/all-tests.xml`](local/all-tests.xml).
- Manifiesto de exportación en [`local/model-provenance.json`](local/model-provenance.json)
  y versiones del entorno utilizado en [`local/versions.json`](local/versions.json).
- **Construcción y ejecución del contenedor en Docker (comprobado exitosamente):**
  - La imagen `mibici-p3:1` fue construida con éxito mediante `docker build -f Containerfile -t mibici-p3:1 .`
  - Se ejecutó el script `scripts/test_container.sh` con Docker 29.9.0. Registro completo en [`container/build.txt`](container/build.txt) y [`container/engine.txt`](container/engine.txt).
  - El contenedor respondió `200 OK` en `/healthz` ([`container/health.json`](container/health.json)) y `200 OK` en `/predict` con las predicciones `[0, 2, 1]` ([`container/response.json`](container/response.json)).
  - La validación de entrada rechazó lotes vacíos con `422 Unprocessable Content` ([`container/invalid-response.json`](container/invalid-response.json)).
  - Se verificó la coherencia de predicciones directas del modelo dentro del contenedor en [`container/verification.txt`](container/verification.txt).
  - Se generó la captura real de la API respondiendo desde el contenedor en Swagger UI: [`container/api-container.png`](container/api-container.png).

**Pendiente para completar la entrega en GitHub/Canvas:**

1. Conciliar la rama de trabajo con `origin/main` (revisar la divergencia entre la implementación de P1 existente en `main` y esta versión P3).
2. Distribuir y registrar las contribuciones de los 4 integrantes del equipo en Git mediante ramas y Pull Requests revisados.
3. Verificar que el repositorio `https://github.com/oscarM190/TC5061-Proyecto-Equipo12` sea público o accesible para el profesor.
4. Enviar el enlace del repositorio en Canvas antes del domingo 8 de noviembre de 2026 a las 23:59.
