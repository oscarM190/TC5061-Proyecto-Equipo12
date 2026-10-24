"""Punto de entrada de `mlflow run` (entry points train, search y register del MLproject).

Los hiperparámetros llegan como parámetros no declarados de `mlflow run` (MLflow los agrega al
comando como `--nombre valor`), así cada pregunta recibe solo los suyos:

    mlflow run . -e train -P question=p3 -P model=kmeans -P k=3 -P pca=2

Si no se indican, la semilla y las fechas se toman de config.py.
"""

import argparse

from loguru import logger
import mlflow

from mi_bici_experiment.config import (
    MLFLOW_EXPERIMENT,
    P2_COMBINACIONES,
    P3_INICIO_ENTRENAMIENTO,
    P3_K_CANDIDATOS,
    P3_N_INIT,
    RANDOM_STATE,
)


def _argumentos_p3(extra: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p3")
    p.add_argument("--k", type=int, default=3, help="Número de grupos.")
    p.add_argument("--pca", type=int, default=0, help="Componentes de PCA (0 = sin PCA).")
    p.add_argument("--cov", default="full", help="GMM: tipo de covarianza.")
    p.add_argument("--linkage", default="ward", help="Aglomerativo: criterio de enlace.")
    p.add_argument("--n_init", type=int, default=P3_N_INIT, help="K-means: inicializaciones.")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--stage", default="candidate", choices=["baseline", "search", "candidate"])
    p.add_argument("--inicio", default=P3_INICIO_ENTRENAMIENTO, help="Inicio del entrenamiento.")
    p.add_argument("--ks", default=",".join(map(str, P3_K_CANDIDATOS)), help="search: valores de k.")
    p.add_argument("--pcas", default="0,2", help="search: componentes de PCA.")
    p.add_argument("--alias", default="champion", help="register: alias de la versión.")
    return p.parse_args(extra)


def _p3(accion: str, model: str, run_id: str, extra: list[str]) -> None:
    from mi_bici_experiment.modeling.modules.p3_clustering import models as p3

    a = _argumentos_p3(extra)
    if accion == "register":
        with mlflow.start_run(run_name=f"p3-register-{a.alias}"):
            mlflow.set_tags({"question": "p3", "stage_of_work": "registry"})
            version, modelo = p3.registrar(run_id, a.alias)
        logger.info(f"{p3.NOMBRE_REGISTRO} v{version} @{a.alias} cargado: {modelo.metadata.run_id}")
        return
    datos = p3.DatosP3(a.inicio)
    if accion == "search":
        ks = [int(x) for x in a.ks.split(",")]
        pcas = [int(x) for x in a.pcas.split(",")]
        _, k, pca = p3.buscar(datos, model, ks, pcas, a.seed)
        logger.info(f"P3 búsqueda {model}: mejor k={k}, pca={pca}")
        return
    _, metricas = p3.entrenar(datos, model, a.k, a.pca, a.cov, a.linkage, a.n_init, a.seed,
                              a.stage)
    logger.info(f"P3 {model}: {metricas}")

def _argumentos_p1(extra: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p1")
    p.add_argument("--n_estimators", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--num_leaves", type=int, default=None)
    p.add_argument("--min_child_samples", type=int, default=None)
    p.add_argument("--inicio", default=None, help="Fecha de inicio.")
    p.add_argument("--fin", default=None, help="Fecha de fin.")
    return p.parse_args(extra)


def _p1(accion: str, run_id: str, extra: list[str]) -> None:
    from mi_bici_experiment.config import FECHA_FIN, FECHA_INICIO, P1_COMBINACIONES
    from mi_bici_experiment.modeling.modules.p1_regression import models as p1

    a = _argumentos_p1(extra)
    if accion == "register":
        p1.promover_a_produccion(run_id)
        return

    inicio = a.inicio or FECHA_INICIO
    fin = a.fin or FECHA_FIN
    elegidos = {
        k: getattr(a, k)
        for k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples")
        if getattr(a, k) is not None
    }
    if accion == "train" and elegidos:
        base = next(iter(P1_COMBINACIONES.values()))
        p1.buscar_hiperparametros({"personalizada": {**base, **elegidos}}, inicio, fin)
    else: 
        p1.buscar_hiperparametros(None, inicio, fin)

def _baseline_actual(question: str, extra: list[str]) -> None:
    """Ejecuta únicamente el baseline de regresión (P1) o logístico (P2)."""
    if extra:
        logger.warning(f"{question}: se ignoran {extra}; el modelo aún no está parametrizado")
    if question == "p1":
        from mi_bici_experiment.modeling.modules.p1_regression.models import RegressionModel

        RegressionModel().ejecutar()
    elif question == "p2":
        from mi_bici_experiment.modeling.modules.p2_classification.models import ClassificationModel

        ClassificationModel.ejecutar(solo_baseline=True)


def _argumentos_p2(extra: list[str]) -> argparse.Namespace:
    """Acepta los nombres de MLflow (-P n_estimators=...) y de la CLI de P2."""
    p = argparse.ArgumentParser(prog="p2")
    for name, value_type in (
        ("n_estimators", int), ("max_depth", int), ("min_samples_split", int),
        ("min_samples_leaf", int), ("max_features", str), ("max_samples", float),
        ("class_weight", str), ("random_state", int), ("n_jobs", int),
    ):
        p.add_argument(f"--{name}", f"--{name.replace('_', '-')}", type=value_type, default=None)

    def boolean(value):
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise argparse.ArgumentTypeError("bootstrap debe ser true o false")

    group = p.add_mutually_exclusive_group()
    group.add_argument("--bootstrap", type=boolean, nargs="?", const=True, default=None)
    group.add_argument("--no-bootstrap", dest="bootstrap", action="store_false", default=None)
    return p.parse_args(extra)


def _p2(accion: str, model: str, run_id: str, extra: list[str]) -> None:
    from mi_bici_experiment.modeling.modules.p2_classification import models as p2

    if accion == "train" and model in ("logistic_regression", "regresion_logistica", "baseline"):
        _baseline_actual("p2", extra)
        return
    if model not in ("random_forest", "random-forest", "rf"):
        raise SystemExit("p2: usa --model random_forest")
    if accion not in ("train", "search"):
        raise SystemExit("p2: usa train o search; la búsqueda registra al ganador con alias champion")
    options = vars(_argumentos_p2(extra))
    if accion == "train" and all(value is None for value in options.values()):
        # Force a single training; remaining options inherit modelo_3 in the P2 CLI.
        options["n_estimators"] = P2_COMBINACIONES["modelo_3"]["n_estimators"]
    p2.main(**options)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("accion", choices=["train", "search", "register"])
    p.add_argument("--question", required=True, choices=["p1", "p2", "p3"])
    p.add_argument("--model", default="kmeans")
    p.add_argument("--run_id", default="", help="register: corrida cuyo modelo se registra.")
    args, extra = p.parse_known_args(argv)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    if args.question == "p3":
        _p3(args.accion, args.model, args.run_id, extra)
    elif args.question == "p2":
        _p2(args.accion, args.model, args.run_id, extra)
    elif args.question == "p1":
        _p1(args.accion, args.run_id, extra)
    elif args.accion == "train":
        _baseline_actual(args.question, extra)
    else:
        raise SystemExit(f"{args.question}: '{args.accion}' aún no está implementado")


if __name__ == "__main__":
    main()
