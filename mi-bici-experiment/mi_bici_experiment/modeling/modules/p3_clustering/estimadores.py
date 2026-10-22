"""Estimadores propios de P3 que se guardan dentro del Pipeline registrado en MLflow."""

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid


class AglomerativoConCentroides(BaseEstimator, ClusterMixin):
    """Clustering aglomerativo que asigna estaciones nuevas al centroide del grupo más cercano.

    `AgglomerativeClustering` no tiene `predict`; sin él no se puede medir la estabilidad en el
    periodo posterior ni servir el modelo desde el Model Registry.
    """

    def __init__(self, n_clusters: int = 3, linkage: str = "ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X, y=None):
        modelo = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        self.labels_ = modelo.fit_predict(X)
        self.centroides_ = NearestCentroid().fit(X, self.labels_)
        return self

    def predict(self, X):
        return self.centroides_.predict(X)
