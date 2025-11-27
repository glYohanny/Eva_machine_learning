"""
Pipeline principal de aprendizaje no supervisado.
Integra clustering, reducción de dimensionalidad y detección de anomalías.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .clustering.pipeline import create_pipeline as create_clustering_pipeline
from .dimensionality_reduction.pipeline import create_pipeline as create_dimensionality_reduction_pipeline
from .anomaly_detection.pipeline import create_pipeline as create_anomaly_detection_pipeline
from .nodes import integrate_clustering_features


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline completo de aprendizaje no supervisado.
    
    Returns:
        Pipeline de Kedro integrado
    """
    # Crear sub-pipelines
    clustering_pipeline = create_clustering_pipeline()
    dim_reduction_pipeline = create_dimensionality_reduction_pipeline()
    anomaly_pipeline = create_anomaly_detection_pipeline()
    
    # Pipeline principal: clustering + reducción dimensional + anomalías
    unsupervised_pipeline = clustering_pipeline + dim_reduction_pipeline + anomaly_pipeline
    
    # Agregar integración de clustering como features
    integration_node = node(
        func=integrate_clustering_features,
        inputs=["X_train_scaled", "kmeans_labels", "X_test_scaled", "kmeans_model"],
        outputs=["X_train_with_clusters", "X_test_with_clusters"],
        name="integrate_clustering_features_node",
    )
    
    unsupervised_pipeline = unsupervised_pipeline + pipeline([integration_node])
    
    return unsupervised_pipeline

