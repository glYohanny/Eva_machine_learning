"""
Pipeline de detección de anomalías.
Implementa Isolation Forest y LOF.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    apply_isolation_forest,
    apply_lof,
    analyze_anomalies,
    compare_anomaly_detection,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de detección de anomalías.
    
    Returns:
        Pipeline de Kedro con detección de anomalías completa
    """
    return pipeline(
        [
            # Isolation Forest
            node(
                func=apply_isolation_forest,
                inputs=["X_train_scaled", "params:unsupervised_learning.anomaly_detection.contamination", "params:model_options.random_state"],
                outputs=["isolation_forest_model", "isolation_forest_labels", "isolation_forest_metrics"],
                name="isolation_forest_node",
            ),
            # Local Outlier Factor
            node(
                func=apply_lof,
                inputs=["X_train_scaled", "params:unsupervised_learning.anomaly_detection.lof_n_neighbors", "params:unsupervised_learning.anomaly_detection.contamination"],
                outputs=["lof_model", "lof_labels", "lof_metrics"],
                name="lof_node",
            ),
            # Análisis de anomalías
            node(
                func=analyze_anomalies,
                inputs=["X_train_scaled", "isolation_forest_labels", "lof_labels", "params:model_options.feature_columns"],
                outputs="anomaly_analysis",
                name="analyze_anomalies_node",
            ),
            # Comparación de métodos
            node(
                func=compare_anomaly_detection,
                inputs=["isolation_forest_metrics", "lof_metrics"],
                outputs="anomaly_detection_comparison",
                name="compare_anomaly_detection_node",
            ),
        ]
    )

