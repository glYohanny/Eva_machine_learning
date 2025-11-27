"""
Pipeline de clustering.
Implementa K-Means, DBSCAN, Hierarchical Clustering y GMM.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    find_optimal_k_elbow,
    apply_kmeans_clustering,
    apply_dbscan_clustering,
    apply_hierarchical_clustering,
    apply_gmm_clustering,
    analyze_cluster_patterns,
    compare_clustering_algorithms,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de clustering con múltiples algoritmos.
    
    Returns:
        Pipeline de Kedro con clustering completo
    """
    return pipeline(
        [
            # Método del codo para encontrar K óptimo
            node(
                func=find_optimal_k_elbow,
                inputs=["X_train_scaled", "params:unsupervised_learning.clustering"],
                outputs="elbow_method_results",
                name="find_optimal_k_node",
            ),
            # K-Means Clustering
            node(
                func=apply_kmeans_clustering,
                inputs=["X_train_scaled", "params:unsupervised_learning.clustering.n_clusters", "params:model_options.random_state"],
                outputs=["kmeans_model", "kmeans_labels", "kmeans_metrics"],
                name="kmeans_clustering_node",
            ),
            # DBSCAN Clustering
            node(
                func=apply_dbscan_clustering,
                inputs=["X_train_scaled", "params:unsupervised_learning.clustering.dbscan_eps", "params:unsupervised_learning.clustering.dbscan_min_samples"],
                outputs=["dbscan_model", "dbscan_labels", "dbscan_metrics"],
                name="dbscan_clustering_node",
            ),
            # Hierarchical Clustering
            node(
                func=apply_hierarchical_clustering,
                inputs=["X_train_scaled", "params:unsupervised_learning.clustering.n_clusters", "params:unsupervised_learning.clustering.linkage_method"],
                outputs=["hierarchical_model", "hierarchical_labels", "hierarchical_metrics", "hierarchical_linkage_matrix"],
                name="hierarchical_clustering_node",
            ),
            # Gaussian Mixture Model (opcional pero recomendado)
            node(
                func=apply_gmm_clustering,
                inputs=["X_train_scaled", "params:unsupervised_learning.clustering.n_clusters", "params:model_options.random_state"],
                outputs=["gmm_model", "gmm_labels", "gmm_metrics"],
                name="gmm_clustering_node",
            ),
            # Análisis de patrones por cluster (usando K-Means como referencia)
            node(
                func=analyze_cluster_patterns,
                inputs=["X_train_scaled", "kmeans_labels", "params:model_options.feature_columns"],
                outputs="cluster_patterns_analysis",
                name="analyze_cluster_patterns_node",
            ),
            # Comparación de algoritmos
            node(
                func=compare_clustering_algorithms,
                inputs=["kmeans_metrics", "dbscan_metrics", "hierarchical_metrics", "gmm_metrics"],
                outputs="clustering_comparison_table",
                name="compare_clustering_node",
            ),
        ]
    )

