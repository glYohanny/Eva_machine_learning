"""
Pipeline de reducción de dimensionalidad.
Implementa PCA, t-SNE y UMAP.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    apply_pca,
    apply_tsne,
    apply_umap,
    analyze_pca_components,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de reducción de dimensionalidad.
    
    Returns:
        Pipeline de Kedro con reducción dimensional completa
    """
    nodes_list = [
        # PCA
        node(
            func=apply_pca,
            inputs=["X_train_scaled", "params:unsupervised_learning.dimensionality_reduction.pca_n_components", "params:model_options.random_state"],
            outputs=["pca_model", "X_pca", "pca_metrics"],
            name="pca_node",
        ),
        # Análisis de componentes principales
        node(
            func=analyze_pca_components,
            inputs=["pca_model", "params:model_options.feature_columns", "params:unsupervised_learning.dimensionality_reduction.pca_n_components"],
            outputs="pca_loadings_analysis",
            name="analyze_pca_components_node",
        ),
    ]
    
    # t-SNE (puede ser opcional si es muy lento)
    try:
        nodes_list.append(
            node(
                func=apply_tsne,
                inputs=["X_train_scaled", "params:unsupervised_learning.dimensionality_reduction.tsne_n_components", "params:unsupervised_learning.dimensionality_reduction.tsne_perplexity", "params:model_options.random_state", "params:unsupervised_learning.dimensionality_reduction.tsne_n_iter"],
                outputs=["tsne_model", "X_tsne", "tsne_metrics"],
                name="tsne_node",
            )
        )
    except:
        pass
    
    # UMAP (si está disponible)
    try:
        import umap
        nodes_list.append(
            node(
                func=apply_umap,
                inputs=["X_train_scaled", "params:unsupervised_learning.dimensionality_reduction.umap_n_components", "params:unsupervised_learning.dimensionality_reduction.umap_n_neighbors", "params:unsupervised_learning.dimensionality_reduction.umap_min_dist", "params:model_options.random_state"],
                outputs=["umap_model", "X_umap", "umap_metrics"],
                name="umap_node",
            )
        )
    except ImportError:
        pass
    
    return pipeline(nodes_list)

