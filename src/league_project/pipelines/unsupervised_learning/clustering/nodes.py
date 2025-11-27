"""
Nodos para el pipeline de clustering.
Implementa K-Means, DBSCAN, Hierarchical Clustering con métricas completas.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import logging
import pickle

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


def find_optimal_k_elbow(
    X: pd.DataFrame,
    clustering_params: Dict,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Encuentra el K óptimo usando el método del codo (Elbow Method).
    
    Args:
        X: Features para clustering
        k_range: Rango de valores de K a probar
        random_state: Semilla para reproducibilidad
        
    Returns:
        Diccionario con K óptimo y métricas de inercia
    """
    logger.info("="*80)
    logger.info("MÉTODO DEL CODO (ELBOW METHOD) - Búsqueda de K óptimo")
    logger.info("="*80)
    
    # Obtener rango de k_range desde parámetros
    k_range_start = clustering_params.get('k_range_start', 2)
    k_range_end = clustering_params.get('k_range_end', 11)
    k_range = range(k_range_start, k_range_end)
    
    inertias = []
    k_values = list(k_range)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        logger.info(f"K={k}: Inercia = {kmeans.inertia_:.2f}")
    
    # Calcular diferencias para encontrar el "codo"
    diffs = np.diff(inertias)
    diff_diffs = np.diff(diffs)
    optimal_k_idx = np.argmax(diff_diffs) + 2  # +2 porque empezamos en k=2
    optimal_k = k_values[optimal_k_idx] if optimal_k_idx < len(k_values) else k_values[len(k_values)//2]
    
    logger.info(f"[OK] K optimo sugerido: {optimal_k}")
    
    return {
        'k_values': [int(k) for k in k_values],
        'inertias': [float(i) for i in inertias],
        'optimal_k': int(optimal_k),
        'optimal_k_idx': int(optimal_k_idx)
    }


def apply_kmeans_clustering(
    X: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = 42
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Aplica K-Means clustering.
    
    Args:
        X: Features para clustering
        n_clusters: Número de clusters
        random_state: Semilla para reproducibilidad
        
    Returns:
        Tupla con (modelo, labels, métricas)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"K-MEANS CLUSTERING (K={n_clusters})")
    logger.info(f"{'='*60}")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Calcular métricas
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
    else:
        silhouette = -1
        davies_bouldin = float('inf')
        calinski_harabasz = 0
    
    metrics = {
        'silhouette_score': float(silhouette),
        'davies_bouldin_index': float(davies_bouldin),
        'calinski_harabasz_index': float(calinski_harabasz),
        'inertia': float(kmeans.inertia_),
        'n_clusters': int(n_clusters),
        'n_samples': int(len(X))
    }
    
    logger.info(f"Silhouette Score: {silhouette:.4f}")
    logger.info(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    logger.info(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
    logger.info(f"Inercia: {kmeans.inertia_:.2f}")
    logger.info(f"[OK] K-Means completado")
    
    return kmeans, labels, metrics


def apply_dbscan_clustering(
    X: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    """
    Aplica DBSCAN clustering.
    
    Args:
        X: Features para clustering
        eps: Distancia máxima entre muestras en el mismo cluster
        min_samples: Número mínimo de muestras en un cluster
        
    Returns:
        Tupla con (modelo, labels, métricas)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DBSCAN CLUSTERING (eps={eps}, min_samples={min_samples})")
    logger.info(f"{'='*60}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Calcular métricas (solo si hay más de 1 cluster)
    if n_clusters > 1:
        # Filtrar ruido para métricas
        mask = labels != -1
        if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:
            silhouette = silhouette_score(X[mask], labels[mask])
            davies_bouldin = davies_bouldin_score(X[mask], labels[mask])
            calinski_harabasz = calinski_harabasz_score(X[mask], labels[mask])
        else:
            silhouette = -1
            davies_bouldin = float('inf')
            calinski_harabasz = 0
    else:
        silhouette = -1
        davies_bouldin = float('inf')
        calinski_harabasz = 0
    
    metrics = {
        'silhouette_score': float(silhouette),
        'davies_bouldin_index': float(davies_bouldin),
        'calinski_harabasz_index': float(calinski_harabasz),
        'n_clusters': int(n_clusters),
        'n_noise': int(n_noise),
        'n_samples': int(len(X)),
        'eps': float(eps),
        'min_samples': int(min_samples)
    }
    
    logger.info(f"Número de clusters: {n_clusters}")
    logger.info(f"Puntos de ruido: {n_noise} ({100*n_noise/len(X):.2f}%)")
    if n_clusters > 1:
        logger.info(f"Silhouette Score: {silhouette:.4f}")
        logger.info(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        logger.info(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
    logger.info(f"[OK] DBSCAN completado")
    
    return dbscan, labels, metrics


def apply_hierarchical_clustering(
    X: pd.DataFrame,
    n_clusters: int = 5,
    linkage_method: str = 'ward'
) -> Tuple[Any, np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Aplica Hierarchical Clustering.
    
    Args:
        X: Features para clustering
        n_clusters: Número de clusters
        linkage_method: Método de linkage ('ward', 'complete', 'average')
        
    Returns:
        Tupla con (modelo, labels, métricas, linkage_matrix)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"HIERARCHICAL CLUSTERING (K={n_clusters}, linkage={linkage_method})")
    logger.info(f"{'='*60}")
    
    # Calcular linkage matrix para dendrograma
    if linkage_method == 'ward':
        linkage_matrix = linkage(X, method='ward')
    else:
        linkage_matrix = linkage(X, method=linkage_method, metric='euclidean')
    
    # Aplicar clustering
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method
    )
    labels = hierarchical.fit_predict(X)
    
    # Calcular métricas
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
    else:
        silhouette = -1
        davies_bouldin = float('inf')
        calinski_harabasz = 0
    
    metrics = {
        'silhouette_score': float(silhouette),
        'davies_bouldin_index': float(davies_bouldin),
        'calinski_harabasz_index': float(calinski_harabasz),
        'n_clusters': int(n_clusters),
        'n_samples': int(len(X)),
        'linkage_method': str(linkage_method)
    }
    
    logger.info(f"Silhouette Score: {silhouette:.4f}")
    logger.info(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    logger.info(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
    logger.info(f"[OK] Hierarchical Clustering completado")
    
    return hierarchical, labels, metrics, linkage_matrix


def apply_gmm_clustering(
    X: pd.DataFrame,
    n_components: int = 5,
    random_state: int = 42
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Aplica Gaussian Mixture Models clustering.
    
    Args:
        X: Features para clustering
        n_components: Número de componentes (clusters)
        random_state: Semilla para reproducibilidad
        
    Returns:
        Tupla con (modelo, labels, métricas)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"GAUSSIAN MIXTURE MODEL (n_components={n_components})")
    logger.info(f"{'='*60}")
    
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = gmm.fit_predict(X)
    
    # Calcular métricas
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
    else:
        silhouette = -1
        davies_bouldin = float('inf')
        calinski_harabasz = 0
    
    metrics = {
        'silhouette_score': float(silhouette),
        'davies_bouldin_index': float(davies_bouldin),
        'calinski_harabasz_index': float(calinski_harabasz),
        'aic': float(gmm.aic(X)),
        'bic': float(gmm.bic(X)),
        'n_components': int(n_components),
        'n_samples': int(len(X))
    }
    
    logger.info(f"Silhouette Score: {silhouette:.4f}")
    logger.info(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    logger.info(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
    logger.info(f"AIC: {gmm.aic(X):.2f}")
    logger.info(f"BIC: {gmm.bic(X):.2f}")
    logger.info(f"[OK] GMM completado")
    
    return gmm, labels, metrics


def analyze_cluster_patterns(
    X: pd.DataFrame,
    labels: np.ndarray,
    feature_names: list = None
) -> pd.DataFrame:
    """
    Analiza patrones por cluster: estadísticas, perfiles y características.
    
    Args:
        X: Features originales
        labels: Etiquetas de cluster asignadas
        feature_names: Nombres de las features
        
    Returns:
        DataFrame con análisis de patrones por cluster
    """
    logger.info("\n" + "="*80)
    logger.info("ANÁLISIS DE PATRONES POR CLUSTER")
    logger.info("="*80)
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['cluster'] = labels
    
    cluster_stats = []
    
    for cluster_id in sorted(np.unique(labels)):
        cluster_data = X_df[X_df['cluster'] == cluster_id]
        n_samples = len(cluster_data)
        
        stats = {
            'cluster_id': int(cluster_id),
            'n_samples': int(n_samples),
            'percentage': float(100 * n_samples / len(X_df))
        }
        
        # Estadísticas por feature
        for feature in feature_names:
            stats[f'{feature}_mean'] = float(cluster_data[feature].mean())
            stats[f'{feature}_std'] = float(cluster_data[feature].std())
            stats[f'{feature}_min'] = float(cluster_data[feature].min())
            stats[f'{feature}_max'] = float(cluster_data[feature].max())
        
        cluster_stats.append(stats)
        logger.info(f"\nCluster {cluster_id}: {n_samples} muestras ({100*n_samples/len(X_df):.2f}%)")
        logger.info(f"  Características principales:")
        for feature in feature_names[:5]:  # Mostrar primeras 5
            logger.info(f"    {feature}: {cluster_data[feature].mean():.2f} ± {cluster_data[feature].std():.2f}")
    
    result_df = pd.DataFrame(cluster_stats)
    logger.info(f"\n[OK] Análisis de patrones completado")
    
    return result_df


def compare_clustering_algorithms(
    kmeans_metrics: Dict,
    dbscan_metrics: Dict,
    hierarchical_metrics: Dict,
    gmm_metrics: Dict = None
) -> pd.DataFrame:
    """
    Compara los resultados de diferentes algoritmos de clustering.
    
    Args:
        kmeans_metrics: Métricas de K-Means
        dbscan_metrics: Métricas de DBSCAN
        hierarchical_metrics: Métricas de Hierarchical
        gmm_metrics: Métricas de GMM (opcional)
        
    Returns:
        DataFrame comparativo
    """
    logger.info("\n" + "="*80)
    logger.info("COMPARACIÓN DE ALGORITMOS DE CLUSTERING")
    logger.info("="*80)
    
    comparison = [
        {
            'algorithm': 'K-Means',
            'n_clusters': kmeans_metrics['n_clusters'],
            'silhouette_score': kmeans_metrics['silhouette_score'],
            'davies_bouldin_index': kmeans_metrics['davies_bouldin_index'],
            'calinski_harabasz_index': kmeans_metrics['calinski_harabasz_index']
        },
        {
            'algorithm': 'DBSCAN',
            'n_clusters': dbscan_metrics['n_clusters'],
            'silhouette_score': dbscan_metrics['silhouette_score'],
            'davies_bouldin_index': dbscan_metrics['davies_bouldin_index'],
            'calinski_harabasz_index': dbscan_metrics['calinski_harabasz_index']
        },
        {
            'algorithm': 'Hierarchical',
            'n_clusters': hierarchical_metrics['n_clusters'],
            'silhouette_score': hierarchical_metrics['silhouette_score'],
            'davies_bouldin_index': hierarchical_metrics['davies_bouldin_index'],
            'calinski_harabasz_index': hierarchical_metrics['calinski_harabasz_index']
        }
    ]
    
    if gmm_metrics:
        comparison.append({
            'algorithm': 'GMM',
            'n_clusters': gmm_metrics['n_components'],
            'silhouette_score': gmm_metrics['silhouette_score'],
            'davies_bouldin_index': gmm_metrics['davies_bouldin_index'],
            'calinski_harabasz_index': gmm_metrics['calinski_harabasz_index']
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    logger.info("\n" + comparison_df.to_string(index=False))
    logger.info("\n[OK] Comparación completada")
    
    return comparison_df

