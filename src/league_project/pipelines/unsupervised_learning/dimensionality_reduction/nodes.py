"""
Nodos para el pipeline de reducción de dimensionalidad.
Implementa PCA, t-SNE y UMAP con análisis completo.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import logging

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP no disponible. Instalar con: pip install umap-learn")

logger = logging.getLogger(__name__)


def apply_pca(
    X: pd.DataFrame,
    n_components: int = None,
    random_state: int = 42
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    """
    Aplica PCA (Principal Component Analysis).
    
    Args:
        X: Features originales
        n_components: Número de componentes (None = todos)
        random_state: Semilla para reproducibilidad
        
    Returns:
        Tupla con (modelo, datos transformados, métricas)
    """
    logger.info("="*80)
    logger.info("PCA - ANÁLISIS DE COMPONENTES PRINCIPALES")
    logger.info("="*80)
    
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    # Calcular varianza explicada
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Crear DataFrame con componentes principales
    component_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    X_pca_df = pd.DataFrame(X_pca, columns=component_names, index=X.index if hasattr(X, 'index') else None)
    
    # Métricas
    metrics = {
        'n_components': int(pca.n_components_),
        'explained_variance_ratio': explained_variance.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'total_variance_explained': float(cumulative_variance[-1]) if len(cumulative_variance) > 0 else 0.0,
        'n_features_original': int(X.shape[1])
    }
    
    logger.info(f"Número de componentes: {pca.n_components_}")
    logger.info(f"Varianza total explicada: {cumulative_variance[-1]:.4f} ({cumulative_variance[-1]*100:.2f}%)")
    logger.info(f"\nVarianza explicada por componente:")
    for i, (var, cum_var) in enumerate(zip(explained_variance[:10], cumulative_variance[:10])):
        logger.info(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%) - Acumulado: {cum_var:.4f} ({cum_var*100:.2f}%)")
    
    # Loadings (contribución de cada feature original a cada componente)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=component_names,
        index=X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
    )
    
    metrics['loadings'] = loadings.to_dict()
    
    logger.info(f"\n[OK] PCA completado")
    
    return pca, X_pca_df, metrics


def apply_tsne(
    X: pd.DataFrame,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
    n_iter: int = 1000
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    """
    Aplica t-SNE para visualización.
    
    Args:
        X: Features originales
        n_components: Dimensiones de salida (2 o 3 para visualización)
        perplexity: Perplejidad (típicamente entre 5 y 50)
        random_state: Semilla para reproducibilidad
        n_iter: Número de iteraciones
        
    Returns:
        Tupla con (modelo, datos transformados, métricas)
    """
    logger.info("="*80)
    logger.info(f"t-SNE - REDUCCIÓN DIMENSIONAL (n_components={n_components})")
    logger.info("="*80)
    
    # t-SNE es computacionalmente costoso, usar PCA previo si hay muchas features
    if X.shape[1] > 50:
        logger.info(f"Reduciendo dimensionalidad con PCA previo (de {X.shape[1]} a 50 features)...")
        pca_pre = PCA(n_components=50, random_state=random_state)
        X_reduced = pca_pre.fit_transform(X)
        logger.info("[OK] PCA previo completado")
    else:
        X_reduced = X.values if isinstance(X, pd.DataFrame) else X
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=n_iter,
        verbose=1
    )
    
    logger.info("Aplicando t-SNE (esto puede tardar varios minutos)...")
    X_tsne = tsne.fit_transform(X_reduced)
    
    # Crear DataFrame
    component_names = [f'tSNE_{i+1}' for i in range(n_components)]
    X_tsne_df = pd.DataFrame(
        X_tsne,
        columns=component_names,
        index=X.index if hasattr(X, 'index') else None
    )
    
    metrics = {
        'n_components': int(n_components),
        'perplexity': float(perplexity),
        'n_iter': int(n_iter),
        'kl_divergence': float(tsne.kl_divergence_),
        'n_features_original': int(X.shape[1])
    }
    
    logger.info(f"KL Divergence: {tsne.kl_divergence_:.4f}")
    logger.info(f"[OK] t-SNE completado")
    
    return tsne, X_tsne_df, metrics


def apply_umap(
    X: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    """
    Aplica UMAP para reducción de dimensionalidad.
    
    Args:
        X: Features originales
        n_components: Dimensiones de salida (2 o 3 para visualización)
        n_neighbors: Número de vecinos
        min_dist: Distancia mínima entre puntos
        random_state: Semilla para reproducibilidad
        
    Returns:
        Tupla con (modelo, datos transformados, métricas)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP no está instalado. Instalar con: pip install umap-learn")
    
    logger.info("="*80)
    logger.info(f"UMAP - REDUCCIÓN DIMENSIONAL (n_components={n_components})")
    logger.info("="*80)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    
    logger.info("Aplicando UMAP...")
    X_umap = reducer.fit_transform(X)
    
    # Crear DataFrame
    component_names = [f'UMAP_{i+1}' for i in range(n_components)]
    X_umap_df = pd.DataFrame(
        X_umap,
        columns=component_names,
        index=X.index if hasattr(X, 'index') else None
    )
    
    metrics = {
        'n_components': int(n_components),
        'n_neighbors': int(n_neighbors),
        'min_dist': float(min_dist),
        'n_features_original': int(X.shape[1])
    }
    
    logger.info(f"[OK] UMAP completado")
    
    return reducer, X_umap_df, metrics


def analyze_pca_components(
    pca_model: Any,
    feature_names: list,
    n_components: int = 5
) -> pd.DataFrame:
    """
    Analiza los componentes principales: loadings y contribuciones.
    
    Args:
        pca_model: Modelo PCA entrenado
        feature_names: Nombres de las features originales
        n_components: Número de componentes a analizar
        
    Returns:
        DataFrame con análisis de componentes
    """
    logger.info("\n" + "="*80)
    logger.info("ANÁLISIS DE COMPONENTES PRINCIPALES (LOADINGS)")
    logger.info("="*80)
    
    # Si n_components es None, usar todos los componentes
    if n_components is None:
        n_components = pca_model.n_components_
    else:
        n_components = min(n_components, pca_model.n_components_)
    
    loadings_df = pd.DataFrame(
        pca_model.components_[:n_components].T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )
    
    # Calcular contribución absoluta de cada feature
    contributions = {}
    for pc in loadings_df.columns:
        contributions[f'{pc}_abs_contribution'] = loadings_df[pc].abs()
    
    contributions_df = pd.DataFrame(contributions, index=feature_names)
    
    # Top features por componente
    logger.info("\nTop 5 features por componente principal:")
    for pc in loadings_df.columns:
        top_features = loadings_df[pc].abs().nlargest(5)
        logger.info(f"\n{pc}:")
        for feature, value in top_features.items():
            logger.info(f"  {feature}: {value:.4f}")
    
    logger.info(f"\n[OK] Análisis de componentes completado")
    
    return loadings_df

