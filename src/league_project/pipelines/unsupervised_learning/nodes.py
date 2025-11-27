"""
Nodos de integración para aprendizaje no supervisado.
Integra clustering con modelos supervisados.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Any
import logging

logger = logging.getLogger(__name__)


def integrate_clustering_features(
    X_train: pd.DataFrame,
    train_labels: np.ndarray,
    X_test: pd.DataFrame,
    clustering_model: Any
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Integra las etiquetas de clustering como features para modelos supervisados.
    
    Args:
        X_train: Features de entrenamiento
        train_labels: Etiquetas de cluster para entrenamiento
        X_test: Features de test
        clustering_model: Modelo de clustering entrenado
        
    Returns:
        Tupla con (X_train con clusters, X_test con clusters)
    """
    logger.info("="*80)
    logger.info("INTEGRACIÓN: CLUSTERING COMO FEATURE ENGINEERING")
    logger.info("="*80)
    
    # Predecir clusters para test
    test_labels = clustering_model.predict(X_test)
    
    # Agregar cluster como feature one-hot encoded
    train_df = X_train.copy()
    test_df = X_test.copy()
    
    # One-hot encoding de clusters
    n_clusters = len(np.unique(train_labels))
    
    for i in range(n_clusters):
        train_df[f'cluster_{i}'] = (train_labels == i).astype(int)
        test_df[f'cluster_{i}'] = (test_labels == i).astype(int)
    
    logger.info(f"[OK] Features de clustering agregadas: {n_clusters} features one-hot")
    logger.info(f"  Dimensiones originales: {X_train.shape[1]}")
    logger.info(f"  Dimensiones con clusters: {train_df.shape[1]}")
    
    return train_df, test_df

