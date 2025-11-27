"""
Nodos para el pipeline de detección de anomalías.
Implementa Isolation Forest y Local Outlier Factor (LOF).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import logging

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
try:
    from pyod.models.iforest import IForest as PyOD_IForest
    from pyod.models.lof import LOF as PyOD_LOF
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    logging.warning("PyOD no disponible. Usando scikit-learn para detección de anomalías.")

logger = logging.getLogger(__name__)


def apply_isolation_forest(
    X: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 42
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    """
    Aplica Isolation Forest para detección de anomalías.
    
    Args:
        X: Features para análisis
        contamination: Proporción esperada de anomalías
        random_state: Semilla para reproducibilidad
        
    Returns:
        Tupla con (modelo, predicciones, métricas)
    """
    logger.info("="*80)
    logger.info("ISOLATION FOREST - DETECCIÓN DE ANOMALÍAS")
    logger.info("="*80)
    
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    
    predictions = iso_forest.fit_predict(X)
    # Convertir: 1 = normal, -1 = anomalía
    anomaly_labels = (predictions == -1).astype(int)
    anomaly_scores = iso_forest.score_samples(X)
    
    n_anomalies = anomaly_labels.sum()
    n_normal = len(X) - n_anomalies
    
    metrics = {
        'n_anomalies': int(n_anomalies),
        'n_normal': int(n_normal),
        'anomaly_percentage': float(100 * n_anomalies / len(X)),
        'contamination': float(contamination),
        'mean_anomaly_score': float(anomaly_scores.mean()),
        'std_anomaly_score': float(anomaly_scores.std()),
        'min_anomaly_score': float(anomaly_scores.min()),
        'max_anomaly_score': float(anomaly_scores.max())
    }
    
    logger.info(f"Anomalías detectadas: {n_anomalies} ({100*n_anomalies/len(X):.2f}%)")
    logger.info(f"Puntos normales: {n_normal} ({100*n_normal/len(X):.2f}%)")
    logger.info(f"Score promedio: {anomaly_scores.mean():.4f} ± {anomaly_scores.std():.4f}")
    logger.info(f"[OK] Isolation Forest completado")
    
    return iso_forest, anomaly_labels, metrics


def apply_lof(
    X: pd.DataFrame,
    n_neighbors: int = 20,
    contamination: float = 0.1
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    """
    Aplica Local Outlier Factor (LOF) para detección de anomalías.
    
    Args:
        X: Features para análisis
        n_neighbors: Número de vecinos
        contamination: Proporción esperada de anomalías
        
    Returns:
        Tupla con (modelo, predicciones, métricas)
    """
    logger.info("="*80)
    logger.info("LOCAL OUTLIER FACTOR (LOF) - DETECCIÓN DE ANOMALÍAS")
    logger.info("="*80)
    
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False
    )
    
    predictions = lof.fit_predict(X)
    # Convertir: 1 = normal, -1 = anomalía
    anomaly_labels = (predictions == -1).astype(int)
    anomaly_scores = -lof.negative_outlier_factor_  # Negativo porque LOF usa negativo
    
    n_anomalies = anomaly_labels.sum()
    n_normal = len(X) - n_anomalies
    
    metrics = {
        'n_anomalies': int(n_anomalies),
        'n_normal': int(n_normal),
        'anomaly_percentage': float(100 * n_anomalies / len(X)),
        'n_neighbors': int(n_neighbors),
        'contamination': float(contamination),
        'mean_anomaly_score': float(anomaly_scores.mean()),
        'std_anomaly_score': float(anomaly_scores.std()),
        'min_anomaly_score': float(anomaly_scores.min()),
        'max_anomaly_score': float(anomaly_scores.max())
    }
    
    logger.info(f"Anomalías detectadas: {n_anomalies} ({100*n_anomalies/len(X):.2f}%)")
    logger.info(f"Puntos normales: {n_normal} ({100*n_normal/len(X):.2f}%)")
    logger.info(f"Score promedio: {anomaly_scores.mean():.4f} ± {anomaly_scores.std():.4f}")
    logger.info(f"[OK] LOF completado")
    
    return lof, anomaly_labels, metrics


def analyze_anomalies(
    X: pd.DataFrame,
    iso_forest_labels: np.ndarray,
    lof_labels: np.ndarray,
    feature_names: list = None
) -> pd.DataFrame:
    """
    Analiza las anomalías detectadas por ambos métodos.
    
    Args:
        X: Features originales
        iso_forest_labels: Etiquetas de Isolation Forest
        lof_labels: Etiquetas de LOF
        feature_names: Nombres de las features
        
    Returns:
        DataFrame con análisis de anomalías
    """
    logger.info("\n" + "="*80)
    logger.info("ANÁLISIS DE ANOMALÍAS")
    logger.info("="*80)
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['isolation_forest_anomaly'] = iso_forest_labels
    X_df['lof_anomaly'] = lof_labels
    X_df['both_methods_anomaly'] = ((iso_forest_labels == 1) & (lof_labels == 1)).astype(int)
    
    # Estadísticas de anomalías
    analysis_data = []
    
    # Isolation Forest
    n_iso_anomalies = int(iso_forest_labels.sum())
    analysis_data.append({
        'method': 'Isolation Forest',
        'n_anomalies': n_iso_anomalies,
        'percentage': float(100 * n_iso_anomalies / len(X)),
        'n_normal': int(len(X) - n_iso_anomalies)
    })
    
    # LOF
    n_lof_anomalies = int(lof_labels.sum())
    analysis_data.append({
        'method': 'LOF',
        'n_anomalies': n_lof_anomalies,
        'percentage': float(100 * n_lof_anomalies / len(X)),
        'n_normal': int(len(X) - n_lof_anomalies)
    })
    
    # Ambos métodos
    n_both_anomalies = int(X_df['both_methods_anomaly'].sum())
    analysis_data.append({
        'method': 'Ambos métodos',
        'n_anomalies': n_both_anomalies,
        'percentage': float(100 * n_both_anomalies / len(X)),
        'n_normal': int(len(X) - n_both_anomalies)
    })
    
    analysis_df = pd.DataFrame(analysis_data)
    
    logger.info("\nResumen de anomalías detectadas:")
    logger.info(analysis_df.to_string(index=False))
    logger.info(f"\n[OK] Análisis de anomalías completado")
    
    return analysis_df


def compare_anomaly_detection(
    iso_forest_metrics: Dict,
    lof_metrics: Dict
) -> pd.DataFrame:
    """
    Compara los resultados de diferentes métodos de detección de anomalías.
    
    Args:
        iso_forest_metrics: Métricas de Isolation Forest
        lof_metrics: Métricas de LOF
        
    Returns:
        DataFrame comparativo
    """
    logger.info("\n" + "="*80)
    logger.info("COMPARACIÓN DE MÉTODOS DE DETECCIÓN DE ANOMALÍAS")
    logger.info("="*80)
    
    comparison = pd.DataFrame([
        {
            'method': 'Isolation Forest',
            'n_anomalies': iso_forest_metrics['n_anomalies'],
            'anomaly_percentage': iso_forest_metrics['anomaly_percentage'],
            'mean_score': iso_forest_metrics['mean_anomaly_score'],
            'std_score': iso_forest_metrics['std_anomaly_score']
        },
        {
            'method': 'LOF',
            'n_anomalies': lof_metrics['n_anomalies'],
            'anomaly_percentage': lof_metrics['anomaly_percentage'],
            'mean_score': lof_metrics['mean_anomaly_score'],
            'std_score': lof_metrics['std_anomaly_score']
        }
    ])
    
    logger.info("\n" + comparison.to_string(index=False))
    logger.info("\n[OK] Comparación completada")
    
    return comparison

