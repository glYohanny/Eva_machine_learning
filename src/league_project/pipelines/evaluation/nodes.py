"""
Nodos para el pipeline de evaluación.
Métricas y visualizaciones de modelos.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


def evaluate_regression_models(
    predictions: Dict[str, Dict[str, np.ndarray]],
    y_train: pd.Series,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Evalúa modelos de regresión con múltiples métricas.
    
    Args:
        predictions: Predicciones de todos los modelos
        y_train: Target real de entrenamiento
        y_test: Target real de test
        
    Returns:
        DataFrame con métricas de evaluación
    """
    logger.info("="*80)
    logger.info("EVALUACIÓN DE MODELOS DE REGRESIÓN")
    logger.info("="*80)
    
    results = []
    
    for model_name, preds in predictions.items():
        # Métricas en train
        train_rmse = np.sqrt(mean_squared_error(y_train, preds['train']))
        train_mae = mean_absolute_error(y_train, preds['train'])
        train_r2 = r2_score(y_train, preds['train'])
        
        # Métricas en test
        test_rmse = np.sqrt(mean_squared_error(y_test, preds['test']))
        test_mae = mean_absolute_error(y_test, preds['test'])
        test_r2 = r2_score(y_test, preds['test'])
        
        results.append({
            'model': model_name,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        })
        
        logger.info(f"\n✓ {model_name}")
        logger.info(f"   RMSE - Train: {train_rmse:.2f} | Test: {test_rmse:.2f}")
        logger.info(f"   MAE  - Train: {train_mae:.2f} | Test: {test_mae:.2f}")
        logger.info(f"   R²   - Train: {train_r2:.4f} | Test: {test_r2:.4f}")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('test_r2', ascending=False)
    
    logger.info(f"\n🏆 MEJOR MODELO: {df_results.iloc[0]['model']}")
    logger.info(f"   R² Test: {df_results.iloc[0]['test_r2']:.4f}")
    
    return df_results


def evaluate_classification_models(
    predictions: Dict[str, Dict[str, np.ndarray]],
    y_train: pd.Series,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Evalúa modelos de clasificación con múltiples métricas.
    
    Args:
        predictions: Predicciones de todos los modelos
        y_train: Target real de entrenamiento
        y_test: Target real de test
        
    Returns:
        DataFrame con métricas de evaluación
    """
    logger.info("="*80)
    logger.info("EVALUACIÓN DE MODELOS DE CLASIFICACIÓN")
    logger.info("="*80)
    
    results = []
    
    for model_name, preds in predictions.items():
        # Métricas en train
        train_acc = accuracy_score(y_train, preds['train_pred'])
        
        # Métricas en test
        test_acc = accuracy_score(y_test, preds['test_pred'])
        test_precision = precision_score(y_test, preds['test_pred'])
        test_recall = recall_score(y_test, preds['test_pred'])
        test_f1 = f1_score(y_test, preds['test_pred'])
        test_auc = roc_auc_score(y_test, preds['test_proba'])
        
        results.append({
            'model': model_name,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'auc_roc': test_auc
        })
        
        logger.info(f"\n✓ {model_name}")
        logger.info(f"   Accuracy  - Train: {train_acc:.4f} | Test: {test_acc:.4f}")
        logger.info(f"   Precision - Test: {test_precision:.4f}")
        logger.info(f"   Recall    - Test: {test_recall:.4f}")
        logger.info(f"   F1-Score  - Test: {test_f1:.4f}")
        logger.info(f"   AUC-ROC   - Test: {test_auc:.4f}")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('f1_score', ascending=False)
    
    logger.info(f"\n🏆 MEJOR MODELO: {df_results.iloc[0]['model']}")
    logger.info(f"   F1-Score: {df_results.iloc[0]['f1_score']:.4f}")
    
    return df_results


def get_feature_importance(
    models: Dict[str, Any],
    feature_names: list
) -> pd.DataFrame:
    """
    Extrae feature importance de modelos basados en árboles.
    
    Args:
        models: Modelos entrenados
        feature_names: Nombres de las features
        
    Returns:
        DataFrame con importancia de features
    """
    logger.info("Extrayendo importancia de features...")
    
    importance_data = []
    
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            for feature, importance in zip(feature_names, model.feature_importances_):
                importance_data.append({
                    'model': model_name,
                    'feature': feature,
                    'importance': importance
                })
    
    df_importance = pd.DataFrame(importance_data)
    
    logger.info(f"✓ Feature importance extraída de {df_importance['model'].nunique()} modelos")
    
    return df_importance


def create_regression_report(
    metrics: pd.DataFrame,
    feature_importance: pd.DataFrame
) -> Dict[str, Any]:
    """
    Crea reporte completo de regresión.
    
    Args:
        metrics: DataFrame con métricas
        feature_importance: DataFrame con importancia de features
        
    Returns:
        Diccionario con el reporte
    """
    logger.info("Creando reporte de regresión...")
    
    best_model = metrics.iloc[0]
    
    # Top 5 features más importantes del mejor modelo basado en árboles
    top_features = None
    if len(feature_importance) > 0:
        rf_importance = feature_importance[feature_importance['model'] == 'random_forest']
        if len(rf_importance) > 0:
            top_features = rf_importance.nlargest(5, 'importance')['feature'].tolist()
    
    report = {
        'best_model': best_model['model'],
        'best_r2': float(best_model['test_r2']),
        'best_rmse': float(best_model['test_rmse']),
        'best_mae': float(best_model['test_mae']),
        'all_metrics': metrics.to_dict('records'),
        'top_features': top_features
    }
    
    logger.info(f"✓ Reporte de regresión creado")
    logger.info(f"   Mejor modelo: {report['best_model']}")
    logger.info(f"   R² Test: {report['best_r2']:.4f}")
    
    return report


def create_classification_report(
    metrics: pd.DataFrame,
    feature_importance: pd.DataFrame
) -> Dict[str, Any]:
    """
    Crea reporte completo de clasificación.
    
    Args:
        metrics: DataFrame con métricas
        feature_importance: DataFrame con importancia de features
        
    Returns:
        Diccionario con el reporte
    """
    logger.info("Creando reporte de clasificación...")
    
    best_model = metrics.iloc[0]
    
    # Top 5 features más importantes del mejor modelo basado en árboles
    top_features = None
    if len(feature_importance) > 0:
        rf_importance = feature_importance[feature_importance['model'] == 'random_forest']
        if len(rf_importance) > 0:
            top_features = rf_importance.nlargest(5, 'importance')['feature'].tolist()
    
    report = {
        'best_model': best_model['model'],
        'best_accuracy': float(best_model['test_accuracy']),
        'best_f1': float(best_model['f1_score']),
        'best_auc': float(best_model['auc_roc']),
        'all_metrics': metrics.to_dict('records'),
        'top_features': top_features
    }
    
    logger.info(f"✓ Reporte de clasificación creado")
    logger.info(f"   Mejor modelo: {report['best_model']}")
    logger.info(f"   F1-Score: {report['best_f1']:.4f}")
    
    return report




