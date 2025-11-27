"""
Nodos para el pipeline de evaluación.
Métricas y visualizaciones de modelos.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
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

# SHAP para interpretabilidad
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP no disponible. Instalar con: pip install shap")

logger = logging.getLogger(__name__)


def evaluate_regression_models(
    predictions: Dict[str, Dict[str, np.ndarray]],
    y_train: pd.Series,
    y_test: pd.Series,
    cv_results: Dict[str, Dict] = None
) -> pd.DataFrame:
    """
    Evalúa modelos de regresión con múltiples métricas, incluyendo CV scores.
    
    Args:
        predictions: Predicciones de todos los modelos
        y_train: Target real de entrenamiento
        y_test: Target real de test
        cv_results: Resultados de CrossValidation (opcional)
        
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
        
        result_dict = {
            'model': model_name,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        # Agregar CV scores si existen
        if cv_results and model_name in cv_results:
            cv_data = cv_results[model_name]
            result_dict['cv_r2_mean'] = cv_data['cv_mean']
            result_dict['cv_r2_std'] = cv_data['cv_std']
            result_dict['best_params'] = str(cv_data['best_params'])
        
        results.append(result_dict)
        
        logger.info(f"\n✓ {model_name}")
        logger.info(f"   RMSE - Train: {train_rmse:.2f} | Test: {test_rmse:.2f}")
        logger.info(f"   MAE  - Train: {train_mae:.2f} | Test: {test_mae:.2f}")
        logger.info(f"   R²   - Train: {train_r2:.4f} | Test: {test_r2:.4f}")
        if cv_results and model_name in cv_results:
            cv_data = cv_results[model_name]
            logger.info(f"   CV R² - Mean: {cv_data['cv_mean']:.4f} (± {cv_data['cv_std']:.4f})")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('test_r2', ascending=False)
    
    logger.info(f"\n🏆 MEJOR MODELO: {df_results.iloc[0]['model']}")
    logger.info(f"   R² Test: {df_results.iloc[0]['test_r2']:.4f}")
    
    return df_results


def evaluate_classification_models(
    predictions: Dict[str, Dict[str, np.ndarray]],
    y_train: pd.Series,
    y_test: pd.Series,
    cv_results: Dict[str, Dict] = None
) -> pd.DataFrame:
    """
    Evalúa modelos de clasificación con múltiples métricas, incluyendo CV scores.
    
    Args:
        predictions: Predicciones de todos los modelos
        y_train: Target real de entrenamiento
        y_test: Target real de test
        cv_results: Resultados de CrossValidation (opcional)
        
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
        
        result_dict = {
            'model': model_name,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'auc_roc': test_auc
        }
        
        # Agregar CV scores si existen
        if cv_results and model_name in cv_results:
            cv_data = cv_results[model_name]
            result_dict['cv_accuracy_mean'] = cv_data['cv_mean']
            result_dict['cv_accuracy_std'] = cv_data['cv_std']
            result_dict['best_params'] = str(cv_data['best_params'])
        
        results.append(result_dict)
        
        logger.info(f"\n✓ {model_name}")
        logger.info(f"   Accuracy  - Train: {train_acc:.4f} | Test: {test_acc:.4f}")
        logger.info(f"   Precision - Test: {test_precision:.4f}")
        logger.info(f"   Recall    - Test: {test_recall:.4f}")
        logger.info(f"   F1-Score  - Test: {test_f1:.4f}")
        logger.info(f"   AUC-ROC   - Test: {test_auc:.4f}")
        if cv_results and model_name in cv_results:
            cv_data = cv_results[model_name]
            logger.info(f"   CV Accuracy - Mean: {cv_data['cv_mean']:.4f} (± {cv_data['cv_std']:.4f})")
    
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


def generate_cv_comparison_table_regression(
    metrics: pd.DataFrame,
    cv_results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Genera tabla comparativa de regresión con mean±std de CrossValidation.
    
    Args:
        metrics: DataFrame con métricas de evaluación
        cv_results: Resultados de CrossValidation
        
    Returns:
        DataFrame con tabla comparativa formateada
    """
    logger.info("="*80)
    logger.info("GENERANDO TABLA COMPARATIVA - REGRESIÓN (CV k=5)")
    logger.info("="*80)
    
    table_data = []
    
    for _, row in metrics.iterrows():
        model_name = row['model']
        
        if model_name in cv_results:
            cv_data = cv_results[model_name]
            table_data.append({
                'Model': model_name,
                'CV R² (mean ± std)': f"{cv_data['cv_mean']:.4f} ± {cv_data['cv_std']:.4f}",
                'Test R²': f"{row['test_r2']:.4f}",
                'Test RMSE': f"{row['test_rmse']:.2f}",
                'Test MAE': f"{row['test_mae']:.2f}",
                'Best Params': cv_data['best_params'] if cv_data['best_params'] else 'default'
            })
        else:
            table_data.append({
                'Model': model_name,
                'CV R² (mean ± std)': 'N/A',
                'Test R²': f"{row['test_r2']:.4f}",
                'Test RMSE': f"{row['test_rmse']:.2f}",
                'Test MAE': f"{row['test_mae']:.2f}",
                'Best Params': 'N/A'
            })
    
    df_table = pd.DataFrame(table_data)
    
    logger.info("\n" + "="*80)
    logger.info("TABLA COMPARATIVA - MODELOS DE REGRESIÓN")
    logger.info("="*80)
    logger.info("\n" + df_table.to_string(index=False))
    logger.info("="*80)
    
    return df_table


def generate_cv_comparison_table_classification(
    metrics: pd.DataFrame,
    cv_results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Genera tabla comparativa de clasificación con mean±std de CrossValidation.
    
    Args:
        metrics: DataFrame con métricas de evaluación
        cv_results: Resultados de CrossValidation
        
    Returns:
        DataFrame con tabla comparativa formateada
    """
    logger.info("="*80)
    logger.info("GENERANDO TABLA COMPARATIVA - CLASIFICACIÓN (CV k=5)")
    logger.info("="*80)
    
    table_data = []
    
    for _, row in metrics.iterrows():
        model_name = row['model']
        
        if model_name in cv_results:
            cv_data = cv_results[model_name]
            table_data.append({
                'Model': model_name,
                'CV Accuracy (mean ± std)': f"{cv_data['cv_mean']:.4f} ± {cv_data['cv_std']:.4f}",
                'Test Accuracy': f"{row['test_accuracy']:.4f}",
                'Precision': f"{row['precision']:.4f}",
                'Recall': f"{row['recall']:.4f}",
                'F1-Score': f"{row['f1_score']:.4f}",
                'AUC-ROC': f"{row['auc_roc']:.4f}",
                'Best Params': cv_data['best_params'] if cv_data['best_params'] else 'default'
            })
        else:
            table_data.append({
                'Model': model_name,
                'CV Accuracy (mean ± std)': 'N/A',
                'Test Accuracy': f"{row['test_accuracy']:.4f}",
                'Precision': f"{row['precision']:.4f}",
                'Recall': f"{row['recall']:.4f}",
                'F1-Score': f"{row['f1_score']:.4f}",
                'AUC-ROC': f"{row['auc_roc']:.4f}",
                'Best Params': 'N/A'
            })
    
    df_table = pd.DataFrame(table_data)
    
    logger.info("\n" + "="*80)
    logger.info("TABLA COMPARATIVA - MODELOS DE CLASIFICACIÓN")
    logger.info("="*80)
    logger.info("\n" + df_table.to_string(index=False))
    logger.info("="*80)
    
    return df_table


def calculate_shap_values_regression(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    metrics: pd.DataFrame,
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Calcula SHAP values para el mejor modelo de regresión.
    
    Args:
        models: Diccionario con modelos entrenados
        X_train: Features de entrenamiento
        X_test: Features de test
        metrics: DataFrame con métricas (para identificar mejor modelo)
        n_samples: Número de muestras para calcular SHAP (para velocidad)
        
    Returns:
        Diccionario con SHAP values y estadísticas
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP no disponible. Saltando cálculo de SHAP values.")
        return {}
    
    logger.info("="*80)
    logger.info("CALCULANDO SHAP VALUES - REGRESIÓN")
    logger.info("="*80)
    
    # Obtener mejor modelo
    best_model_name = metrics.iloc[0]['model']
    
    if best_model_name not in models:
        logger.warning(f"Modelo {best_model_name} no encontrado. Usando primer modelo disponible.")
        best_model_name = list(models.keys())[0]
    
    model = models[best_model_name]
    
    # Usar muestra para velocidad (SHAP puede ser lento)
    if len(X_test) > n_samples:
        X_test_sample = X_test.sample(n=n_samples, random_state=42)
        logger.info(f"Usando muestra de {n_samples} instancias para SHAP (de {len(X_test)} totales)")
    else:
        X_test_sample = X_test
    
    # Seleccionar explainer según tipo de modelo
    try:
        if hasattr(model, 'feature_importances_'):
            # Modelos basados en árboles (Random Forest, Gradient Boosting)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
        else:
            # Modelos lineales (usar KernelExplainer con muestra de background)
            background = X_train.sample(min(100, len(X_train)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_test_sample)
        
        # Calcular estadísticas de SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance_shap = pd.DataFrame({
            'feature': X_test_sample.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        logger.info(f"\n✓ SHAP values calculados para {best_model_name}")
        logger.info(f"\nTop 5 features más importantes (SHAP):")
        for idx, row in feature_importance_shap.head(5).iterrows():
            logger.info(f"   {row['feature']}: {row['mean_abs_shap']:.4f}")
        
        return {
            'model_name': best_model_name,
            'feature_importance': feature_importance_shap.to_dict('records'),
            'explainer_type': 'TreeExplainer' if hasattr(model, 'feature_importances_') else 'KernelExplainer',
            'n_samples': len(X_test_sample)
        }
        
    except Exception as e:
        logger.warning(f"Error calculando SHAP values: {str(e)}")
        return {}


def calculate_shap_values_classification(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    metrics: pd.DataFrame,
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Calcula SHAP values para el mejor modelo de clasificación.
    
    Args:
        models: Diccionario con modelos entrenados
        X_train: Features de entrenamiento
        X_test: Features de test
        metrics: DataFrame con métricas (para identificar mejor modelo)
        n_samples: Número de muestras para calcular SHAP (para velocidad)
        
    Returns:
        Diccionario con SHAP values y estadísticas
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP no disponible. Saltando cálculo de SHAP values.")
        return {}
    
    logger.info("="*80)
    logger.info("CALCULANDO SHAP VALUES - CLASIFICACIÓN")
    logger.info("="*80)
    
    # Obtener mejor modelo
    best_model_name = metrics.iloc[0]['model']
    
    if best_model_name not in models:
        logger.warning(f"Modelo {best_model_name} no encontrado. Usando primer modelo disponible.")
        best_model_name = list(models.keys())[0]
    
    model = models[best_model_name]
    
    # Usar muestra para velocidad (SHAP puede ser lento)
    if len(X_test) > n_samples:
        X_test_sample = X_test.sample(n=n_samples, random_state=42)
        logger.info(f"Usando muestra de {n_samples} instancias para SHAP (de {len(X_test)} totales)")
    else:
        X_test_sample = X_test
    
    # Seleccionar explainer según tipo de modelo
    try:
        if hasattr(model, 'feature_importances_'):
            # Modelos basados en árboles (Random Forest, Gradient Boosting)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
            # Para clasificación binaria, shap_values puede ser una lista
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Usar valores para clase positiva
        else:
            # Modelos lineales o SVM (usar KernelExplainer con muestra de background)
            background = X_train.sample(min(100, len(X_train)), random_state=42)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_test_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Usar valores para clase positiva
        
        # Calcular estadísticas de SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance_shap = pd.DataFrame({
            'feature': X_test_sample.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        logger.info(f"\n✓ SHAP values calculados para {best_model_name}")
        logger.info(f"\nTop 5 features más importantes (SHAP):")
        for idx, row in feature_importance_shap.head(5).iterrows():
            logger.info(f"   {row['feature']}: {row['mean_abs_shap']:.4f}")
        
        return {
            'model_name': best_model_name,
            'feature_importance': feature_importance_shap.to_dict('records'),
            'explainer_type': 'TreeExplainer' if hasattr(model, 'feature_importances_') else 'KernelExplainer',
            'n_samples': len(X_test_sample)
        }
        
    except Exception as e:
        logger.warning(f"Error calculando SHAP values: {str(e)}")
        return {}

