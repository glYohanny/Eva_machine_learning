"""
Nodos para el pipeline de data science.
Entrenamiento de modelos de regresión y clasificación.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import logging

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB

logger = logging.getLogger(__name__)


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict
) -> Dict[str, Any]:
    """
    Entrena múltiples modelos de regresión.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de regresión (gamelength)
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con modelos entrenados
    """
    logger.info("="*80)
    logger.info("ENTRENANDO MODELOS DE REGRESIÓN")
    logger.info("="*80)
    
    reg_params = parameters.get('regression_models', {})
    
    models = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(
            alpha=reg_params.get('ridge_alpha', 1.0),
            random_state=parameters['random_state']
        ),
        'lasso': Lasso(
            alpha=reg_params.get('lasso_alpha', 0.1),
            random_state=parameters['random_state']
        ),
        'random_forest': RandomForestRegressor(
            n_estimators=reg_params.get('rf_n_estimators', 100),
            max_depth=reg_params.get('rf_max_depth', 15),
            random_state=parameters['random_state'],
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=reg_params.get('gb_n_estimators', 100),
            learning_rate=reg_params.get('gb_learning_rate', 0.1),
            random_state=parameters['random_state']
        ),
    }
    
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"✓ {name} entrenado")
    
    logger.info(f"\n✓ Total modelos de regresión entrenados: {len(trained_models)}")
    
    return trained_models


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict
) -> Dict[str, Any]:
    """
    Entrena múltiples modelos de clasificación.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de clasificación (bResult)
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con modelos entrenados
    """
    logger.info("="*80)
    logger.info("ENTRENANDO MODELOS DE CLASIFICACIÓN")
    logger.info("="*80)
    
    cls_params = parameters.get('classification_models', {})
    
    models = {
        'logistic_regression': LogisticRegression(
            max_iter=cls_params.get('lr_max_iter', 1000),
            random_state=parameters['random_state']
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=cls_params.get('rf_n_estimators', 100),
            max_depth=cls_params.get('rf_max_depth', 15),
            random_state=parameters['random_state'],
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=cls_params.get('gb_n_estimators', 100),
            learning_rate=cls_params.get('gb_learning_rate', 0.1),
            random_state=parameters['random_state']
        ),
        'svm': SVC(
            kernel=cls_params.get('svm_kernel', 'rbf'),
            probability=True,
            random_state=parameters['random_state']
        ),
        'naive_bayes': GaussianNB(),
    }
    
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"✓ {name} entrenado")
    
    logger.info(f"\n✓ Total modelos de clasificación entrenados: {len(trained_models)}")
    
    return trained_models


def make_regression_predictions(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Genera predicciones de todos los modelos de regresión.
    
    Args:
        models: Modelos entrenados
        X_train: Features de entrenamiento
        X_test: Features de test
        
    Returns:
        Diccionario con predicciones train y test por modelo
    """
    logger.info("Generando predicciones de regresión...")
    
    predictions = {}
    
    for name, model in models.items():
        predictions[name] = {
            'train': model.predict(X_train),
            'test': model.predict(X_test)
        }
    
    logger.info(f"✓ Predicciones generadas para {len(models)} modelos")
    
    return predictions


def make_classification_predictions(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Genera predicciones de todos los modelos de clasificación.
    
    Args:
        models: Modelos entrenados
        X_train: Features de entrenamiento
        X_test: Features de test
        
    Returns:
        Diccionario con predicciones y probabilidades train y test por modelo
    """
    logger.info("Generando predicciones de clasificación...")
    
    predictions = {}
    
    for name, model in models.items():
        predictions[name] = {
            'train_pred': model.predict(X_train),
            'test_pred': model.predict(X_test),
            'train_proba': model.predict_proba(X_train)[:, 1],
            'test_proba': model.predict_proba(X_test)[:, 1]
        }
    
    logger.info(f"✓ Predicciones generadas para {len(models)} modelos")
    
    return predictions




