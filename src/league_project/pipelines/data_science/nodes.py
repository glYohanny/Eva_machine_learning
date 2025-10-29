"""
Nodos para el pipeline de data science.
Entrenamiento de modelos de regresión y clasificación.
Incluye GridSearchCV y CrossValidation (k=5).
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
from sklearn.model_selection import GridSearchCV, cross_val_score

logger = logging.getLogger(__name__)


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict
) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
    """
    Entrena múltiples modelos de regresión con GridSearchCV y CrossValidation (k=5).
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de regresión (gamelength)
        parameters: Parámetros de configuración
        
    Returns:
        Tupla con (modelos entrenados, resultados de CV)
    """
    logger.info("="*80)
    logger.info("ENTRENANDO MODELOS DE REGRESIÓN CON GRIDSEARCHCV + CV (k=5)")
    logger.info("="*80)
    
    # Definir modelos y sus grids de hiperparámetros
    models_params = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {}  # Linear Regression no tiene hiperparámetros
        },
        'ridge': {
            'model': Ridge(random_state=parameters['random_state']),
            'params': {
                'alpha': [0.1, 0.5, 1.0, 5.0, 10.0],
                'solver': ['auto', 'svd', 'cholesky']
            }
        },
        'lasso': {
            'model': Lasso(random_state=parameters['random_state'], max_iter=10000),
            'params': {
                'alpha': [0.01, 0.05, 0.1, 0.5, 1.0],
                'selection': ['cyclic', 'random']
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(random_state=parameters['random_state'], n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor(random_state=parameters['random_state']),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            }
        },
    }
    
    trained_models = {}
    cv_results = {}
    
    for name, config in models_params.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Entrenando: {name}")
        logger.info(f"{'='*60}")
        
        # GridSearchCV (si tiene hiperparámetros)
        if config['params']:
            logger.info(f"Ejecutando GridSearchCV con {len(config['params'])} hiperparámetros...")
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=5,  # k=5
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Mejores hiperparámetros encontrados:")
            for param, value in best_params.items():
                logger.info(f"   {param}: {value}")
        else:
            # Modelos sin hiperparámetros (Linear Regression)
            best_model = config['model']
            best_model.fit(X_train, y_train)
            best_params = {}
            logger.info("Modelo sin hiperparámetros, entrenamiento directo.")
        
        # CrossValidation (k=5) con el mejor modelo
        logger.info(f"Ejecutando CrossValidation (k=5)...")
        cv_scores = cross_val_score(
            best_model, X_train, y_train, 
            cv=5, scoring='r2', n_jobs=-1
        )
        
        trained_models[name] = best_model
        cv_results[name] = {
            'best_params': best_params,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        logger.info(f"CV R² Score: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
        logger.info(f"✓ {name} entrenado y validado")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✓ Total modelos de regresión entrenados: {len(trained_models)}")
    logger.info(f"{'='*80}")
    
    return trained_models, cv_results


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict
) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
    """
    Entrena múltiples modelos de clasificación con GridSearchCV y CrossValidation (k=5).
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de clasificación (bResult)
        parameters: Parámetros de configuración
        
    Returns:
        Tupla con (modelos entrenados, resultados de CV)
    """
    logger.info("="*80)
    logger.info("ENTRENANDO MODELOS DE CLASIFICACIÓN CON GRIDSEARCHCV + CV (k=5)")
    logger.info("="*80)
    
    # Definir modelos y sus grids de hiperparámetros
    models_params = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=parameters['random_state'], max_iter=1000),
            'params': {
                'C': [0.1, 0.5, 1.0, 5.0, 10.0],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=parameters['random_state'], n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=parameters['random_state']),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            }
        },
        'svm': {
            'model': SVC(random_state=parameters['random_state'], probability=True),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'naive_bayes': {
            'model': GaussianNB(),
            'params': {}  # Naive Bayes no tiene hiperparámetros a tunear
        },
    }
    
    trained_models = {}
    cv_results = {}
    
    for name, config in models_params.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Entrenando: {name}")
        logger.info(f"{'='*60}")
        
        # GridSearchCV (si tiene hiperparámetros)
        if config['params']:
            logger.info(f"Ejecutando GridSearchCV con {len(config['params'])} hiperparámetros...")
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=5,  # k=5
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Mejores hiperparámetros encontrados:")
            for param, value in best_params.items():
                logger.info(f"   {param}: {value}")
        else:
            # Modelos sin hiperparámetros (Naive Bayes)
            best_model = config['model']
            best_model.fit(X_train, y_train)
            best_params = {}
            logger.info("Modelo sin hiperparámetros, entrenamiento directo.")
        
        # CrossValidation (k=5) con el mejor modelo
        logger.info(f"Ejecutando CrossValidation (k=5)...")
        cv_scores = cross_val_score(
            best_model, X_train, y_train, 
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        trained_models[name] = best_model
        cv_results[name] = {
            'best_params': best_params,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
        logger.info(f"✓ {name} entrenado y validado")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✓ Total modelos de clasificación entrenados: {len(trained_models)}")
    logger.info(f"{'='*80}")
    
    return trained_models, cv_results


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




