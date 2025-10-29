"""
Nodos para el pipeline de procesamiento de datos.
Feature engineering y preparación de datos para modelado.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def aggregate_kills_features(
    df_matches: pd.DataFrame,
    df_kills: pd.DataFrame
) -> pd.DataFrame:
    """
    Agrega features de kills por partida.
    
    Args:
        df_matches: DataFrame con información de partidas
        df_kills: DataFrame con información de kills
        
    Returns:
        DataFrame con features de kills agregados
    """
    logger.info("Agregando features de kills...")
    
    # Contar kills por equipo
    kills_blue = df_kills[df_kills['Team'] == 'bKills'].groupby('Address').size().reset_index(name='blue_kills')
    kills_red = df_kills[df_kills['Team'] == 'rKills'].groupby('Address').size().reset_index(name='red_kills')
    
    # Merge con el dataset principal
    df_features = df_matches.copy()
    df_features = df_features.merge(kills_blue, on='Address', how='left')
    df_features = df_features.merge(kills_red, on='Address', how='left')
    
    # Llenar NaN con 0
    df_features['blue_kills'] = df_features['blue_kills'].fillna(0)
    df_features['red_kills'] = df_features['red_kills'].fillna(0)
    
    # Crear feature de diferencia
    df_features['kill_diff'] = df_features['blue_kills'] - df_features['red_kills']
    
    logger.info(f"✓ Features de kills agregados: blue_kills, red_kills, kill_diff")
    
    return df_features


def aggregate_monsters_features(
    df_features: pd.DataFrame,
    df_monsters: pd.DataFrame
) -> pd.DataFrame:
    """
    Agrega features de objetivos (dragones y barones).
    
    Args:
        df_features: DataFrame con features actuales
        df_monsters: DataFrame con información de monstruos
        
    Returns:
        DataFrame con features de monstruos agregados
    """
    logger.info("Agregando features de objetivos (dragones/barones)...")
    
    # Dragones por equipo
    dragons_blue = df_monsters[df_monsters['Team'] == 'bDragons'].groupby('Address').size().reset_index(name='blue_dragons')
    dragons_red = df_monsters[df_monsters['Team'] == 'rDragons'].groupby('Address').size().reset_index(name='red_dragons')
    
    # Barones por equipo
    barons_blue = df_monsters[df_monsters['Team'] == 'bBarons'].groupby('Address').size().reset_index(name='blue_barons')
    barons_red = df_monsters[df_monsters['Team'] == 'rBarons'].groupby('Address').size().reset_index(name='red_barons')
    
    # Merge
    df_result = df_features.copy()
    df_result = df_result.merge(dragons_blue, on='Address', how='left')
    df_result = df_result.merge(dragons_red, on='Address', how='left')
    df_result = df_result.merge(barons_blue, on='Address', how='left')
    df_result = df_result.merge(barons_red, on='Address', how='left')
    
    # Llenar NaN con 0
    df_result['blue_dragons'] = df_result['blue_dragons'].fillna(0)
    df_result['red_dragons'] = df_result['red_dragons'].fillna(0)
    df_result['blue_barons'] = df_result['blue_barons'].fillna(0)
    df_result['red_barons'] = df_result['red_barons'].fillna(0)
    
    # Features de diferencia
    df_result['dragon_diff'] = df_result['blue_dragons'] - df_result['red_dragons']
    df_result['baron_diff'] = df_result['blue_barons'] - df_result['red_barons']
    
    logger.info(f"✓ Features de objetivos agregados")
    
    return df_result


def aggregate_structures_features(
    df_features: pd.DataFrame,
    df_structures: pd.DataFrame
) -> pd.DataFrame:
    """
    Agrega features de estructuras (torres e inhibidores).
    
    Args:
        df_features: DataFrame con features actuales
        df_structures: DataFrame con información de estructuras
        
    Returns:
        DataFrame con features de estructuras agregados
    """
    logger.info("Agregando features de estructuras (torres/inhibidores)...")
    
    # Torres destruidas por equipo
    towers_blue = df_structures[df_structures['Team'] == 'bTowers'].groupby('Address').size().reset_index(name='blue_towers')
    towers_red = df_structures[df_structures['Team'] == 'rTowers'].groupby('Address').size().reset_index(name='red_towers')
    
    # Inhibidores
    inhibs_blue = df_structures[df_structures['Team'] == 'bInhibs'].groupby('Address').size().reset_index(name='blue_inhibs')
    inhibs_red = df_structures[df_structures['Team'] == 'rInhibs'].groupby('Address').size().reset_index(name='red_inhibs')
    
    # Merge
    df_result = df_features.copy()
    df_result = df_result.merge(towers_blue, on='Address', how='left')
    df_result = df_result.merge(towers_red, on='Address', how='left')
    df_result = df_result.merge(inhibs_blue, on='Address', how='left')
    df_result = df_result.merge(inhibs_red, on='Address', how='left')
    
    # Llenar NaN con 0
    df_result['blue_towers'] = df_result['blue_towers'].fillna(0)
    df_result['red_towers'] = df_result['red_towers'].fillna(0)
    df_result['blue_inhibs'] = df_result['blue_inhibs'].fillna(0)
    df_result['red_inhibs'] = df_result['red_inhibs'].fillna(0)
    
    # Features de diferencia
    df_result['tower_diff'] = df_result['blue_towers'] - df_result['red_towers']
    df_result['inhib_diff'] = df_result['blue_inhibs'] - df_result['red_inhibs']
    
    logger.info(f"✓ Features de estructuras agregados")
    
    return df_result


def aggregate_gold_features(
    df_features: pd.DataFrame,
    df_gold: pd.DataFrame
) -> pd.DataFrame:
    """
    Agrega features de diferencia de oro en momentos clave.
    
    Args:
        df_features: DataFrame con features actuales
        df_gold: DataFrame con información de oro
        
    Returns:
        DataFrame con features de oro agregados
    """
    logger.info("Agregando features de diferencia de oro...")
    
    # Obtener diferencia de gold en minutos específicos
    df_gold_features = df_gold[df_gold['Type'] == 'golddiff'].copy()
    
    # Extraer gold difference en minutos 10, 15, 20
    gold_stats = []
    for address in df_gold_features['Address'].unique():
        match_gold = df_gold_features[df_gold_features['Address'] == address]
        
        gold_10 = match_gold['min_10'].values[0] if len(match_gold) > 0 else np.nan
        gold_15 = match_gold['min_15'].values[0] if len(match_gold) > 0 else np.nan
        gold_20 = match_gold['min_20'].values[0] if len(match_gold) > 0 else np.nan
        
        gold_stats.append({
            'Address': address,
            'gold_diff_10': gold_10,
            'gold_diff_15': gold_15,
            'gold_diff_20': gold_20
        })
    
    df_gold_agg = pd.DataFrame(gold_stats)
    
    # Merge con features
    df_result = df_features.merge(df_gold_agg, on='Address', how='left')
    
    # Imputar con mediana
    df_result['gold_diff_10'] = df_result['gold_diff_10'].fillna(df_result['gold_diff_10'].median())
    df_result['gold_diff_15'] = df_result['gold_diff_15'].fillna(df_result['gold_diff_15'].median())
    df_result['gold_diff_20'] = df_result['gold_diff_20'].fillna(df_result['gold_diff_20'].median())
    
    logger.info(f"✓ Features de oro agregados")
    
    return df_result


def select_features(
    df_features: pd.DataFrame,
    parameters: Dict
) -> pd.DataFrame:
    """
    Selecciona features finales para modelado.
    
    Args:
        df_features: DataFrame con todas las features
        parameters: Diccionario con configuración
        
    Returns:
        DataFrame con features seleccionados y targets
    """
    logger.info("Seleccionando features finales...")
    
    feature_cols = parameters['feature_columns']
    target_cols = parameters['target_columns']
    
    # Seleccionar columnas
    df_model = df_features[feature_cols + target_cols].copy()
    
    logger.info(f"✓ Features seleccionados: {len(feature_cols)}")
    logger.info(f"✓ Tamaño del dataset: {df_model.shape}")
    
    return df_model


def split_data(
    df_model: pd.DataFrame,
    parameters: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Divide datos en train/test para regresión y clasificación.
    
    Args:
        df_model: DataFrame con features y targets
        parameters: Diccionario con configuración
        
    Returns:
        Tupla con (X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test)
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("Dividiendo datos en train/test...")
    
    feature_cols = parameters['feature_columns']
    test_size = parameters['test_size']
    random_state = parameters['random_state']
    
    X = df_model[feature_cols].copy()
    y_regression = df_model['gamelength'].copy()
    y_classification = df_model['bResult'].copy()
    
    # División train/test
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=test_size, random_state=random_state
    )
    
    _, _, y_cls_train, y_cls_test = train_test_split(
        X, y_classification, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"✓ Train set: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"✓ Test set: {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Estandariza features usando StandardScaler.
    
    Args:
        X_train: Features de entrenamiento
        X_test: Features de test
        
    Returns:
        Tupla con (X_train_scaled, X_test_scaled, scaler)
    """
    from sklearn.preprocessing import StandardScaler
    
    logger.info("Estandarizando features con StandardScaler...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir de vuelta a DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    logger.info(f"✓ Estandarización completada")
    logger.info(f"   Media después: {X_train_scaled.mean().mean():.6f}")
    logger.info(f"   Std después: {X_train_scaled.std().mean():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler


