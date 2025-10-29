"""
Nodos del Pipeline de Limpieza de Datos

Este módulo contiene todas las funciones (nodos) para limpiar los datos raw.
Cada función transforma un dataset específico.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


# ============================================================================
# NODO 1: Limpieza de LeagueofLegends.csv (Dataset Principal)
# ============================================================================

def clean_main_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset principal de League of Legends.
    
    Args:
        df: DataFrame raw de LeagueofLegends.csv
        
    Returns:
        DataFrame limpio y procesado
        
    Transformaciones:
        - Elimina duplicados
        - Maneja valores faltantes
        - Convierte tipos de datos
        - Estandariza nombres de columnas
        - Elimina outliers extremos
    """
    logger.info(f"Iniciando limpieza del dataset principal: {len(df)} filas")
    
    # Crear copia para no modificar el original
    df_clean = df.copy()
    
    # 1. Estandarizar nombres de columnas (minúsculas y sin espacios)
    df_clean.columns = df_clean.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # 2. Eliminar duplicados completos
    duplicados_antes = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicados_eliminados = duplicados_antes - len(df_clean)
    if duplicados_eliminados > 0:
        logger.info(f"Eliminados {duplicados_eliminados} duplicados completos")
    
    # 3. Manejar valores faltantes
    missing_antes = df_clean.isnull().sum().sum()
    
    # Para columnas numéricas: imputar con mediana
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            mediana = df_clean[col].median()
            df_clean[col].fillna(mediana, inplace=True)
            logger.info(f"Columna '{col}': imputados valores faltantes con mediana {mediana:.2f}")
    
    # Para columnas categóricas: imputar con moda o 'Unknown'
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            if df_clean[col].mode().empty:
                df_clean[col].fillna('Unknown', inplace=True)
            else:
                moda = df_clean[col].mode().iloc[0]
                df_clean[col].fillna(moda, inplace=True)
            logger.info(f"Columna '{col}': imputados valores faltantes")
    
    missing_despues = df_clean.isnull().sum().sum()
    logger.info(f"Valores faltantes: {missing_antes} → {missing_despues}")
    
    # 4. Convertir tipos de datos apropiados
    # Convertir gameid a string si existe
    if 'gameid' in df_clean.columns:
        df_clean['gameid'] = df_clean['gameid'].astype(str)
    
    # Convertir result a binario si existe (Win=1, Loss=0)
    if 'result' in df_clean.columns:
        df_clean['result_binary'] = (df_clean['result'].str.lower() == 'win').astype(int)
    
    # Convertir gamelength de segundos a minutos si existe
    if 'gamelength' in df_clean.columns:
        df_clean['gamelength_minutes'] = df_clean['gamelength'] / 60
    
    # 5. Eliminar outliers extremos usando IQR para columnas numéricas clave
    outliers_eliminados = 0
    columnas_para_outliers = ['kills', 'deaths', 'assists', 'gold'] if all(
        col in df_clean.columns for col in ['kills', 'deaths', 'assists', 'gold']
    ) else []
    
    for col in columnas_para_outliers:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 3 IQR para outliers extremos
        upper_bound = Q3 + 3 * IQR
        
        antes = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        despues = len(df_clean)
        outliers_col = antes - despues
        
        if outliers_col > 0:
            outliers_eliminados += outliers_col
            logger.info(f"Columna '{col}': eliminados {outliers_col} outliers extremos")
    
    # 6. Crear features derivadas básicas
    if all(col in df_clean.columns for col in ['kills', 'deaths', 'assists']):
        # KDA (Kill-Death-Assist ratio)
        df_clean['kda'] = df_clean.apply(
            lambda row: (row['kills'] + row['assists']) if row['deaths'] == 0 
            else (row['kills'] + row['assists']) / row['deaths'],
            axis=1
        )
        logger.info("Creada feature 'kda'")
    
    logger.info(f"Limpieza completada: {len(df_clean)} filas finales")
    logger.info(f"Resumen: {duplicados_eliminados} duplicados, "
                f"{missing_antes - missing_despues} valores faltantes imputados, "
                f"{outliers_eliminados} outliers eliminados")
    
    return df_clean


# ============================================================================
# NODO 2: Limpieza de matchinfo.csv
# ============================================================================

def clean_matchinfo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de información de partidos.
    
    Args:
        df: DataFrame raw de matchinfo.csv
        
    Returns:
        DataFrame limpio con información de partidos
    """
    logger.info(f"Iniciando limpieza de matchinfo: {len(df)} filas")
    
    df_clean = df.copy()
    
    # Estandarizar nombres de columnas
    df_clean.columns = df_clean.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Eliminar duplicados
    df_clean = df_clean.drop_duplicates()
    
    # Convertir gameid a string
    if 'gameid' in df_clean.columns:
        df_clean['gameid'] = df_clean['gameid'].astype(str)
    
    # Convertir fecha a datetime si existe
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        # Extraer año, mes, día de la semana
        df_clean['year'] = df_clean['date'].dt.year
        df_clean['month'] = df_clean['date'].dt.month
        df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
        logger.info("Convertida columna 'date' a datetime y extraídas features temporales")
    
    # Limpiar nombres de ligas
    if 'league' in df_clean.columns:
        df_clean['league'] = df_clean['league'].str.strip().str.upper()
    
    # Limpiar split/season
    if 'split' in df_clean.columns:
        df_clean['split'] = df_clean['split'].str.strip()
    
    logger.info(f"Limpieza de matchinfo completada: {len(df_clean)} filas")
    
    return df_clean


# ============================================================================
# NODO 3: Limpieza de bans.csv
# ============================================================================

def clean_bans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de bans de campeones.
    
    Args:
        df: DataFrame raw de bans.csv
        
    Returns:
        DataFrame limpio con información de bans
    """
    logger.info(f"Iniciando limpieza de bans: {len(df)} filas")
    
    df_clean = df.copy()
    
    # Estandarizar nombres de columnas
    df_clean.columns = df_clean.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Eliminar duplicados
    df_clean = df_clean.drop_duplicates()
    
    # Convertir gameid a string
    if 'gameid' in df_clean.columns:
        df_clean['gameid'] = df_clean['gameid'].astype(str)
    
    # Limpiar nombres de campeones
    if 'champion' in df_clean.columns:
        df_clean['champion'] = df_clean['champion'].str.strip().str.title()
        # Eliminar filas donde el campeón es nulo o vacío
        df_clean = df_clean[df_clean['champion'].notna()]
        df_clean = df_clean[df_clean['champion'] != '']
        logger.info("Limpiados nombres de campeones")
    
    # Estandarizar nombres de equipos
    if 'team' in df_clean.columns:
        df_clean['team'] = df_clean['team'].str.strip().str.title()
    
    # Validar orden de ban (debe ser 1-5)
    if 'ban' in df_clean.columns:
        df_clean['ban'] = pd.to_numeric(df_clean['ban'], errors='coerce')
        df_clean = df_clean[(df_clean['ban'] >= 1) & (df_clean['ban'] <= 5)]
    
    logger.info(f"Limpieza de bans completada: {len(df_clean)} filas")
    
    return df_clean


# ============================================================================
# NODO 4: Limpieza de gold.csv
# ============================================================================

def clean_gold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de estadísticas de oro.
    
    Args:
        df: DataFrame raw de gold.csv
        
    Returns:
        DataFrame limpio con estadísticas de oro
    """
    logger.info(f"Iniciando limpieza de gold: {len(df)} filas")
    
    df_clean = df.copy()
    
    # Estandarizar nombres de columnas
    df_clean.columns = df_clean.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Eliminar duplicados
    df_clean = df_clean.drop_duplicates()
    
    # Convertir gameid a string
    if 'gameid' in df_clean.columns:
        df_clean['gameid'] = df_clean['gameid'].astype(str)
    
    # Validar que valores de oro sean positivos
    gold_columns = [col for col in df_clean.columns if 'gold' in col.lower()]
    for col in gold_columns:
        if df_clean[col].dtype in [np.int64, np.float64]:
            # Reemplazar valores negativos con 0
            df_clean[col] = df_clean[col].clip(lower=0)
    
    # Calcular diferencia de oro entre equipos si existen columnas apropiadas
    if 'goldblue' in df_clean.columns and 'goldred' in df_clean.columns:
        df_clean['gold_diff'] = df_clean['goldblue'] - df_clean['goldred']
        logger.info("Calculada diferencia de oro entre equipos")
    
    logger.info(f"Limpieza de gold completada: {len(df_clean)} filas")
    
    return df_clean


# ============================================================================
# NODO 5: Limpieza de kills.csv
# ============================================================================

def clean_kills(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de kills.
    
    Args:
        df: DataFrame raw de kills.csv
        
    Returns:
        DataFrame limpio con información de kills
    """
    logger.info(f"Iniciando limpieza de kills: {len(df)} filas")
    
    df_clean = df.copy()
    
    # Estandarizar nombres de columnas
    df_clean.columns = df_clean.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Eliminar duplicados
    df_clean = df_clean.drop_duplicates()
    
    # Convertir gameid a string
    if 'gameid' in df_clean.columns:
        df_clean['gameid'] = df_clean['gameid'].astype(str)
    
    # Validar que time sea positivo
    if 'time' in df_clean.columns:
        df_clean['time'] = pd.to_numeric(df_clean['time'], errors='coerce')
        df_clean = df_clean[df_clean['time'] >= 0]
    
    # Limpiar nombres de jugadores
    for col in ['killer', 'victim']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].str.strip()
    
    logger.info(f"Limpieza de kills completada: {len(df_clean)} filas")
    
    return df_clean


# ============================================================================
# NODO 6: Limpieza de monsters.csv
# ============================================================================

def clean_monsters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de objetivos neutrales (dragones, baron, herald).
    
    Args:
        df: DataFrame raw de monsters.csv
        
    Returns:
        DataFrame limpio con información de objetivos
    """
    logger.info(f"Iniciando limpieza de monsters: {len(df)} filas")
    
    df_clean = df.copy()
    
    # Estandarizar nombres de columnas
    df_clean.columns = df_clean.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Eliminar duplicados
    df_clean = df_clean.drop_duplicates()
    
    # Convertir gameid a string
    if 'gameid' in df_clean.columns:
        df_clean['gameid'] = df_clean['gameid'].astype(str)
    
    # Validar que time sea positivo
    if 'time' in df_clean.columns:
        df_clean['time'] = pd.to_numeric(df_clean['time'], errors='coerce')
        df_clean = df_clean[df_clean['time'] >= 0]
    
    # Estandarizar tipo de monstruo
    if 'type' in df_clean.columns:
        df_clean['type'] = df_clean['type'].str.strip().str.lower()
    
    # Estandarizar nombre de equipo
    if 'team' in df_clean.columns:
        df_clean['team'] = df_clean['team'].str.strip().str.title()
    
    logger.info(f"Limpieza de monsters completada: {len(df_clean)} filas")
    
    return df_clean


# ============================================================================
# NODO 7: Limpieza de structures.csv
# ============================================================================

def clean_structures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de estructuras destruidas (torres, inhibidores).
    
    Args:
        df: DataFrame raw de structures.csv
        
    Returns:
        DataFrame limpio con información de estructuras
    """
    logger.info(f"Iniciando limpieza de structures: {len(df)} filas")
    
    df_clean = df.copy()
    
    # Estandarizar nombres de columnas
    df_clean.columns = df_clean.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Eliminar duplicados
    df_clean = df_clean.drop_duplicates()
    
    # Convertir gameid a string
    if 'gameid' in df_clean.columns:
        df_clean['gameid'] = df_clean['gameid'].astype(str)
    
    # Validar que time sea positivo
    if 'time' in df_clean.columns:
        df_clean['time'] = pd.to_numeric(df_clean['time'], errors='coerce')
        df_clean = df_clean[df_clean['time'] >= 0]
    
    # Estandarizar tipo de estructura
    if 'type' in df_clean.columns:
        df_clean['type'] = df_clean['type'].str.strip().str.lower()
    
    # Estandarizar lane
    if 'lane' in df_clean.columns:
        df_clean['lane'] = df_clean['lane'].str.strip().str.lower()
    
    # Estandarizar nombre de equipo
    if 'team' in df_clean.columns:
        df_clean['team'] = df_clean['team'].str.strip().str.title()
    
    logger.info(f"Limpieza de structures completada: {len(df_clean)} filas")
    
    return df_clean


# ============================================================================
# NODO 8: Generar Reporte de Calidad de Datos
# ============================================================================

def generate_data_quality_report(
    main_df: pd.DataFrame,
    matchinfo_df: pd.DataFrame,
    bans_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    kills_df: pd.DataFrame,
    monsters_df: pd.DataFrame,
    structures_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Genera un reporte de calidad de datos después de la limpieza.
    
    Args:
        *_df: DataFrames limpios
        
    Returns:
        DataFrame con métricas de calidad por dataset
    """
    logger.info("Generando reporte de calidad de datos")
    
    datasets = {
        'main': main_df,
        'matchinfo': matchinfo_df,
        'bans': bans_df,
        'gold': gold_df,
        'kills': kills_df,
        'monsters': monsters_df,
        'structures': structures_df
    }
    
    report_data = []
    
    for name, df in datasets.items():
        report = {
            'dataset': name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'missing_values': int(df.isnull().sum().sum()),
            'missing_percentage': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            'duplicate_rows': int(df.duplicated().sum()),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)
        }
        
        report_data.append(report)
        logger.info(f"Dataset '{name}': {report['total_rows']} filas, "
                   f"{report['missing_values']} valores faltantes, "
                   f"{report['duplicate_rows']} duplicados")
    
    report_df = pd.DataFrame(report_data)
    
    logger.info("Reporte de calidad generado exitosamente")
    
    return report_df

