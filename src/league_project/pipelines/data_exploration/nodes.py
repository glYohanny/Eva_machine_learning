"""
Nodos del Pipeline de Exploración de Datos

Este módulo contiene funciones para análisis exploratorio de datos (EDA).
Genera estadísticas, visualizaciones y reportes de insights.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# NODO 1: Estadísticas Descriptivas del Dataset Principal
# ============================================================================

def generate_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera estadísticas descriptivas completas del dataset principal.
    
    Args:
        df: DataFrame limpio del dataset principal
        
    Returns:
        DataFrame con estadísticas descriptivas
    """
    logger.info("Generando estadísticas descriptivas")
    
    # Estadísticas básicas
    stats = df.describe().transpose()
    
    # Agregar estadísticas adicionales
    stats['median'] = df.median(numeric_only=True)
    stats['mode'] = df.mode(numeric_only=True).iloc[0] if len(df.mode(numeric_only=True)) > 0 else np.nan
    stats['skewness'] = df.skew(numeric_only=True)
    stats['kurtosis'] = df.kurtosis(numeric_only=True)
    stats['missing_count'] = df.isnull().sum()
    stats['missing_percentage'] = (df.isnull().sum() / len(df)) * 100
    
    # Redondear valores
    stats = stats.round(2)
    
    logger.info(f"Estadísticas generadas para {len(stats)} columnas")
    
    return stats


# ============================================================================
# NODO 2: Análisis de Win Rate por Equipo
# ============================================================================

def analyze_team_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza el rendimiento de equipos (win rate, estadísticas promedio).
    
    Args:
        df: DataFrame limpio con datos de partidos
        
    Returns:
        DataFrame con análisis de rendimiento por equipo
    """
    logger.info("Analizando rendimiento de equipos")
    
    team_stats = []
    
    # Procesar equipos azules
    if 'blueteamtag' in df.columns and 'bresult' in df.columns:
        blue_teams = df.groupby('blueteamtag').agg({
            'bresult': ['count', 'sum', 'mean'],
            'gamelength_minutes': 'mean'
        }).reset_index()
        
        blue_teams.columns = ['team', 'total_games', 'wins', 'win_rate', 'avg_game_length']
        
        for _, row in blue_teams.iterrows():
            team_stats.append({
                'team': row['team'],
                'total_games': int(row['total_games']),
                'wins': int(row['wins']),
                'losses': int(row['total_games'] - row['wins']),
                'win_rate': float(row['win_rate']),
                'avg_game_length': float(row['avg_game_length'])
            })
    
    # Procesar equipos rojos
    if 'redteamtag' in df.columns and 'rresult' in df.columns:
        red_teams = df.groupby('redteamtag').agg({
            'rresult': ['count', 'sum', 'mean'],
            'gamelength_minutes': 'mean'
        }).reset_index()
        
        red_teams.columns = ['team', 'total_games', 'wins', 'win_rate', 'avg_game_length']
        
        for _, row in red_teams.iterrows():
            # Buscar si el equipo ya existe (jugó en azul)
            existing_team = next((t for t in team_stats if t['team'] == row['team']), None)
            
            if existing_team:
                # Combinar estadísticas
                total_games = existing_team['total_games'] + int(row['total_games'])
                total_wins = existing_team['wins'] + int(row['wins'])
                existing_team['total_games'] = total_games
                existing_team['wins'] = total_wins
                existing_team['losses'] = total_games - total_wins
                existing_team['win_rate'] = total_wins / total_games
                existing_team['avg_game_length'] = (
                    (existing_team['avg_game_length'] * (existing_team['total_games'] - int(row['total_games'])) +
                     row['avg_game_length'] * int(row['total_games'])) / total_games
                )
            else:
                team_stats.append({
                    'team': row['team'],
                    'total_games': int(row['total_games']),
                    'wins': int(row['wins']),
                    'losses': int(row['total_games'] - row['wins']),
                    'win_rate': float(row['win_rate']),
                    'avg_game_length': float(row['avg_game_length'])
                })
    
    result_df = pd.DataFrame(team_stats)
    
    # Ordenar por win rate
    if 'win_rate' in result_df.columns and len(result_df) > 0:
        result_df = result_df.sort_values('win_rate', ascending=False)
        # Redondear valores
        result_df['win_rate'] = result_df['win_rate'].round(3)
        result_df['avg_game_length'] = result_df['avg_game_length'].round(2)
    
    logger.info(f"Análisis completado para {len(result_df)} equipos")
    
    return result_df


# ============================================================================
# NODO 3: Análisis de Campeones Más Baneados
# ============================================================================

def analyze_champion_bans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza los campeones más baneados y su frecuencia.
    
    Args:
        df: DataFrame limpio de bans
        
    Returns:
        DataFrame con análisis de bans por campeón
    """
    logger.info("Analizando bans de campeones")
    
    # El dataset de bans tiene columnas ban_1, ban_2, ban_3, ban_4, ban_5
    ban_columns = ['ban_1', 'ban_2', 'ban_3', 'ban_4', 'ban_5']
    
    # Verificar que existan las columnas
    available_ban_cols = [col for col in ban_columns if col in df.columns]
    
    if not available_ban_cols:
        logger.warning("No se encontraron columnas de bans (ban_1, ban_2, etc.)")
        return pd.DataFrame()
    
    # Combinar todos los bans en una lista
    all_bans = []
    for col in available_ban_cols:
        bans = df[col].dropna()
        bans = bans[bans != '']  # Eliminar valores vacíos
        all_bans.extend(bans.tolist())
    
    if not all_bans:
        logger.warning("No se encontraron bans en los datos")
        return pd.DataFrame()
    
    # Crear DataFrame con todos los bans
    bans_series = pd.Series(all_bans)
    
    # Contar bans por campeón
    ban_counts = bans_series.value_counts().reset_index()
    ban_counts.columns = ['champion', 'ban_count']
    
    # Calcular porcentaje
    total_bans = len(bans_series)
    ban_counts['ban_percentage'] = (ban_counts['ban_count'] / total_bans * 100).round(2)
    
    # Calcular en cuántos juegos únicos fue baneado
    games_banned = []
    for champion in ban_counts['champion']:
        games = 0
        for col in available_ban_cols:
            games += (df[col] == champion).sum()
        games_banned.append(games)
    
    ban_counts['games_banned'] = games_banned
    
    # Calcular prioridad promedio de ban (1 = primer ban, 5 = último ban)
    ban_priorities = []
    for champion in ban_counts['champion']:
        priorities = []
        for i, col in enumerate(available_ban_cols, 1):
            if champion in df[col].values:
                count = (df[col] == champion).sum()
                priorities.extend([i] * count)
        if priorities:
            ban_priorities.append(sum(priorities) / len(priorities))
        else:
            ban_priorities.append(0)
    
    ban_counts['avg_ban_priority'] = [round(p, 2) for p in ban_priorities]
    
    logger.info(f"Análisis de bans completado para {len(ban_counts)} campeones")
    
    return ban_counts


# ============================================================================
# NODO 4: Análisis de Objetivos Neutrales
# ============================================================================

def analyze_neutral_objectives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza la captura de objetivos neutrales (dragones, baron, herald).
    
    Args:
        df: DataFrame limpio de monsters
        
    Returns:
        DataFrame con análisis de objetivos
    """
    logger.info("Analizando objetivos neutrales")
    
    if 'type' not in df.columns:
        logger.warning("Columna 'type' no encontrada")
        return pd.DataFrame()
    
    # Contar por tipo de objetivo
    objective_counts = df['type'].value_counts().reset_index()
    objective_counts.columns = ['objective_type', 'total_captured']
    
    # Tiempo promedio de captura
    if 'time' in df.columns:
        avg_time = df.groupby('type')['time'].mean().reset_index()
        avg_time.columns = ['objective_type', 'avg_capture_time_seconds']
        objective_counts = objective_counts.merge(avg_time, on='objective_type', how='left')
        objective_counts['avg_capture_time_minutes'] = (
            objective_counts['avg_capture_time_seconds'] / 60
        ).round(2)
    
    # Captura por equipo
    if 'team' in df.columns:
        team_captures = df.groupby(['type', 'team']).size().reset_index(name='captures')
        # Pivotear para tener equipos como columnas
        team_pivot = team_captures.pivot(
            index='type', columns='team', values='captures'
        ).reset_index()
        team_pivot.columns.name = None
        objective_counts = objective_counts.merge(
            team_pivot, left_on='objective_type', right_on='type', how='left'
        )
        if 'type' in objective_counts.columns:
            objective_counts = objective_counts.drop('type', axis=1)
    
    logger.info(f"Análisis de objetivos completado para {len(objective_counts)} tipos")
    
    return objective_counts


# ============================================================================
# NODO 5: Análisis de Estructuras Destruidas
# ============================================================================

def analyze_structures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza la destrucción de estructuras (torres, inhibidores).
    
    Args:
        df: DataFrame limpio de structures
        
    Returns:
        DataFrame con análisis de estructuras
    """
    logger.info("Analizando destrucción de estructuras")
    
    if 'type' not in df.columns:
        logger.warning("Columna 'type' no encontrada")
        return pd.DataFrame()
    
    # Contar por tipo de estructura
    structure_counts = df['type'].value_counts().reset_index()
    structure_counts.columns = ['structure_type', 'total_destroyed']
    
    # Tiempo promedio de destrucción
    if 'time' in df.columns:
        avg_time = df.groupby('type')['time'].mean().reset_index()
        avg_time.columns = ['structure_type', 'avg_destruction_time_seconds']
        structure_counts = structure_counts.merge(avg_time, on='structure_type', how='left')
        structure_counts['avg_destruction_time_minutes'] = (
            structure_counts['avg_destruction_time_seconds'] / 60
        ).round(2)
    
    # Destrucción por lane
    if 'lane' in df.columns:
        lane_counts = df.groupby(['type', 'lane']).size().reset_index(name='count')
        lane_pivot = lane_counts.pivot(
            index='type', columns='lane', values='count'
        ).reset_index()
        lane_pivot.columns.name = None
        structure_counts = structure_counts.merge(
            lane_pivot, left_on='structure_type', right_on='type', how='left'
        )
        if 'type' in structure_counts.columns:
            structure_counts = structure_counts.drop('type', axis=1)
    
    logger.info(f"Análisis de estructuras completado para {len(structure_counts)} tipos")
    
    return structure_counts


# ============================================================================
# NODO 6: Análisis de Correlaciones
# ============================================================================

def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza correlaciones entre variables numéricas.
    
    Args:
        df: DataFrame limpio del dataset principal
        
    Returns:
        DataFrame con matriz de correlación
    """
    logger.info("Analizando correlaciones")
    
    # Seleccionar solo columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calcular matriz de correlación
    corr_matrix = numeric_df.corr()
    
    # Convertir a formato largo para mejor análisis
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'variable_1': corr_matrix.columns[i],
                'variable_2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j],
                'abs_correlation': abs(corr_matrix.iloc[i, j])
            })
    
    corr_df = pd.DataFrame(corr_pairs)
    
    # Ordenar por correlación absoluta
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    # Clasificar fuerza de correlación
    def classify_correlation(corr):
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            return 'Fuerte'
        elif abs_corr >= 0.4:
            return 'Moderada'
        elif abs_corr >= 0.2:
            return 'Débil'
        else:
            return 'Muy Débil'
    
    corr_df['strength'] = corr_df['correlation'].apply(classify_correlation)
    corr_df['correlation'] = corr_df['correlation'].round(3)
    
    logger.info(f"Análisis de correlaciones completado: {len(corr_df)} pares analizados")
    
    return corr_df


# ============================================================================
# NODO 7: Análisis de Duración de Partidos
# ============================================================================

def analyze_game_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza la duración de los partidos y su distribución.
    
    Args:
        df: DataFrame limpio del dataset principal
        
    Returns:
        DataFrame con análisis de duración
    """
    logger.info("Analizando duración de partidos")
    
    duration_col = None
    if 'gamelength_minutes' in df.columns:
        duration_col = 'gamelength_minutes'
    elif 'gamelength' in df.columns:
        df['gamelength_minutes'] = df['gamelength'] / 60
        duration_col = 'gamelength_minutes'
    
    if duration_col is None:
        logger.warning("No se encontró columna de duración")
        return pd.DataFrame()
    
    # Estadísticas descriptivas
    stats = {
        'metric': ['count', 'mean', 'median', 'std', 'min', 'max', 'q25', 'q75'],
        'value': [
            len(df[duration_col]),
            df[duration_col].mean(),
            df[duration_col].median(),
            df[duration_col].std(),
            df[duration_col].min(),
            df[duration_col].max(),
            df[duration_col].quantile(0.25),
            df[duration_col].quantile(0.75),
        ]
    }
    
    stats_df = pd.DataFrame(stats)
    stats_df['value'] = stats_df['value'].round(2)
    
    # Categorizar duración
    bins = [0, 25, 35, 45, float('inf')]
    labels = ['Corta (<25 min)', 'Media (25-35 min)', 'Larga (35-45 min)', 'Muy Larga (>45 min)']
    df['duration_category'] = pd.cut(df[duration_col], bins=bins, labels=labels)
    
    # Contar por categoría
    category_counts = df['duration_category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']
    category_counts['percentage'] = (category_counts['count'] / len(df) * 100).round(2)
    
    # Combinar estadísticas
    stats_df['category'] = 'Estadísticas Generales'
    category_counts['metric'] = category_counts['category']
    category_counts['value'] = category_counts['percentage']
    category_counts = category_counts[['metric', 'value', 'category']]
    
    logger.info(f"Análisis de duración completado")
    
    return stats_df


# ============================================================================
# NODO 8: Generar Reporte de EDA Completo
# ============================================================================

def generate_eda_report(
    descriptive_stats: pd.DataFrame,
    team_performance: pd.DataFrame,
    champion_bans: pd.DataFrame,
    neutral_objectives: pd.DataFrame,
    structures_analysis: pd.DataFrame,
    correlations: pd.DataFrame,
    game_duration: pd.DataFrame
) -> Dict[str, Any]:
    """
    Genera un reporte completo de EDA combinando todos los análisis.
    
    Args:
        Todos los DataFrames de análisis previos
        
    Returns:
        Diccionario con resumen del reporte
    """
    logger.info("Generando reporte completo de EDA")
    
    report = {
        'total_analyses': 7,
        'descriptive_statistics_columns': len(descriptive_stats) if not descriptive_stats.empty else 0,
        'teams_analyzed': len(team_performance) if not team_performance.empty else 0,
        'champions_analyzed': len(champion_bans) if not champion_bans.empty else 0,
        'objective_types': len(neutral_objectives) if not neutral_objectives.empty else 0,
        'structure_types': len(structures_analysis) if not structures_analysis.empty else 0,
        'correlation_pairs': len(correlations) if not correlations.empty else 0,
        'duration_metrics': len(game_duration) if not game_duration.empty else 0,
    }
    
    # Top insights
    insights = []
    
    if not team_performance.empty and 'win_rate' in team_performance.columns:
        top_team = team_performance.iloc[0]
        insights.append(f"Mejor equipo: {top_team['team']} con {top_team['win_rate']:.2%} win rate")
    
    if not champion_bans.empty:
        top_ban = champion_bans.iloc[0]
        insights.append(f"Campeón más baneado: {top_ban['champion']} ({top_ban['ban_count']} bans)")
    
    if not correlations.empty:
        strongest_corr = correlations.iloc[0]
        insights.append(
            f"Correlación más fuerte: {strongest_corr['variable_1']} - "
            f"{strongest_corr['variable_2']} ({strongest_corr['correlation']:.3f})"
        )
    
    report['key_insights'] = insights
    
    logger.info("Reporte de EDA generado exitosamente")
    logger.info(f"Insights clave: {len(insights)}")
    
    return report

