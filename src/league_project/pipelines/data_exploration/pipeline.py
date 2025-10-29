"""
Pipeline de Exploración de Datos

Este pipeline orquesta todos los nodos de análisis exploratorio de datos (EDA).
Genera estadísticas, análisis y reportes de insights.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    generate_descriptive_statistics,
    analyze_team_performance,
    analyze_champion_bans,
    analyze_neutral_objectives,
    analyze_structures,
    analyze_correlations,
    analyze_game_duration,
    generate_eda_report
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de exploración de datos.
    
    Este pipeline:
    1. Genera estadísticas descriptivas
    2. Analiza rendimiento de equipos
    3. Analiza bans de campeones
    4. Analiza objetivos neutrales
    5. Analiza destrucción de estructuras
    6. Analiza correlaciones
    7. Analiza duración de partidos
    8. Genera reporte completo de EDA
    
    Returns:
        Pipeline de Kedro con todos los nodos de exploración
    """
    return pipeline(
        [
            # ================================================================
            # NODO 1: Estadísticas descriptivas
            # ================================================================
            node(
                func=generate_descriptive_statistics,
                inputs="intermediate_main_data",
                outputs="descriptive_statistics",
                name="generate_descriptive_stats_node",
                tags=["eda", "statistics"],
            ),
            
            # ================================================================
            # NODO 2: Análisis de rendimiento de equipos
            # ================================================================
            node(
                func=analyze_team_performance,
                inputs="intermediate_main_data",
                outputs="team_performance_analysis",
                name="analyze_team_performance_node",
                tags=["eda", "teams"],
            ),
            
            # ================================================================
            # NODO 3: Análisis de bans de campeones
            # ================================================================
            node(
                func=analyze_champion_bans,
                inputs="intermediate_bans",
                outputs="champion_bans_analysis",
                name="analyze_champion_bans_node",
                tags=["eda", "champions"],
            ),
            
            # ================================================================
            # NODO 4: Análisis de objetivos neutrales
            # ================================================================
            node(
                func=analyze_neutral_objectives,
                inputs="intermediate_monsters",
                outputs="neutral_objectives_analysis",
                name="analyze_neutral_objectives_node",
                tags=["eda", "objectives"],
            ),
            
            # ================================================================
            # NODO 5: Análisis de estructuras
            # ================================================================
            node(
                func=analyze_structures,
                inputs="intermediate_structures",
                outputs="structures_analysis",
                name="analyze_structures_node",
                tags=["eda", "structures"],
            ),
            
            # ================================================================
            # NODO 6: Análisis de correlaciones
            # ================================================================
            node(
                func=analyze_correlations,
                inputs="intermediate_main_data",
                outputs="correlations_analysis",
                name="analyze_correlations_node",
                tags=["eda", "correlations"],
            ),
            
            # ================================================================
            # NODO 7: Análisis de duración de partidos
            # ================================================================
            node(
                func=analyze_game_duration,
                inputs="intermediate_main_data",
                outputs="game_duration_analysis",
                name="analyze_game_duration_node",
                tags=["eda", "duration"],
            ),
            
            # ================================================================
            # NODO 8: Generar reporte completo de EDA
            # ================================================================
            node(
                func=generate_eda_report,
                inputs=[
                    "descriptive_statistics",
                    "team_performance_analysis",
                    "champion_bans_analysis",
                    "neutral_objectives_analysis",
                    "structures_analysis",
                    "correlations_analysis",
                    "game_duration_analysis",
                ],
                outputs="eda_complete_report",
                name="generate_eda_report_node",
                tags=["eda", "reporting"],
            ),
        ],
        tags=["data_exploration_pipeline"]
    )

