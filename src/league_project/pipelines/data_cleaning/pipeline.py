"""
Pipeline de Limpieza de Datos

Este pipeline orquesta todos los nodos de limpieza de datos.
Transforma los datos raw en datos limpios y listos para análisis.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    clean_main_dataset,
    clean_matchinfo,
    clean_bans,
    clean_gold,
    clean_kills,
    clean_monsters,
    clean_structures,
    generate_data_quality_report
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de limpieza de datos.
    
    Este pipeline:
    1. Limpia cada dataset raw independientemente
    2. Genera un reporte de calidad de datos
    3. Guarda los datos limpios en intermediate
    
    Returns:
        Pipeline de Kedro con todos los nodos de limpieza
    """
    return pipeline(
        [
            # ================================================================
            # NODO 1: Limpiar dataset principal
            # ================================================================
            node(
                func=clean_main_dataset,
                inputs="raw_main_data",  # data/01_raw/LeagueofLegends.csv
                outputs="intermediate_main_data",  # data/02_intermediate/main_clean.csv
                name="clean_main_dataset_node",
                tags=["cleaning", "main"],
            ),
            
            # ================================================================
            # NODO 2: Limpiar matchinfo
            # ================================================================
            node(
                func=clean_matchinfo,
                inputs="raw_matchinfo",  # data/01_raw/matchinfo.csv
                outputs="intermediate_matchinfo",  # data/02_intermediate/matchinfo_clean.csv
                name="clean_matchinfo_node",
                tags=["cleaning", "matchinfo"],
            ),
            
            # ================================================================
            # NODO 3: Limpiar bans
            # ================================================================
            node(
                func=clean_bans,
                inputs="raw_bans",  # data/01_raw/bans.csv
                outputs="intermediate_bans",  # data/02_intermediate/bans_clean.csv
                name="clean_bans_node",
                tags=["cleaning", "bans"],
            ),
            
            # ================================================================
            # NODO 4: Limpiar gold
            # ================================================================
            node(
                func=clean_gold,
                inputs="raw_gold",  # data/01_raw/gold.csv
                outputs="intermediate_gold",  # data/02_intermediate/gold_clean.csv
                name="clean_gold_node",
                tags=["cleaning", "gold"],
            ),
            
            # ================================================================
            # NODO 5: Limpiar kills
            # ================================================================
            node(
                func=clean_kills,
                inputs="raw_kills",  # data/01_raw/kills.csv
                outputs="intermediate_kills",  # data/02_intermediate/kills_clean.csv
                name="clean_kills_node",
                tags=["cleaning", "kills"],
            ),
            
            # ================================================================
            # NODO 6: Limpiar monsters
            # ================================================================
            node(
                func=clean_monsters,
                inputs="raw_monsters",  # data/01_raw/monsters.csv
                outputs="intermediate_monsters",  # data/02_intermediate/monsters_clean.csv
                name="clean_monsters_node",
                tags=["cleaning", "monsters"],
            ),
            
            # ================================================================
            # NODO 7: Limpiar structures
            # ================================================================
            node(
                func=clean_structures,
                inputs="raw_structures",  # data/01_raw/structures.csv
                outputs="intermediate_structures",  # data/02_intermediate/structures_clean.csv
                name="clean_structures_node",
                tags=["cleaning", "structures"],
            ),
            
            # ================================================================
            # NODO 8: Generar reporte de calidad
            # ================================================================
            node(
                func=generate_data_quality_report,
                inputs=[
                    "intermediate_main_data",
                    "intermediate_matchinfo",
                    "intermediate_bans",
                    "intermediate_gold",
                    "intermediate_kills",
                    "intermediate_monsters",
                    "intermediate_structures",
                ],
                outputs="data_quality_report",  # data/08_reporting/data_quality_report_cleaning.csv
                name="generate_quality_report_node",
                tags=["reporting", "quality"],
            ),
        ],
        tags=["data_cleaning_pipeline"]
    )

