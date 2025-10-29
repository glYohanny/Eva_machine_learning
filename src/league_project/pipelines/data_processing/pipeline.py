"""
Pipeline de procesamiento de datos.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    aggregate_kills_features,
    aggregate_monsters_features,
    aggregate_structures_features,
    aggregate_gold_features,
    select_features,
    split_data,
    scale_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de procesamiento de datos.
    
    Returns:
        Pipeline de Kedro con todos los nodos de procesamiento
    """
    return pipeline(
        [
            node(
                func=aggregate_kills_features,
                inputs=["matchinfo", "kills"],
                outputs="features_with_kills",
                name="aggregate_kills_node",
            ),
            node(
                func=aggregate_monsters_features,
                inputs=["features_with_kills", "monsters"],
                outputs="features_with_monsters",
                name="aggregate_monsters_node",
            ),
            node(
                func=aggregate_structures_features,
                inputs=["features_with_monsters", "structures"],
                outputs="features_with_structures",
                name="aggregate_structures_node",
            ),
            node(
                func=aggregate_gold_features,
                inputs=["features_with_structures", "gold"],
                outputs="features_complete",
                name="aggregate_gold_node",
            ),
            node(
                func=select_features,
                inputs=["features_complete", "params:model_options"],
                outputs="model_input_table",
                name="select_features_node",
            ),
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train", "X_test", "y_reg_train", "y_reg_test", "y_cls_train", "y_cls_test"],
                name="split_data_node",
            ),
            node(
                func=scale_features,
                inputs=["X_train", "X_test"],
                outputs=["X_train_scaled", "X_test_scaled", "scaler"],
                name="scale_features_node",
            ),
        ]
    )




