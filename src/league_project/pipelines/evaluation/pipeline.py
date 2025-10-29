"""
Pipeline de evaluación.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    evaluate_regression_models,
    evaluate_classification_models,
    get_feature_importance,
    create_regression_report,
    create_classification_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de evaluación.
    
    Returns:
        Pipeline de Kedro con evaluación de modelos
    """
    return pipeline(
        [
            # Evaluación de Regresión
            node(
                func=evaluate_regression_models,
                inputs=["regression_predictions", "y_reg_train", "y_reg_test"],
                outputs="regression_metrics",
                name="evaluate_regression_node",
            ),
            node(
                func=get_feature_importance,
                inputs=["regression_models", "params:model_options.feature_columns"],
                outputs="regression_feature_importance",
                name="regression_feature_importance_node",
            ),
            node(
                func=create_regression_report,
                inputs=["regression_metrics", "regression_feature_importance"],
                outputs="regression_report",
                name="create_regression_report_node",
            ),
            # Evaluación de Clasificación
            node(
                func=evaluate_classification_models,
                inputs=["classification_predictions", "y_cls_train", "y_cls_test"],
                outputs="classification_metrics",
                name="evaluate_classification_node",
            ),
            node(
                func=get_feature_importance,
                inputs=["classification_models", "params:model_options.feature_columns"],
                outputs="classification_feature_importance",
                name="classification_feature_importance_node",
            ),
            node(
                func=create_classification_report,
                inputs=["classification_metrics", "classification_feature_importance"],
                outputs="classification_report",
                name="create_classification_report_node",
            ),
        ]
    )




