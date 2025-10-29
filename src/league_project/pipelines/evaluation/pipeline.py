"""
Pipeline de evaluación con soporte para CrossValidation.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    evaluate_regression_models,
    evaluate_classification_models,
    get_feature_importance,
    create_regression_report,
    create_classification_report,
    generate_cv_comparison_table_regression,
    generate_cv_comparison_table_classification,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de evaluación con tablas comparativas de CV.
    
    Returns:
        Pipeline de Kedro con evaluación de modelos y CV scores
    """
    return pipeline(
        [
            # Evaluación de Regresión con CV
            node(
                func=evaluate_regression_models,
                inputs=["regression_predictions", "y_reg_train", "y_reg_test", "regression_cv_results"],
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
            node(
                func=generate_cv_comparison_table_regression,
                inputs=["regression_metrics", "regression_cv_results"],
                outputs="regression_cv_comparison_table",
                name="generate_regression_cv_table_node",
            ),
            # Evaluación de Clasificación con CV
            node(
                func=evaluate_classification_models,
                inputs=["classification_predictions", "y_cls_train", "y_cls_test", "classification_cv_results"],
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
            node(
                func=generate_cv_comparison_table_classification,
                inputs=["classification_metrics", "classification_cv_results"],
                outputs="classification_cv_comparison_table",
                name="generate_classification_cv_table_node",
            ),
        ]
    )




