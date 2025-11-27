"""Project pipelines."""

from kedro.pipeline import Pipeline

from league_project.pipelines import data_cleaning
from league_project.pipelines import data_exploration
from league_project.pipelines import data_processing
from league_project.pipelines import data_science
from league_project.pipelines import evaluation
from league_project.pipelines import unsupervised_learning


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.
    
    Este proyecto sigue la metodología CRISP-DM con cinco pipelines principales:
    1. data_cleaning: Limpieza y preparación inicial de datos raw
    2. data_exploration: Análisis exploratorio de datos (EDA)
    3. data_processing: Feature engineering y preparación avanzada
    4. data_science: Entrenamiento de modelos de regresión y clasificación
    5. evaluation: Evaluación de modelos y generación de reportes

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    
    # Registrar pipelines individuales
    dc_pipeline = data_cleaning.create_pipeline()
    de_pipeline = data_exploration.create_pipeline()
    dp_pipeline = data_processing.create_pipeline()
    ds_pipeline = data_science.create_pipeline()
    eval_pipeline = evaluation.create_pipeline()
    unsupervised_pipeline = unsupervised_learning.create_pipeline()
    
    # Pipeline completo por defecto (ejecuta todos en orden incluyendo no supervisado)
    default_pipeline = dc_pipeline + de_pipeline + dp_pipeline + unsupervised_pipeline + ds_pipeline + eval_pipeline
    
    # Pipeline de limpieza y exploración (solo análisis inicial)
    eda_pipeline = dc_pipeline + de_pipeline
    
    # Pipeline completo con aprendizaje no supervisado
    full_ml_pipeline = dc_pipeline + de_pipeline + dp_pipeline + unsupervised_pipeline + ds_pipeline + eval_pipeline
    
    return {
        "__default__": default_pipeline,
        "data_cleaning": dc_pipeline,
        "data_exploration": de_pipeline,
        "data_processing": dp_pipeline,
        "data_science": ds_pipeline,
        "evaluation": eval_pipeline,
        "unsupervised_learning": unsupervised_pipeline,
        "eda": eda_pipeline,  # Pipeline combinado de limpieza + exploración
        "full_ml": full_ml_pipeline,  # Pipeline completo con no supervisado
        "dc": dc_pipeline,  # Alias corto
        "de": de_pipeline,  # Alias corto
        "dp": dp_pipeline,  # Alias corto
        "ds": ds_pipeline,  # Alias corto
        "eval": eval_pipeline,  # Alias corto
        "unsupervised": unsupervised_pipeline,  # Alias corto
    }
