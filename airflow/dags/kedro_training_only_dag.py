"""
DAG de Airflow para ejecutar solo entrenamiento y evaluación de modelos
(Asume que ya existen datos procesados)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
import os

# Configuración por defecto
default_args = {
    'owner': 'league-ml-team',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

dag = DAG(
    'kedro_model_training_pipeline',
    default_args=default_args,
    description='Pipeline de Entrenamiento y Evaluación de Modelos',
    schedule_interval=None,  # Trigger manual
    catchup=False,
    tags=['kedro', 'ml', 'training'],
)

# Función para verificar si existen los datos procesados
def check_processed_data(**context):
    """Verifica si existen los datos procesados necesarios."""
    data_path = '/opt/airflow/kedro_project/data/05_model_input/model_input_table.parquet'
    
    if os.path.exists(data_path):
        print("✅ Datos procesados encontrados. Procediendo al entrenamiento.")
        return 'train_models'
    else:
        print("❌ Datos procesados no encontrados. Ejecutando procesamiento primero.")
        return 'process_data'

# Task: Verificar datos
check_data_task = BranchPythonOperator(
    task_id='check_data',
    python_callable=check_processed_data,
    dag=dag,
)

# Task: Procesar datos (si es necesario)
process_data_task = BashOperator(
    task_id='process_data',
    bash_command='cd /opt/airflow/kedro_project && python -m kedro run --pipeline data_processing',
    dag=dag,
)

# Task: Entrenar modelos
train_models_task = BashOperator(
    task_id='train_models',
    bash_command='cd /opt/airflow/kedro_project && python -m kedro run --pipeline data_science',
    trigger_rule='none_failed_or_skipped',
    dag=dag,
)

# Task: Evaluar modelos
evaluate_models_task = BashOperator(
    task_id='evaluate_models',
    bash_command='cd /opt/airflow/kedro_project && python -m kedro run --pipeline evaluation',
    dag=dag,
)

# Dependencias
check_data_task >> [process_data_task, train_models_task]
process_data_task >> train_models_task
train_models_task >> evaluate_models_task


