"""
DAG de Airflow para ejecutar el pipeline completo de Kedro
League of Legends ML Project
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Configuración por defecto del DAG
default_args = {
    'owner': 'league-ml-team',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

# Definir el DAG
dag = DAG(
    'kedro_league_ml_pipeline',
    default_args=default_args,
    description='Pipeline completo de ML para League of Legends',
    schedule_interval='@weekly',  # Ejecutar semanalmente
    catchup=False,
    tags=['kedro', 'ml', 'league-of-legends'],
)

# ============================================================================
# TASKS - Cada pipeline de Kedro como una tarea
# ============================================================================

# Task 1: Data Cleaning
data_cleaning_task = BashOperator(
    task_id='data_cleaning',
    bash_command='cd /opt/airflow/kedro_project && python -m kedro run --pipeline data_cleaning',
    dag=dag,
)

# Task 2: Data Exploration
data_exploration_task = BashOperator(
    task_id='data_exploration',
    bash_command='cd /opt/airflow/kedro_project && python -m kedro run --pipeline data_exploration',
    dag=dag,
)

# Task 3: Data Processing (Feature Engineering)
data_processing_task = BashOperator(
    task_id='data_processing',
    bash_command='cd /opt/airflow/kedro_project && python -m kedro run --pipeline data_processing',
    dag=dag,
)

# Task 4: Unsupervised Learning (Clustering, Dimensionality Reduction, Anomaly Detection)
unsupervised_learning_task = BashOperator(
    task_id='unsupervised_learning',
    bash_command='cd /opt/airflow/kedro_project && python -m kedro run --pipeline unsupervised_learning',
    dag=dag,
)

# Task 5: Model Training (Data Science)
model_training_task = BashOperator(
    task_id='model_training',
    bash_command='cd /opt/airflow/kedro_project && python -m kedro run --pipeline data_science',
    dag=dag,
)

# Task 5: Model Evaluation
model_evaluation_task = BashOperator(
    task_id='model_evaluation',
    bash_command='cd /opt/airflow/kedro_project && python -m kedro run --pipeline evaluation',
    dag=dag,
)

# Task 6: Generate Final Report (Python function)
def generate_final_report(**context):
    """Genera un reporte final consolidado."""
    import json
    from datetime import datetime
    
    report = {
        'pipeline_name': 'League of Legends ML',
        'execution_date': str(datetime.now()),
        'status': 'completed',
        'message': 'Pipeline ejecutado exitosamente'
    }
    
    print(f"📊 REPORTE FINAL: {json.dumps(report, indent=2)}")
    return report

final_report_task = PythonOperator(
    task_id='generate_final_report',
    python_callable=generate_final_report,
    dag=dag,
)

# ============================================================================
# DEPENDENCIAS - Orden de ejecución
# ============================================================================

# Flujo lineal del pipeline
data_cleaning_task >> data_exploration_task
data_exploration_task >> data_processing_task
data_processing_task >> unsupervised_learning_task
unsupervised_learning_task >> model_training_task
model_training_task >> model_evaluation_task
model_evaluation_task >> final_report_task


