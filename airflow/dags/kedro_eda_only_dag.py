"""
DAG de Airflow para ejecutar solo el pipeline de EDA
(Data Cleaning + Data Exploration)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Configuración por defecto
default_args = {
    'owner': 'league-ml-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': datetime(2024, 1, 1),
}

# DAG de EDA
dag = DAG(
    'kedro_eda_pipeline',
    default_args=default_args,
    description='Pipeline de EDA - Limpieza y Exploración',
    schedule_interval='@daily',  # Ejecutar diariamente
    catchup=False,
    tags=['kedro', 'eda', 'exploration'],
)

# Task 1: Ejecutar pipeline completo de EDA
eda_task = BashOperator(
    task_id='run_eda_pipeline',
    bash_command='cd /opt/airflow/kedro_project && python -m kedro run --pipeline eda',
    dag=dag,
)

eda_task


