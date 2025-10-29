# 📊 GUÍA: Visualizar Resultados de Kedro en Airflow

## ✅ **TUS RESULTADOS ACTUALES**

Ya tienes resultados generados. Los acabas de ver con `python ver_resultados.py`:

- **Clasificación:** SVM con 98.56% accuracy
- **Regresión:** Gradient Boosting con R² = 0.7928
- **5 modelos** de cada tipo entrenados
- **Archivos generados** en `data/08_reporting/`

---

## 🎯 **FORMAS DE VISUALIZAR RESULTADOS**

### **Método 1: Archivos JSON Directos** ⭐ **MÁS SIMPLE**

Los resultados ya están guardados como archivos JSON después de ejecutar Kedro:

```powershell
# Ver resultados de clasificación
Get-Content data/08_reporting/classification_report.json | ConvertFrom-Json | ConvertTo-Json -Depth 10

# Ver resultados de regresión
Get-Content data/08_reporting/regression_report.json | ConvertFrom-Json | ConvertTo-Json -Depth 10

# Ver análisis de equipos
Import-Csv data/08_reporting/team_performance_analysis.csv | Format-Table
```

**Ventajas:**
- ✅ Instantáneo
- ✅ No requiere instalación adicional
- ✅ Formato estructurado (JSON/CSV)

---

### **Método 2: Script Python Rápido** ⭐ **RECOMENDADO**

Ya creamos el script `ver_resultados.py`:

```powershell
# Ejecutar script
python ver_resultados.py
```

**Ventajas:**
- ✅ Formateado bonito
- ✅ Tablas organizadas
- ✅ Resumen automático
- ✅ Interpretación de métricas

---

### **Método 3: Dashboard de Streamlit** ⭐ **MÁS PROFESIONAL**

```powershell
# 1. Instalar Streamlit (si no lo tienes)
pip install streamlit plotly

# 2. Iniciar dashboard
streamlit run dashboard_ml.py

# 3. Abrir navegador: http://localhost:8501
```

**Ventajas:**
- ✅ Gráficos interactivos
- ✅ 6 páginas diferentes
- ✅ Visualización profesional
- ✅ Exploración de datos
- ✅ Actualización automática

**Páginas disponibles:**
1. 🏠 Resumen General - KPIs y comparaciones
2. 🎯 Clasificación - Métricas de clasificación
3. 📈 Regresión - Métricas de regresión
4. 🔍 Features - Importancia de features
5. 📊 Datos - Exploración de datos
6. ⚙️ Configuración - Info técnica

---

### **Método 4: Jupyter Notebook**

```powershell
# Iniciar Jupyter con Kedro
kedro jupyter notebook

# O Jupyter Lab
kedro jupyter lab
```

Luego crea un notebook nuevo:

```python
# Cargar resultados
import json
import pandas as pd

# Clasificación
with open('data/08_reporting/classification_report.json') as f:
    classification = json.load(f)

print(f"Mejor modelo: {classification['best_model']}")
print(f"Accuracy: {classification['best_accuracy']:.4f}")

# Crear DataFrame de métricas
df = pd.DataFrame(classification['all_metrics'])
print(df[['model', 'test_accuracy', 'f1_score']])

# Gráfico
import matplotlib.pyplot as plt
models = [m['model'] for m in classification['all_metrics']]
accuracies = [m['test_accuracy'] for m in classification['all_metrics']]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies)
plt.title('Accuracy por Modelo')
plt.ylabel('Accuracy')
plt.ylim([0.96, 1.0])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## 🌊 **VISUALIZAR RESULTADOS DESDE AIRFLOW**

### **¿Se pueden ver los resultados en Airflow?**

**Respuesta:** Sí, pero con limitaciones. Airflow está diseñado para **orquestar** workflows, no para visualizar datos. Sin embargo, hay varias formas:

---

### **Opción 1: Ver Logs en Airflow UI** ⭐ **MÁS FÁCIL**

Los logs de ejecución muestran los resultados impresos por Kedro.

#### **Paso 1: Ejecutar DAG desde Airflow**

```
1. Abrir Airflow: http://localhost:8080
2. Usuario: admin / Password: admin
3. Ir a "DAGs"
4. Activar DAG: "kedro_league_ml"
5. Click en el botón "Play" (▶)
6. Trigger DAG manualmente
```

#### **Paso 2: Ver Logs**

```
1. Click en el DAG ejecutado
2. Click en "Grid" (ver ejecuciones)
3. Click en el cuadrado verde/rojo de la ejecución
4. Click en la tarea "run_kedro_pipeline"
5. Click en "Log"
6. Scroll down para ver resultados
```

**Lo que verás en los logs:**
```
[2025-10-29] INFO - ===== Modelo: SVM =====
[2025-10-29] INFO - Accuracy: 0.9856
[2025-10-29] INFO - F1-Score: 0.9868
[2025-10-29] INFO - Modelo guardado en: data/06_models/svm.pkl
[2025-10-29] INFO - Reportes guardados en: data/08_reporting/
```

**Ventajas:**
- ✅ No requiere configuración adicional
- ✅ Historial de ejecuciones
- ✅ Fácil de acceder

**Desventajas:**
- ❌ Solo texto, sin gráficos
- ❌ No interactivo
- ❌ Difícil de leer si hay muchos logs

---

### **Opción 2: XCom (Compartir Datos entre Tareas)** ⭐ **PARA MÉTRICAS PEQUEÑAS**

XCom permite que tareas compartan datos en Airflow.

#### **Paso 1: Crear tarea que extraiga métricas**

Crea un archivo: `airflow/dags/kedro_with_metrics_dag.py`

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import json

default_args = {
    'owner': 'league-ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def extract_metrics(**context):
    """Extrae métricas de los reportes JSON"""
    try:
        # Leer reporte de clasificación
        with open('/opt/airflow/kedro_project/data/08_reporting/classification_report.json') as f:
            classification = json.load(f)
        
        # Leer reporte de regresión
        with open('/opt/airflow/kedro_project/data/08_reporting/regression_report.json') as f:
            regression = json.load(f)
        
        # Extraer métricas clave
        metrics = {
            'classification': {
                'best_model': classification['best_model'],
                'accuracy': classification['best_accuracy'],
                'f1_score': classification['best_f1'],
                'auc_roc': classification['best_auc']
            },
            'regression': {
                'best_model': regression['best_model'],
                'r2_score': regression['best_r2'],
                'rmse': regression['best_rmse'],
                'mae': regression['best_mae']
            }
        }
        
        # Push a XCom
        context['task_instance'].xcom_push(key='ml_metrics', value=metrics)
        
        # Imprimir para logs
        print("\n" + "="*70)
        print("METRICAS EXTRAIDAS:")
        print("="*70)
        print(f"\nClasificacion:")
        print(f"  Modelo: {metrics['classification']['best_model'].upper()}")
        print(f"  Accuracy: {metrics['classification']['accuracy']:.4f}")
        print(f"\nRegresion:")
        print(f"  Modelo: {metrics['regression']['best_model'].upper()}")
        print(f"  R2 Score: {metrics['regression']['r2_score']:.4f}")
        print("="*70)
        
        return json.dumps(metrics, indent=2)
        
    except Exception as e:
        print(f"Error extrayendo metricas: {e}")
        return None

with DAG(
    dag_id='kedro_league_ml_with_metrics',
    default_args=default_args,
    description='Pipeline de Kedro con extraccion de metricas',
    schedule_interval='0 2 * * *',
    catchup=False,
    tags=['kedro', 'ml', 'metrics'],
) as dag:
    
    # Tarea 1: Ejecutar Kedro
    run_kedro = BashOperator(
        task_id='run_kedro_pipeline',
        bash_command='cd /opt/airflow/kedro_project && kedro run',
        env={'KEDRO_ENV': 'local'}
    )
    
    # Tarea 2: Extraer métricas
    extract = PythonOperator(
        task_id='extract_metrics',
        python_callable=extract_metrics,
        provide_context=True
    )
    
    # Orden de ejecución
    run_kedro >> extract
```

#### **Paso 2: Ver métricas en XCom**

```
1. Ejecutar el DAG "kedro_league_ml_with_metrics"
2. Ir a la tarea "extract_metrics"
3. Click en "XCom"
4. Ver la clave "ml_metrics"
5. Ver el JSON con las métricas
```

**Ventajas:**
- ✅ Métricas disponibles en Airflow UI
- ✅ Puedes usarlas en tareas posteriores
- ✅ Historial de métricas por ejecución

**Desventajas:**
- ❌ Solo para datos pequeños (< 48 KB)
- ❌ No es visual (solo JSON)

---

### **Opción 3: Generar Gráficos como Imágenes** ⭐ **PARA GRÁFICOS**

Modifica Kedro para generar gráficos y guárdalos como imágenes.

#### **Paso 1: Añadir nodo para generar gráficos**

En `src/league_project/pipelines/evaluation/nodes.py`:

```python
import matplotlib
matplotlib.use('Agg')  # Backend sin display
import matplotlib.pyplot as plt
import seaborn as sns

def generate_model_comparison_plot(classification_report: dict) -> None:
    """Genera gráfico de comparación de modelos"""
    
    # Extraer datos
    models = [m['model'] for m in classification_report['all_metrics']]
    accuracies = [m['test_accuracy'] for m in classification_report['all_metrics']]
    
    # Crear gráfico
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color='skyblue', edgecolor='navy')
    
    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.title('Comparacion de Accuracy - Modelos de Clasificacion', fontsize=16)
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim([0.96, 1.0])
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Guardar
    output_path = "data/08_reporting/model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grafico guardado en: {output_path}")
```

#### **Paso 2: Registrar nodo en pipeline**

En `src/league_project/pipelines/evaluation/pipeline.py`:

```python
from kedro.pipeline import Pipeline, node
from .nodes import generate_model_comparison_plot

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        # ... otros nodos ...
        
        node(
            func=generate_model_comparison_plot,
            inputs="classification_report",
            outputs=None,
            name="generate_plots"
        )
    ])
```

#### **Paso 3: Acceder a imágenes**

Los gráficos se guardan en `data/08_reporting/`:

```
data/08_reporting/
├── model_comparison.png          ← Gráfico de modelos
├── feature_importance.png        ← Importancia de features
└── confusion_matrix.png          ← Matriz de confusión
```

**Abrir desde Windows:**
```powershell
# Ver gráficos
explorer data\08_reporting\model_comparison.png
```

**Ventajas:**
- ✅ Gráficos profesionales
- ✅ Fácil de compartir
- ✅ Historial visual

**Desventajas:**
- ❌ No interactivo
- ❌ Requiere acceso al filesystem

---

### **Opción 4: Dashboard Externo Conectado a Airflow** ⭐ **SOLUCIÓN PROFESIONAL**

El dashboard de Streamlit se actualiza automáticamente con los resultados de Airflow.

#### **Flujo de trabajo:**

```
1. Airflow ejecuta DAG → Kedro genera modelos
2. Kedro guarda resultados → data/08_reporting/*.json
3. Dashboard lee archivos → Actualiza visualizaciones
4. Usuario abre dashboard → Ve resultados actualizados
```

#### **Configuración:**

```powershell
# Terminal 1: Iniciar Airflow
docker-compose up -d

# Terminal 2: Iniciar Dashboard
streamlit run dashboard_ml.py

# Usar ambos:
# - Airflow: http://localhost:8080 (ejecutar workflows)
# - Dashboard: http://localhost:8501 (ver resultados)
```

**Ventajas:**
- ✅ Mejor de ambos mundos
- ✅ Airflow para orquestación
- ✅ Dashboard para visualización
- ✅ Profesional y escalable

---

### **Opción 5: Plugin Custom de Airflow** (Avanzado)

Crea un plugin que añade una vista personalizada en Airflow.

**Archivo:** `airflow/plugins/league_ml_viz_plugin.py`

```python
from airflow.plugins_manager import AirflowPlugin
from flask import Blueprint
from flask_appbuilder import expose, BaseView as AppBuilderBaseView
import json
from pathlib import Path

class LeagueMLView(AppBuilderBaseView):
    default_view = "metrics"
    route_base = "/league_ml"
    
    @expose("/")
    def metrics(self):
        """Vista de métricas"""
        
        # Cargar métricas
        try:
            with open('/opt/airflow/kedro_project/data/08_reporting/classification_report.json') as f:
                classification = json.load(f)
            
            with open('/opt/airflow/kedro_project/data/08_reporting/regression_report.json') as f:
                regression = json.load(f)
            
            # Renderizar template HTML
            html_content = f"""
            <html>
            <head>
                <title>League ML Metrics</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric-card {{ 
                        background: #f0f0f0; 
                        padding: 20px; 
                        margin: 10px; 
                        border-radius: 8px; 
                    }}
                    h1 {{ color: #1f77b4; }}
                    .best {{ color: #28a745; font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>League of Legends ML - Resultados</h1>
                
                <div class="metric-card">
                    <h2>Clasificacion</h2>
                    <p>Mejor Modelo: <span class="best">{classification['best_model'].upper()}</span></p>
                    <p>Accuracy: {classification['best_accuracy']:.4f}</p>
                    <p>F1-Score: {classification['best_f1']:.4f}</p>
                    <p>AUC-ROC: {classification['best_auc']:.4f}</p>
                </div>
                
                <div class="metric-card">
                    <h2>Regresion</h2>
                    <p>Mejor Modelo: <span class="best">{regression['best_model'].upper()}</span></p>
                    <p>R² Score: {regression['best_r2']:.4f}</p>
                    <p>RMSE: {regression['best_rmse']:.2f} minutos</p>
                    <p>MAE: {regression['best_mae']:.2f} minutos</p>
                </div>
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            return f"<h1>Error cargando metricas: {e}</h1>"

# Crear vista
league_ml_view = LeagueMLView()

# Crear blueprint
league_ml_blueprint = Blueprint(
    "league_ml",
    __name__,
    template_folder="templates",
    url_prefix="/league_ml"
)

# Plugin
class LeagueMLPlugin(AirflowPlugin):
    name = "league_ml_viz"
    flask_blueprints = [league_ml_blueprint]
    appbuilder_views = [{
        "name": "League ML Metrics",
        "category": "ML Monitoring",
        "view": league_ml_view
    }]
```

**Acceso:**
```
http://localhost:8080/league_ml/
```

Aparecerá en el menú de Airflow UI.

---

## 📊 **RESUMEN Y RECOMENDACIONES**

### **Para Desarrollo / Testing:**
```
✅ RECOMENDADO: python ver_resultados.py
✅ ALTERNATIVA: cat data/08_reporting/*.json
```

### **Para Análisis Detallado:**
```
✅ RECOMENDADO: streamlit run dashboard_ml.py
✅ ALTERNATIVA: kedro jupyter notebook
```

### **Para Producción con Airflow:**
```
✅ RECOMENDADO: Dashboard Streamlit + Airflow (separados)
   - Airflow ejecuta workflows
   - Dashboard muestra resultados
   
✅ ALTERNATIVA: XCom para métricas + Logs para detalles
```

### **Para Reportes / Presentaciones:**
```
✅ RECOMENDADO: Generar gráficos PNG con Matplotlib
✅ ALTERNATIVA: Screenshots del dashboard
```

---

## 🎯 **TU CASO ESPECÍFICO**

**Tienes resultados generados**, así que:

### **Opción 1: Más Rápida (1 minuto)**
```powershell
python ver_resultados.py
```

### **Opción 2: Más Visual (2 minutos)**
```powershell
pip install streamlit plotly
streamlit run dashboard_ml.py
# Abrir: http://localhost:8501
```

### **Opción 3: Desde Airflow (5 minutos)**
```powershell
# 1. Iniciar Airflow
.\setup_airflow_windows.ps1
docker-compose up -d

# 2. Abrir: http://localhost:8080
# 3. Ejecutar DAG: kedro_league_ml
# 4. Ver logs de la tarea
```

---

## ✅ **CHECKLIST**

- [ ] Ejecuté `kedro run` y tengo resultados
- [ ] Probé `python ver_resultados.py`
- [ ] Instalé Streamlit: `pip install streamlit plotly`
- [ ] Ejecuté dashboard: `streamlit run dashboard_ml.py`
- [ ] (Opcional) Inicié Airflow: `docker-compose up -d`
- [ ] (Opcional) Ejecuté DAG desde Airflow UI

---

## 📞 **COMANDOS RÁPIDOS**

```powershell
# Ver resultados - Método 1 (texto)
python ver_resultados.py

# Ver resultados - Método 2 (dashboard)
streamlit run dashboard_ml.py

# Ver resultados - Método 3 (JSON raw)
Get-Content data/08_reporting/classification_report.json | ConvertFrom-Json

# Ver gráficos (si los generaste)
explorer data\08_reporting\model_comparison.png

# Iniciar Airflow
docker-compose up -d

# Ver logs de Airflow
docker-compose logs -f airflow-webserver
```

---

**Última actualización:** Octubre 29, 2025  
**Autor:** Pedro Torres


