# 🎤 COMANDOS PARA LA PRESENTACIÓN
**League of Legends ML - Guía de Comandos**

---

## 🚀 PARTE 1: DEMOSTRACIÓN RÁPIDA (5 minutos)

### **Setup Inicial (Solo primera vez)**

```powershell
# 1. Clonar repositorio
git clone https://github.com/glYohanny/Eva_machine_learning.git
cd Eva_machine_learning

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno
.\venv\Scripts\Activate.ps1

# 4. Instalar dependencias
pip install -r requirements.txt
```

---

### **Ejecución del Pipeline Completo**

```powershell
# Ejecutar TODO el proyecto (5 pipelines)
kedro run

# Duración: ~2 minutos
# Output: 10 modelos entrenados + reportes
```

---

### **Ver Resultados (Método Rápido)**

```powershell
# Opción 1: Script Python (RECOMENDADO)
python ver_resultados.py

# Opción 2: Archivos JSON
Get-Content data/08_reporting/classification_report.json | ConvertFrom-Json
Get-Content data/08_reporting/regression_report.json | ConvertFrom-Json
```

---

### **Dashboard Interactivo**

```powershell
# 1. Instalar Streamlit (primera vez)
pip install streamlit plotly

# 2. Iniciar dashboard
streamlit run dashboard_ml.py

# 3. Abrir navegador automáticamente
# URL: http://localhost:8501
```

---

## 📊 PARTE 2: PIPELINES INDIVIDUALES

### **Pipeline 1: Limpieza de Datos**

```powershell
kedro run --pipeline data_cleaning

# Qué hace:
# - Elimina duplicados
# - Imputa valores faltantes
# - Elimina outliers
# Duración: ~10 segundos
```

---

### **Pipeline 2: Análisis Exploratorio (EDA)**

```powershell
kedro run --pipeline eda

# Qué hace:
# - Limpieza + Análisis exploratorio
# - Genera 8 reportes
# - Estadísticas descriptivas
# Duración: ~45 segundos
```

---

### **Pipeline 3: Feature Engineering**

```powershell
kedro run --pipeline data_processing

# Qué hace:
# - Crea 18 features
# - Train/test split (80/20)
# - Normalización
# Duración: ~15 segundos
```

---

### **Pipeline 4: Entrenamiento**

```powershell
kedro run --pipeline data_science

# Qué hace:
# - Entrena 10 modelos
# - 5 clasificación + 5 regresión
# - Guarda modelos (.pkl)
# Duración: ~45 segundos
```

---

### **Pipeline 5: Evaluación**

```powershell
kedro run --pipeline evaluation

# Qué hace:
# - Calcula métricas
# - Genera reportes JSON
# - Feature importance
# Duración: ~15 segundos
```

---

## 📈 PARTE 3: VISUALIZACIÓN DE RESULTADOS

### **Método 1: Consola (Rápido)**

```powershell
# Ver resultados formateados
python ver_resultados.py

# Output:
# - Mejor modelo clasificación: SVM (98.56%)
# - Mejor modelo regresión: Gradient Boosting (R²=0.7928)
# - Tabla de comparación de modelos
# - Top 5 features importantes
```

---

### **Método 2: Dashboard (Visual)**

```powershell
# Iniciar dashboard interactivo
streamlit run dashboard_ml.py

# Páginas disponibles:
# 1. Resumen General
# 2. Modelos de Clasificación
# 3. Modelos de Regresión
# 4. Importancia de Features
# 5. Exploración de Datos
# 6. Configuración
```

---

### **Método 3: Jupyter Notebook**

```powershell
# Iniciar Jupyter con Kedro
kedro jupyter notebook

# Abrir: notebooks/analisis_lol_crisp_dm.ipynb
# Contiene análisis CRISP-DM completo
```

---

### **Método 4: Archivos Directos**

```powershell
# Ver métricas de clasificación
cat data/08_reporting/classification_report.json

# Ver métricas de regresión
cat data/08_reporting/regression_report.json

# Ver análisis de equipos
Import-Csv data/08_reporting/team_performance_analysis.csv | Format-Table

# Listar modelos entrenados
dir data/06_models/*.pkl
```

---

## 🐳 PARTE 4: DOCKER (Opcional)

### **Construir Imagen**

```powershell
# Construir imagen de Kedro
docker build -t league-kedro-ml:latest .

# Duración: 5-10 minutos (primera vez)
```

---

### **Ejecutar Contenedor**

```powershell
# Ejecutar pipeline en Docker
docker run -v ${PWD}/data:/app/data league-kedro-ml:latest kedro run

# Ver ayuda de Kedro
docker run league-kedro-ml:latest kedro --help
```

---

## 🌊 PARTE 5: AIRFLOW (Opcional)

### **Setup Inicial de Airflow**

```powershell
# Script automático de configuración
.\setup_airflow_windows.ps1

# Duración: 5-10 minutos (primera vez)
```

---

### **Iniciar Servicios de Airflow**

```powershell
# Iniciar todos los servicios
docker-compose up -d

# Servicios iniciados:
# - PostgreSQL (Base de datos)
# - Airflow Webserver (UI)
# - Airflow Scheduler (Ejecutor)
```

---

### **Acceder a Airflow UI**

```
URL: http://localhost:8080
Usuario: admin
Password: admin

DAGs disponibles:
1. kedro_league_ml (Pipeline completo)
2. kedro_eda_only (Solo análisis)
3. kedro_training_only (Solo entrenamiento)
```

---

### **Ver Logs de Airflow**

```powershell
# Ver logs de todos los servicios
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f airflow-webserver
docker-compose logs -f airflow-scheduler

# Ver estado de servicios
docker-compose ps
```

---

### **Detener Servicios**

```powershell
# Detener servicios
docker-compose down

# Detener y limpiar volúmenes
docker-compose down -v
```

---

## 🔍 PARTE 6: EXPLORACIÓN DE DATOS

### **Ver Estructura de Datos**

```powershell
# Ver primeras filas del dataset
python -c "import pandas as pd; df = pd.read_csv('data/01_raw/LeagueofLegends.csv'); print(df.head())"

# Ver estadísticas descriptivas
python -c "import pandas as pd; df = pd.read_csv('data/01_raw/LeagueofLegends.csv'); print(df.describe())"

# Ver columnas disponibles
python -c "import pandas as pd; df = pd.read_csv('data/01_raw/LeagueofLegends.csv'); print(df.columns.tolist())"

# Ver shape del dataset
python -c "import pandas as pd; df = pd.read_csv('data/01_raw/LeagueofLegends.csv'); print(f'Shape: {df.shape}')"
```

---

### **Scripts Rápidos**

```powershell
# Script para ver datos
python notebooks/ver_datos.py

# Script para demo de resultados
python notebooks/demo_kedro_results.py

# Verificar pipelines
python verificar_pipelines.py
```

---

## 📋 PARTE 7: COMANDOS DE VERIFICACIÓN

### **Verificar Instalación**

```powershell
# Ver versión de Python
python --version

# Ver versión de Kedro
kedro --version

# Ver versión de Docker
docker --version
docker-compose --version

# Ver paquetes instalados
pip list | Select-String -Pattern "kedro|pandas|sklearn|streamlit"
```

---

### **Listar Recursos del Proyecto**

```powershell
# Listar pipelines disponibles
kedro pipeline list

# Listar datasets en catálogo
kedro catalog list

# Ver estructura del proyecto
tree /F

# Ver tamaño de datos
du -sh data/*
```

---

### **Verificar Resultados**

```powershell
# Verificar que existen modelos
dir data/06_models

# Verificar que existen reportes
dir data/08_reporting

# Contar archivos generados
(Get-ChildItem -Recurse data/06_models, data/08_reporting | Measure-Object).Count
```

---

## 🎯 PARTE 8: DEMOSTRACIÓN SECUENCIAL

### **Flujo Completo (Paso a Paso)**

```powershell
# 1. Activar entorno
.\venv\Scripts\Activate.ps1

# 2. Limpiar datos previos (opcional)
Remove-Item -Recurse data/02_intermediate, data/06_models, data/08_reporting -ErrorAction SilentlyContinue

# 3. Ejecutar pipeline completo
kedro run

# 4. Ver resultados en consola
python ver_resultados.py

# 5. Abrir dashboard
streamlit run dashboard_ml.py

# 6. (Opcional) Abrir Jupyter
kedro jupyter notebook
```

---

## 💡 PARTE 9: TIPS PARA LA PRESENTACIÓN

### **Comandos de Un Solo Paso**

```powershell
# Pipeline completo + ver resultados
kedro run && python ver_resultados.py

# Pipeline + dashboard
kedro run && streamlit run dashboard_ml.py

# Limpiar + ejecutar
Remove-Item -Recurse data/02_intermediate, data/06_models -ErrorAction SilentlyContinue; kedro run
```

---

### **Mostrar Progreso en Tiempo Real**

```powershell
# Ejecutar con logs verbosos
kedro run --verbose

# Ejecutar pipeline específico con logs
kedro run --pipeline eda --verbose
```

---

### **Comandos Rápidos de Demostración**

```powershell
# 1. Ver que hay datos raw
dir data/01_raw/*.csv

# 2. Ejecutar pipeline
kedro run

# 3. Ver modelos generados
dir data/06_models

# 4. Ver resultados
python ver_resultados.py

# 5. Dashboard
streamlit run dashboard_ml.py
```

---

## 📊 PARTE 10: MÉTRICAS CLAVE PARA PRESENTAR

### **Clasificación (Predicción de Ganador)**

```
Mejor Modelo: SVM
- Accuracy:  98.56%
- Precision: 98.56%
- Recall:    98.80%
- F1-Score:  98.68%
- AUC-ROC:   99.88%

Interpretación:
Predice correctamente al ganador en 98.56 de cada 100 partidas
```

---

### **Regresión (Predicción de Duración)**

```
Mejor Modelo: Gradient Boosting
- R² Score: 0.7928 (79.28% varianza explicada)
- RMSE:     3.70 minutos
- MAE:      2.85 minutos

Interpretación:
Si una partida dura 35 minutos, 
el modelo predice entre 32-38 minutos (±3 min)
```

---

### **Features Más Importantes**

```
Clasificación:
1. tower_diff (diferencia torres)
2. red_towers
3. kill_diff (diferencia kills)
4. blue_towers
5. blue_barons

Regresión:
1. red_barons
2. blue_barons
3. blue_towers
4. gold_diff_20 (oro a 20 min)
5. blue_dragons
```

---

### **Datos del Proyecto**

```
Dataset:
- 7,620 partidas profesionales
- 246 equipos analizados
- 137 campeones evaluados
- 8 archivos CSV de entrada

Modelos:
- 10 modelos entrenados
- 5 clasificación + 5 regresión
- Tiempo de entrenamiento: ~2 minutos
```

---

## 🎤 PARTE 11: SCRIPT DE PRESENTACIÓN

### **Introducción (1 minuto)**

```
"Hoy presentaré un sistema completo de Machine Learning 
para predecir resultados de partidas de League of Legends.

Tecnologías:
- Kedro para pipelines modulares
- Docker para reproducibilidad
- Airflow para orquestación
- Streamlit para visualización

Repositorio: github.com/glYohanny/Eva_machine_learning"
```

---

### **Demostración en Vivo (3 minutos)**

```powershell
# 1. Mostrar estructura
tree /F /L 2

# 2. Ejecutar pipeline
kedro run

# 3. Mostrar resultados
python ver_resultados.py

# 4. Abrir dashboard
streamlit run dashboard_ml.py
```

---

### **Resultados (2 minutos)**

```
"Resultados obtenidos:

Clasificación:
- 98.56% de accuracy con SVM
- Predice correctamente 9,856 de cada 10,000 partidas

Regresión:
- R² de 0.7928 con Gradient Boosting
- Error promedio de solo 2.85 minutos

Factores clave identificados:
- Diferencia de torres (35% importancia)
- Diferencia de kills (28% importancia)
- Barones y dragones (13% importancia)"
```

---

### **Arquitectura (2 minutos)**

```
"5 Pipelines modulares:

1. data_cleaning: Limpieza de datos
2. data_exploration: Análisis exploratorio
3. data_processing: Feature engineering (18 features)
4. data_science: Entrenamiento (10 modelos)
5. evaluation: Evaluación y reportes

Todo containerizado con Docker
Orquestado con Airflow
Visualizado con Streamlit"
```

---

## 🔧 PARTE 12: TROUBLESHOOTING RÁPIDO

### **Si algo falla:**

```powershell
# Reinstalar dependencias
pip install --upgrade -r requirements.txt

# Limpiar caché de Python
Remove-Item -Recurse __pycache__, *.pyc -Force

# Verificar entorno virtual
.\venv\Scripts\Activate.ps1
python -c "import sys; print(sys.prefix)"

# Reiniciar servicios Docker
docker-compose down
docker-compose up -d

# Ver logs de error
Get-Content logs/info.log -Tail 50
```

---

## ✅ CHECKLIST DE PRESENTACIÓN

```
Pre-presentación:
[ ] Entorno virtual activado
[ ] Kedro instalado y funcionando
[ ] Datos en data/01_raw/
[ ] Docker corriendo (si se usa)
[ ] Dashboard probado
[ ] Resultados generados

Durante presentación:
[ ] Mostrar repositorio GitHub
[ ] Ejecutar kedro run
[ ] Mostrar resultados con script
[ ] Abrir dashboard
[ ] Explicar arquitectura
[ ] Mostrar métricas clave

Post-presentación:
[ ] Responder preguntas
[ ] Compartir repositorio
[ ] Mostrar documentación
```

---

## 🚀 COMANDOS DE UN VISTAZO

```powershell
# Setup
git clone https://github.com/glYohanny/Eva_machine_learning.git
cd Eva_machine_learning
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Ejecución
kedro run                          # Pipeline completo
python ver_resultados.py           # Ver resultados
streamlit run dashboard_ml.py      # Dashboard

# Pipelines individuales
kedro run --pipeline eda           # Solo análisis
kedro run --pipeline training      # Solo entrenamiento

# Docker
docker build -t league-kedro-ml .
docker-compose up -d

# Airflow
.\setup_airflow_windows.ps1
http://localhost:8080              # admin/admin

# Verificación
kedro pipeline list
kedro catalog list
dir data/06_models
python ver_resultados.py
```

---

## 📞 CONTACTO

```
Autor: Pedro Torres
Email: ped.torres@duocuc.cl
GitHub: github.com/glYohanny/Eva_machine_learning
Curso: Machine Learning - MLY0100
Institución: DuocUC
```

---

**¡Éxito en tu presentación!** 🎉

---

**Última actualización:** Octubre 29, 2025  
**Versión:** 1.0.0

