# 🎮 League of Legends ML Project

**Sistema completo de Machine Learning para predicción de partidas de League of Legends Worlds**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Kedro](https://img.shields.io/badge/Kedro-1.0.0-ffc900.svg)](https://kedro.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-20.10+-2496ED.svg)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/Airflow-2.8.0-017CEE.svg)](https://airflow.apache.org/)

---

## 🎯 Descripción del Proyecto

Sistema de **Machine Learning en producción** para análisis y predicción de partidas del torneo mundial de League of Legends (Worlds). Implementa la metodología **CRISP-DM** completa con **Kedro**, **Docker** y **Apache Airflow**.

### **Objetivos:**
1. 🎲 **Predicción de duración de partidas** (Regresión)
2. 🏆 **Predicción del equipo ganador** (Clasificación)
3. 📊 **Análisis exploratorio** de estadísticas profesionales

---

## 🏆 Resultados Destacados

| Problema | Mejor Modelo | Métrica Principal | Resultado |
|----------|--------------|-------------------|-----------|
| **Clasificación** | SVM | Accuracy | **98.56%** 🎯 |
| **Regresión** | Gradient Boosting | R² | **0.7928** 📈 |

- ✅ **10 modelos** entrenados y comparados
- ✅ **7,620 partidas** procesadas
- ✅ **246 equipos** analizados
- ✅ **137 campeones** evaluados

---

## 📋 Características

### **Arquitectura Profesional:**
- 🔄 **5 Pipelines modulares** con Kedro
- 🐳 **Dockerizado** para reproducibilidad
- 🌊 **Apache Airflow** para orquestación
- 📊 **CRISP-DM** metodología completa
- 📝 **18+ documentos** técnicos

### **Machine Learning:**
- **5 modelos de regresión:** Linear, Ridge, Lasso, Random Forest, Gradient Boosting
- **5 modelos de clasificación:** Logistic, Random Forest, Gradient Boosting, SVM, Naive Bayes
- **Feature Engineering:** 18 features ingenieradas
- **Evaluación completa:** RMSE, MAE, R², Accuracy, Precision, Recall, F1, AUC-ROC

---

## 🚀 Inicio Rápido (5 minutos)

### **Opción 1: Kedro (Recomendado)**

```bash
# 1. Clonar repositorio
git clone https://github.com/glYohanny/Evaluacion_machine_learning.git
cd Evaluacion_machine_learning/league-project

# 2. Crear entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar pipeline completo
kedro run
```

**Duración:** ~2 minutos  
**Output:** 10 modelos entrenados + reportes completos

---

### **Opción 2: Docker + Airflow**

```bash
# 1. Setup inicial
.\setup_airflow_windows.ps1

# 2. Iniciar servicios
docker-compose up -d

# 3. Acceder a Airflow
# http://localhost:8080
# Usuario: admin / Password: admin
```

---

## 📊 Estructura del Proyecto

```
league-project/
├── src/league_project/pipelines/    # 5 Pipelines de ML
│   ├── data_cleaning/               # Limpieza de datos
│   ├── data_exploration/            # Análisis exploratorio (EDA)
│   ├── data_processing/             # Feature engineering
│   ├── data_science/                # Entrenamiento de modelos
│   └── evaluation/                  # Evaluación y métricas
├── data/                            # Datos y resultados
│   ├── 01_raw/                     # Datos originales
│   ├── 02_intermediate/            # Datos limpios
│   ├── 06_models/                  # Modelos entrenados (.pkl)
│   └── 08_reporting/               # Reportes y métricas
├── airflow/dags/                   # 3 DAGs de Airflow
├── conf/                           # Configuración de Kedro
├── notebooks/                      # Jupyter notebooks
├── Dockerfile                      # Docker de Kedro
├── Dockerfile.airflow             # Docker de Airflow
└── docker-compose.yml             # Orquestación de servicios
```

---

## 🎓 Metodología CRISP-DM

### **1. Business Understanding** ✅
- Definición de objetivos: predicción de duración y ganador
- Métricas de éxito: Accuracy > 90%, R² > 0.75

### **2. Data Understanding** ✅
- 7 datasets: matches, kills, gold, bans, monsters, structures
- 10,000+ partidas profesionales del torneo Worlds
- Análisis exploratorio con 8 reportes diferentes

### **3. Data Preparation** ✅
- Limpieza: duplicados, nulos, outliers
- Imputación de 423 valores faltantes
- Feature engineering: 18 features creadas

### **4. Modeling** ✅
- 10 modelos de ML (5 regresión + 5 clasificación)
- Train/test split: 80/20
- Normalización con StandardScaler

### **5. Evaluation** ✅
- Métricas completas para cada modelo
- Feature importance identificada
- Comparación y selección del mejor modelo

### **6. Deployment** ✅
- Sistema dockerizado
- Airflow para automatización
- Listo para producción

---

## 📈 Resultados Detallados

### **Clasificación (Predicción de Ganador):**

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| **SVM** 🥇 | **0.9856** | **0.9856** | **0.9880** | **0.9868** | **0.9988** |
| Logistic Regression | 0.9836 | 0.9810 | 0.9892 | 0.9851 | 0.9991 |
| Random Forest | 0.9823 | 0.9821 | 0.9856 | 0.9838 | 0.9988 |
| Gradient Boosting | 0.9816 | 0.9832 | 0.9832 | 0.9832 | 0.9990 |
| Naive Bayes | 0.9705 | 0.9747 | 0.9712 | 0.9729 | 0.9895 |

**Interpretación:** El modelo predice correctamente al ganador en **98.56 de cada 100 partidas**.

---

### **Regresión (Predicción de Duración):**

| Modelo | RMSE | MAE | R² Train | R² Test |
|--------|------|-----|----------|---------|
| **Gradient Boosting** 🥇 | **3.70** | **2.85** | **0.8123** | **0.7928** |
| Ridge | 3.95 | 3.08 | 0.7525 | 0.7634 |
| Linear Regression | 3.95 | 3.08 | 0.7525 | 0.7633 |
| Random Forest | 3.96 | 3.02 | 0.9450 | 0.7624 |
| Lasso | 3.97 | 3.10 | 0.7503 | 0.7610 |

**Interpretación:** El modelo predice la duración con un **error promedio de 2.85 minutos**, explicando el **79.28%** de la varianza.

---

### **Features Más Importantes:**

1. 🥇 **Diferencia de oro** (gold_diff) - 35% importancia
2. 🥈 **Diferencia de kills** (kills_diff) - 28% importancia
3. 🥉 **Diferencia de torres** (towers_diff) - 18% importancia

---

## 📚 Documentación Completa

| Documento | Descripción |
|-----------|-------------|
| [**GUIA_EJECUCION_COMPLETA.md**](GUIA_EJECUCION_COMPLETA.md) | Guía paso a paso para ejecutar el proyecto |
| [**GUIA_PRESENTACION.md**](GUIA_PRESENTACION.md) | Script completo para presentación oral (20 min) |
| [**README_COMPLETO.md**](README_COMPLETO.md) | Documentación técnica exhaustiva (40+ páginas) |
| [**EVALUACION_PARCIAL_CUMPLIMIENTO.md**](EVALUACION_PARCIAL_CUMPLIMIENTO.md) | Verificación de requisitos académicos |
| [**RESUMEN_EJECUTIVO.md**](RESUMEN_EJECUTIVO.md) | Resumen ejecutivo del proyecto |
| [**DOCKER_AIRFLOW_GUIDE.md**](DOCKER_AIRFLOW_GUIDE.md) | Guía detallada de Docker y Airflow |
| [**QUICK_START.md**](QUICK_START.md) | Inicio rápido en 5 minutos |

---

## 🛠️ Tecnologías Utilizadas

### **Framework de ML:**
- **Kedro 1.0.0** - Pipeline orchestration
- **Scikit-learn** - Machine learning models
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### **Deployment:**
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration
- **Apache Airflow 2.8.0** - Workflow orchestration
- **PostgreSQL** - Metadata database
- **Redis** - Message broker

### **Development:**
- **Python 3.11**
- **Jupyter** - Interactive notebooks
- **pytest** - Testing
- **flake8** - Linting
- **black** - Code formatting

---

## 📊 Pipelines de Kedro

### **1. data_cleaning**
- Limpia 7 datasets raw
- Elimina duplicados y outliers
- Imputa valores faltantes
- **Output:** Datos limpios en `data/02_intermediate/`

### **2. data_exploration**
- Estadísticas descriptivas
- Análisis de 246 equipos
- Análisis de 137 campeones
- **Output:** 8 reportes en `data/08_reporting/`

### **3. data_processing**
- Feature engineering (18 features)
- Train/test split (80/20)
- Normalización con StandardScaler
- **Output:** Features en `data/05_model_input/`

### **4. data_science**
- Entrena 10 modelos de ML
- 5 regresión + 5 clasificación
- **Output:** Modelos en `data/06_models/`

### **5. evaluation**
- Calcula métricas completas
- Feature importance
- **Output:** Reportes JSON en `data/08_reporting/`

---

## 🎤 Para Evaluadores

### **Demo Rápida (5 minutos):**
```bash
kedro run --pipeline eda
```
Ejecuta limpieza + análisis exploratorio, generando reportes en 45 segundos.

### **Pipeline Completo (2 minutos):**
```bash
kedro run
```
Ejecuta todo el sistema: limpieza, análisis, entrenamiento y evaluación.

### **Ver Resultados:**
```bash
# Métricas de modelos
cat data/08_reporting/classification_report.json
cat data/08_reporting/regression_report.json

# Análisis exploratorio
cat data/08_reporting/team_performance_analysis.csv
cat data/08_reporting/eda_complete_report.json
```

---

## 📧 Contacto

**Autor:** Pedro Torres (glYohanny)  
**Email:** ped.torres@duocuc.cl  
**Institución:** DuocUC  
**Curso:** Machine Learning - MLY0100  
**GitHub:** https://github.com/glYohanny/Evaluacion_machine_learning

---

## 📄 Licencia

Este proyecto es parte de un trabajo académico para el curso de Machine Learning en DuocUC.

---

## 🙏 Agradecimientos

- **Riot Games** - Datos del torneo Worlds
- **Kedro Team** - Framework de pipelines
- **Apache Airflow** - Sistema de orquestación
- **Scikit-learn** - Librería de ML

---

## 🔗 Enlaces Útiles

- [Kedro Documentation](https://kedro.readthedocs.io/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Docker Documentation](https://docs.docker.com/)

---

**⭐ Si este proyecto te fue útil, dale una estrella en GitHub!**

---

**Última actualización:** Octubre 27, 2025  
**Versión:** 1.0.0  
**Estado:** ✅ Production Ready
