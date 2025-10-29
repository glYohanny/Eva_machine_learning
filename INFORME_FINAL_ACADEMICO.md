# 📊 Informe Final Académico - Proyecto de Machine Learning

**Curso:** MLY0100 - Machine Learning  
**Institución:** DuocUC  
**Autor:** Pedro Torres (glYohanny)  
**Email:** ped.torres@duocuc.cl  
**Fecha:** Octubre 27, 2025  
**Repositorio:** https://github.com/glYohanny/Evaluacion_machine_learning

---

## Resumen Ejecutivo

Este proyecto implementa un sistema completo de Machine Learning para análisis y predicción de partidas del torneo mundial de League of Legends (Worlds). Se desarrolló siguiendo la metodología CRISP-DM, utilizando Kedro como framework de pipelines, Docker para containerización y Apache Airflow para orquestación. Los resultados obtenidos fueron excepcionales: **98.56% de accuracy** en predicción de ganador y **R² de 0.7928** en predicción de duración de partidas.

---

## 1. Introducción

### 1.1 Contexto del Proyecto

League of Legends es uno de los videojuegos competitivos más populares del mundo, con torneos profesionales que atraen millones de espectadores. El torneo mundial (Worlds) representa el pináculo de la competición profesional, donde los mejores equipos del mundo compiten por el campeonato.

### 1.2 Problemática

Los analistas deportivos y equipos profesionales necesitan herramientas para:
- Predecir el resultado de las partidas basándose en estadísticas tempranas
- Estimar la duración de las partidas para planificación de torneos
- Identificar los factores más importantes que contribuyen a la victoria

### 1.3 Objetivos del Proyecto

#### Objetivo General:
Desarrollar un sistema de Machine Learning para análisis y predicción de partidas profesionales de League of Legends.

#### Objetivos Específicos:
1. Implementar un pipeline completo de datos utilizando Kedro
2. Entrenar y comparar múltiples modelos de ML (regresión y clasificación)
3. Identificar las features más importantes para la predicción
4. Desplegar el sistema usando Docker y Apache Airflow
5. Documentar el proceso completo siguiendo CRISP-DM

---

## 2. Metodología CRISP-DM

### 2.1 Business Understanding

**Preguntas de negocio:**
1. ¿Qué factores determinan el ganador de una partida?
2. ¿Se puede predecir la duración de una partida con precisión?
3. ¿Cuáles son las métricas más importantes para el éxito?

**Criterios de éxito:**
- Accuracy > 90% para predicción de ganador
- R² > 0.75 para predicción de duración
- Sistema deployable en producción

### 2.2 Data Understanding

**Dataset utilizado:**
- **Fuente:** Partidas profesionales del torneo Worlds
- **Tamaño:** 7,620 partidas
- **Período:** Múltiples temporadas
- **Equipos:** 246 equipos profesionales
- **Campeones:** 137 personajes únicos

**Estructura de datos:**
- 7 archivos CSV relacionados
- Variables: kills, gold, torres, objetivos neutrales, bans
- Target variables: resultado de partida (binario), duración (continuo)

**Análisis exploratorio realizado:**
- Estadísticas descriptivas completas
- Análisis de distribuciones
- Identificación de outliers
- Matriz de correlaciones
- Análisis por equipos y campeones

### 2.3 Data Preparation

**Limpieza de datos:**
- Eliminación de 122 registros duplicados
- Imputación de 423 valores faltantes usando moda
- Detección y tratamiento de 7,611 outliers usando método IQR
- Validación de tipos de datos

**Feature Engineering:**
Se crearon 18 features derivadas, incluyendo:
- `gold_diff`: Diferencia de oro entre equipos
- `kills_diff`: Diferencia de asesinatos
- `towers_diff`: Diferencia de torres destruidas
- `dragons_diff`: Diferencia de dragones capturados
- `barons_diff`: Diferencia de barones capturados

**Transformaciones:**
- Normalización con StandardScaler
- Train/test split: 80/20 (6,096 train, 1,524 test)
- Conversión a formato Parquet para eficiencia

### 2.4 Modeling

**Modelos de Regresión (Predicción de Duración):**
1. Linear Regression (baseline)
2. Ridge Regression (regularización L2)
3. Lasso Regression (regularización L1)
4. Random Forest Regressor
5. Gradient Boosting Regressor

**Modelos de Clasificación (Predicción de Ganador):**
1. Logistic Regression (baseline)
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Support Vector Machine (SVM)
5. Naive Bayes

**Configuración:**
- Todos los modelos utilizan configuraciones estándar de scikit-learn
- Random state = 42 para reproducibilidad
- Train/test split consistente para todos los modelos

### 2.5 Evaluation

**Métricas de Regresión:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coeficiente de Determinación)
- Comparación Train vs Test para detectar overfitting

**Métricas de Clasificación:**
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- Matriz de confusión

### 2.6 Deployment

**Infraestructura implementada:**
- Kedro: Framework de pipelines modulares
- Docker: Containerización del sistema
- Apache Airflow: Orquestación y scheduling
- PostgreSQL: Base de datos de metadatos
- Redis: Message broker

**Automatización:**
- 3 DAGs de Airflow configurados
- Scheduling diario para EDA
- Scheduling semanal para reentrenamiento
- Ejecución manual para experimentación

---

## 3. Resultados

### 3.1 Modelos de Clasificación

**Tabla de Resultados:**

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Train Acc |
|--------|----------|-----------|--------|----------|---------|-----------|
| **SVM** | **0.9856** | **0.9856** | **0.9880** | **0.9868** | **0.9988** | 0.9916 |
| Logistic Regression | 0.9836 | 0.9810 | 0.9892 | 0.9851 | 0.9991 | 0.9882 |
| Random Forest | 0.9823 | 0.9821 | 0.9856 | 0.9838 | 0.9988 | 1.0000 |
| Gradient Boosting | 0.9816 | 0.9832 | 0.9832 | 0.9832 | 0.9990 | 0.9944 |
| Naive Bayes | 0.9705 | 0.9747 | 0.9712 | 0.9729 | 0.9895 | 0.9728 |

**Mejor modelo:** SVM con 98.56% de accuracy

**Interpretación:**
- El modelo SVM predice correctamente el ganador en 98.56 de cada 100 partidas
- El F1-Score de 0.9868 indica un excelente balance entre precision y recall
- El AUC-ROC de 0.9988 demuestra capacidad casi perfecta de discriminación
- No se observa overfitting significativo (train accuracy 99.16% vs test 98.56%)

### 3.2 Modelos de Regresión

**Tabla de Resultados:**

| Modelo | RMSE Train | RMSE Test | MAE Train | MAE Test | R² Train | R² Test |
|--------|------------|-----------|-----------|----------|----------|---------|
| **Gradient Boosting** | **3.44** | **3.70** | **2.68** | **2.85** | **0.8123** | **0.7928** |
| Ridge | 3.95 | 3.95 | 3.09 | 3.08 | 0.7525 | 0.7634 |
| Linear Regression | 3.95 | 3.95 | 3.09 | 3.08 | 0.7525 | 0.7633 |
| Random Forest | 1.86 | 3.96 | 1.44 | 3.02 | 0.9450 | 0.7624 |
| Lasso | 3.97 | 3.97 | 3.10 | 3.10 | 0.7503 | 0.7610 |

**Mejor modelo:** Gradient Boosting con R² de 0.7928

**Interpretación:**
- El modelo explica el 79.28% de la varianza en la duración de partidas
- El error promedio (MAE) es de 2.85 minutos
- Considerando que las partidas duran 25-45 minutos, un error de ~3 minutos es excelente
- Random Forest muestra overfitting (R² train 0.945 vs test 0.762)

### 3.3 Feature Importance

**Top 10 Features por Importancia:**

| Rank | Feature | Importancia (%) | Descripción |
|------|---------|-----------------|-------------|
| 1 | gold_diff | 35.2% | Diferencia de oro entre equipos |
| 2 | kills_diff | 28.7% | Diferencia de asesinatos |
| 3 | towers_diff | 18.4% | Diferencia de torres destruidas |
| 4 | dragons_diff | 7.3% | Diferencia de dragones capturados |
| 5 | barons_diff | 4.8% | Diferencia de barones capturados |
| 6 | blue_kills | 2.1% | Asesinatos del equipo azul |
| 7 | red_kills | 1.5% | Asesinatos del equipo rojo |
| 8 | gamelength_minutes | 1.2% | Duración de la partida |
| 9 | blue_towers | 0.5% | Torres destruidas equipo azul |
| 10 | red_towers | 0.3% | Torres destruidas equipo rojo |

**Insights clave:**
- Las diferencias entre equipos son más predictivas que valores absolutos
- El oro es el factor más importante (35%), reflejando ventaja económica
- Los kills son el segundo factor más importante (29%), indicando dominio en combate
- Las estructuras (torres) también son significativas (18%)

### 3.4 Análisis de Equipos

**Top 10 Equipos por Win Rate:**

| Equipo | Partidas | Victorias | Derrotas | Win Rate | Duración Promedio |
|--------|----------|-----------|----------|----------|-------------------|
| T1 | 85 | 67 | 18 | 78.82% | 31.2 min |
| GEN | 78 | 59 | 19 | 75.64% | 32.1 min |
| JDG | 72 | 54 | 18 | 75.00% | 30.8 min |
| DK | 68 | 49 | 19 | 72.06% | 31.5 min |
| RNG | 65 | 46 | 19 | 70.77% | 32.8 min |

**Total equipos analizados:** 246

### 3.5 Análisis de Campeones

**Top 10 Campeones Más Baneados:**

| Campeón | Bans Totales | % de Partidas | Correlación con Victoria |
|---------|--------------|---------------|--------------------------|
| Zeri | 1,847 | 24.2% | +0.15 |
| Yuumi | 1,623 | 21.3% | +0.22 |
| Kalista | 1,456 | 19.1% | +0.18 |
| LeBlanc | 1,289 | 16.9% | +0.12 |
| Lucian | 1,145 | 15.0% | +0.14 |

**Total campeones analizados:** 137

---

## 4. Análisis y Discusión

### 4.1 Comparación de Modelos

**Clasificación:**
- **SVM** logró el mejor balance de métricas
- Los modelos basados en árboles (RF, GB) tienen métricas muy cercanas
- Naive Bayes, aunque más simple, logra 97% accuracy
- **Conclusión:** SVM es el modelo recomendado para producción

**Regresión:**
- **Gradient Boosting** supera a otros modelos en R² test
- Random Forest muestra overfitting significativo
- Modelos lineales (Linear, Ridge, Lasso) tienen desempeño similar y robusto
- **Conclusión:** Gradient Boosting es el modelo recomendado, con Ridge como alternativa robusta

### 4.2 Factores de Éxito

Los tres factores más importantes identificados son:

1. **Ventaja económica (oro)**: El oro permite comprar mejores items, aumentando el poder del equipo
2. **Ventaja en combate (kills)**: Los kills otorgan oro y experiencia, creando ventaja compuesta
3. **Control del mapa (torres)**: Las torres permiten avanzar territorio y limitar opciones del enemigo

### 4.3 Limitaciones del Estudio

1. **Datos históricos:** Solo incluye partidas profesionales, puede no generalizar a ranked
2. **Parches del juego:** Los cambios de balance pueden afectar la validez temporal de los modelos
3. **Factores cualitativos:** No captura habilidad individual, comunicación de equipo, estrategia
4. **Sesgo de datos:** Posible sesgo hacia equipos más exitosos (más partidas en dataset)

### 4.4 Trabajo Futuro

**Mejoras potenciales:**

1. **Hyperparameter Tuning:**
   - GridSearchCV o RandomizedSearchCV
   - Optimización bayesiana
   - Potential improvement: +2-3% accuracy

2. **Cross-Validation:**
   - K-Fold CV (k=5 o k=10)
   - Validación más robusta
   - Reducir varianza de estimación

3. **Deep Learning:**
   - Redes neuronales recurrentes (LSTM) para secuencias temporales
   - Análisis de progresión de partida minuto a minuto
   - Potential improvement: +5-8% accuracy

4. **Features adicionales:**
   - Composición de campeones (synergies)
   - Métricas de jugadores individuales
   - Contexto del torneo (BO1 vs BO5)

5. **Deployment avanzado:**
   - API REST para predicciones en tiempo real
   - Dashboard interactivo con Streamlit
   - Integración con MLflow para tracking

---

## 5. Arquitectura del Sistema

### 5.1 Arquitectura de Software

```
┌─────────────────────────────────────┐
│      APACHE AIRFLOW (Capa 3)        │
│   • Orquestación                    │
│   • Scheduling                      │
│   • Monitoreo                       │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│     KEDRO PIPELINES (Capa 2)        │
│   • data_cleaning (8 nodos)         │
│   • data_exploration (8 nodos)      │
│   • data_processing (7 nodos)       │
│   • data_science (4 nodos)          │
│   • evaluation (6 nodos)            │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   DOCKER CONTAINERS (Capa 1)        │
│   • PostgreSQL                      │
│   • Redis                           │
│   • Kedro App                       │
└─────────────────────────────────────┘
```

### 5.2 Pipelines Implementados

**Total:** 5 pipelines, 33 nodos

1. **data_cleaning (8 nodos):**
   - Limpieza de 7 datasets
   - Eliminación de duplicados
   - Imputación de nulos
   - Detección de outliers
   - Generación de reporte de calidad

2. **data_exploration (8 nodos):**
   - Estadísticas descriptivas
   - Análisis de equipos (246)
   - Análisis de campeones (137)
   - Objetivos neutrales
   - Estructuras
   - Correlaciones
   - Duración de partidas
   - Reporte consolidado

3. **data_processing (7 nodos):**
   - Agregación de kills
   - Agregación de monsters
   - Agregación de structures
   - Agregación de gold
   - Selección de features
   - Split train/test
   - Normalización

4. **data_science (4 nodos):**
   - Entrenamiento regresión (5 modelos)
   - Entrenamiento clasificación (5 modelos)
   - Predicciones regresión
   - Predicciones clasificación

5. **evaluation (6 nodos):**
   - Evaluación regresión
   - Evaluación clasificación
   - Feature importance regresión
   - Feature importance clasificación
   - Reporte regresión
   - Reporte clasificación

### 5.3 Tecnologías y Herramientas

| Categoría | Tecnología | Versión | Propósito |
|-----------|------------|---------|-----------|
| Lenguaje | Python | 3.11 | Desarrollo principal |
| ML Framework | Kedro | 1.0.0 | Pipeline orchestration |
| ML Library | scikit-learn | 1.3+ | Modelos de ML |
| Data | Pandas | 2.0+ | Manipulación de datos |
| Data | NumPy | 1.24+ | Operaciones numéricas |
| Containerization | Docker | 20.10+ | Empaquetado |
| Orchestration | Airflow | 2.8.0 | Automatización |
| Database | PostgreSQL | 15 | Metadatos |
| Message Broker | Redis | latest | Comunicación |
| Visualization | Matplotlib | 3.7+ | Gráficos |
| Visualization | Seaborn | 0.12+ | Gráficos estadísticos |
| Testing | pytest | 7.4+ | Tests unitarios |
| Linting | flake8 | 6.0+ | Calidad de código |
| Formatting | black | 23.0+ | Formateo de código |

---

## 6. Conclusiones

### 6.1 Logros del Proyecto

1. **Objetivos cumplidos al 100%:**
   - ✅ Accuracy de clasificación: 98.56% (objetivo: > 90%)
   - ✅ R² de regresión: 0.7928 (objetivo: > 0.75)
   - ✅ Sistema deployable en producción

2. **Implementación profesional:**
   - Sistema modular con 5 pipelines
   - 33 nodos de procesamiento
   - Arquitectura de 3 capas (Docker, Kedro, Airflow)
   - Completamente automatizado

3. **Documentación exhaustiva:**
   - 20+ documentos técnicos
   - Más de 100 páginas de documentación
   - Guías paso a paso
   - Scripts de automatización

4. **Reproducibilidad:**
   - Dockerizado completamente
   - Código versionado en GitHub
   - Dependencias especificadas
   - Random state fijo (42)

### 6.2 Contribuciones

Este proyecto demuestra:

1. **Aplicación práctica de CRISP-DM:** Implementación completa end-to-end
2. **MLOps:** Integración de ML con prácticas de DevOps (Docker, CI/CD)
3. **Ingeniería de datos:** Pipelines modulares y mantenibles
4. **Producción:** Sistema listo para deployment real

### 6.3 Aprendizajes Clave

1. **Técnicos:**
   - Kedro facilita la modularización de pipelines complejos
   - Docker garantiza reproducibilidad entre ambientes
   - Airflow permite automatización de workflows
   - Los modelos basados en árboles dominan en datos tabulares

2. **Metodológicos:**
   - CRISP-DM estructura efectivamente proyectos de ML
   - El 80% del tiempo se invierte en preparación de datos
   - La documentación es tan importante como el código
   - El deployment es crítico para proyectos reales

3. **De negocio:**
   - Los factores económicos (oro) son más predictivos que métricas de combate
   - Las diferencias entre equipos son más informativas que valores absolutos
   - La ventaja temprana se compone exponencialmente

---

## 7. Referencias

### Bibliografía

1. Chapman, P., et al. (2000). CRISP-DM 1.0: Step-by-step data mining guide.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
3. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.
4. McKinney, W. (2017). Python for Data Analysis.

### Recursos Técnicos

1. Kedro Documentation: https://kedro.readthedocs.io/
2. Scikit-learn Documentation: https://scikit-learn.org/stable/
3. Apache Airflow Documentation: https://airflow.apache.org/docs/
4. Docker Documentation: https://docs.docker.com/

### Datos

1. League of Legends Worlds Dataset: Partidas profesionales del torneo mundial
2. Riot Games API: https://developer.riotgames.com/

---

## 8. Anexos

### Anexo A: Comandos de Ejecución

```bash
# Ejecución completa
kedro run

# Por pipeline
kedro run --pipeline data_cleaning
kedro run --pipeline data_exploration
kedro run --pipeline data_processing
kedro run --pipeline data_science
kedro run --pipeline evaluation

# Docker + Airflow
docker-compose up -d
```

### Anexo B: Estructura de Archivos

Ver archivo `README.md` para estructura completa del proyecto.

### Anexo C: Métricas Detalladas

Ver archivos en `data/08_reporting/`:
- `classification_report.json`
- `regression_report.json`
- `classification_metrics.parquet`
- `regression_metrics.parquet`

### Anexo D: Código Fuente

Todo el código está disponible en:
- GitHub: https://github.com/glYohanny/Evaluacion_machine_learning
- Carpeta `src/league_project/pipelines/`

---

## Declaración de Autoría

Yo, Pedro Torres (glYohanny), declaro que este trabajo es de mi autoría y ha sido desarrollado siguiendo las normas académicas de DuocUC. Todo el código, documentación y análisis presentados en este informe son producto de mi trabajo individual.

**Firma:** _____________________  
**Fecha:** Octubre 27, 2025

---

**Fin del Informe**


