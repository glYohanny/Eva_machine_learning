# 📍 UBICACIÓN DE ARCHIVOS - EVALUACIÓN PARCIAL 3
## Aprendizaje No Supervisado + Integración Completa

---

## 🎯 RESUMEN RÁPIDO

Este documento indica dónde encontrar todos los archivos relacionados con la **Evaluación Parcial 3: Aprendizaje No Supervisado**.

---

## 📁 ESTRUCTURA DE ARCHIVOS

### 1️⃣ **CÓDIGO FUENTE - PIPELINES**

#### **Pipeline Principal de Aprendizaje No Supervisado**
```
league-project/src/league_project/pipelines/unsupervised_learning/
├── __init__.py                                    # Exporta create_pipeline
├── pipeline.py                                    # Pipeline principal integrado
└── nodes.py                                       # Nodos de integración
```

#### **Clustering (4 algoritmos)**
```
league-project/src/league_project/pipelines/unsupervised_learning/clustering/
├── __init__.py
├── pipeline.py                                    # Pipeline de clustering
└── nodes.py                                       # K-Means, DBSCAN, Hierarchical, GMM
```

#### **Reducción de Dimensionalidad (3 técnicas)**
```
league-project/src/league_project/pipelines/unsupervised_learning/dimensionality_reduction/
├── __init__.py
├── pipeline.py                                    # Pipeline de reducción dimensional
└── nodes.py                                       # PCA, t-SNE, UMAP
```

#### **Detección de Anomalías (2 algoritmos)**
```
league-project/src/league_project/pipelines/unsupervised_learning/anomaly_detection/
├── __init__.py
├── pipeline.py                                    # Pipeline de anomalías
└── nodes.py                                       # Isolation Forest, LOF
```

---

### 2️⃣ **CONFIGURACIÓN**

#### **Parámetros**
```
league-project/conf/base/parameters.yml
```
**Sección:** `unsupervised_learning`
- Clustering: n_clusters, k_range, dbscan_eps, linkage_method
- Dimensionality Reduction: pca_n_components, tsne_perplexity, umap_n_neighbors
- Anomaly Detection: contamination, lof_n_neighbors

#### **Catálogo de Datasets**
```
league-project/conf/base/catalog.yml
```
**Secciones agregadas:**
- `# APRENDIZAJE NO SUPERVISADO - CLUSTERING` (línea ~305)
- `# APRENDIZAJE NO SUPERVISADO - REDUCCIÓN DE DIMENSIONALIDAD` (línea ~340)
- `# APRENDIZAJE NO SUPERVISADO - DETECCIÓN DE ANOMALÍAS` (línea ~365)
- `# INTEGRACIÓN: CLUSTERING COMO FEATURE ENGINEERING` (línea ~390)

#### **Registro de Pipelines**
```
league-project/src/league_project/pipeline_registry.py
```
**Línea 10:** Import de `unsupervised_learning`
**Línea 33:** Creación de `unsupervised_pipeline`
**Línea 36:** Integración en `default_pipeline`

---

### 3️⃣ **DATOS GENERADOS**

#### **Modelos Entrenados** (data/06_models/)
```
league-project/data/06_models/
├── kmeans_model.pkl                              # Modelo K-Means
├── dbscan_model.pkl                              # Modelo DBSCAN
├── hierarchical_model.pkl                        # Modelo Hierarchical
├── gmm_model.pkl                                 # Modelo GMM
├── pca_model.pkl                                 # Modelo PCA
├── tsne_model.pkl                                # Modelo t-SNE
├── umap_model.pkl                                # Modelo UMAP
├── isolation_forest_model.pkl                    # Modelo Isolation Forest
└── lof_model.pkl                                 # Modelo LOF
```

#### **Outputs de Modelos** (data/07_model_output/)
```
league-project/data/07_model_output/
├── kmeans_labels.pkl                             # Etiquetas K-Means
├── dbscan_labels.pkl                             # Etiquetas DBSCAN
├── hierarchical_labels.pkl                       # Etiquetas Hierarchical
├── hierarchical_linkage_matrix.pkl               # Matriz de linkage
├── gmm_labels.pkl                                # Etiquetas GMM
├── X_pca.parquet                                 # Datos transformados PCA
├── X_tsne.parquet                                # Datos transformados t-SNE
├── X_umap.parquet                                # Datos transformados UMAP
├── isolation_forest_labels.pkl                   # Etiquetas anomalías (IF)
└── lof_labels.pkl                                # Etiquetas anomalías (LOF)
```

#### **Features Integradas** (data/04_feature/)
```
league-project/data/04_feature/
├── X_train_with_clusters.parquet                 # Train con features de clustering
└── X_test_with_clusters.parquet                  # Test con features de clustering
```

#### **Reportes y Métricas** (data/08_reporting/)
```
league-project/data/08_reporting/
├── elbow_method_results.json                     # Resultados método del codo
├── kmeans_metrics.json                           # Métricas K-Means
├── dbscan_metrics.json                           # Métricas DBSCAN
├── hierarchical_metrics.json                     # Métricas Hierarchical
├── gmm_metrics.json                              # Métricas GMM
├── clustering_comparison_table.csv               # Comparación de algoritmos
├── cluster_patterns_analysis.parquet             # Análisis de patrones por cluster
├── pca_metrics.json                              # Métricas PCA
├── pca_loadings_analysis.parquet                 # Loadings de componentes
├── tsne_metrics.json                             # Métricas t-SNE
├── umap_metrics.json                             # Métricas UMAP
├── isolation_forest_metrics.json                 # Métricas Isolation Forest
├── lof_metrics.json                              # Métricas LOF
├── anomaly_analysis.parquet                      # Análisis de anomalías
└── anomaly_detection_comparison.csv              # Comparación de métodos
```

---

### 4️⃣ **NOTEBOOKS DE ANÁLISIS**

```
league-project/notebooks/
└── 05_unsupervised_learning.ipynb                # Notebook de análisis no supervisado
```

---

### 5️⃣ **ORQUESTACIÓN Y VERSIONADO**

#### **Airflow DAG**
```
league-project/airflow/dags/kedro_league_ml_dag.py
```
**Línea 58-63:** Task `unsupervised_learning_task`
**Línea 101:** Dependencia en el flujo: `data_processing_task >> unsupervised_learning_task`

#### **DVC Pipeline**
```
league-project/dvc.yaml
```
**Línea 79-130:** Stage `unsupervised_learning` con todas las dependencias y outputs

---

### 6️⃣ **DEPENDENCIAS**

```
league-project/requirements.txt
```
**Líneas agregadas:**
- `plotly>=5.0.0`
- `umap-learn>=0.5.0`
- `pyod>=1.1.0`
- `mlxtend>=0.22.0`
- `hdbscan>=0.8.0`
- `shap>=0.42.0`

---

### 7️⃣ **DOCUMENTACIÓN**

```
league-project/README.md
```
**Actualizado con:**
- Sección de Aprendizaje No Supervisado
- Estructura de pipelines actualizada
- Pipeline #5: unsupervised_learning

---

## 🚀 COMANDOS PARA EJECUTAR

### **Ejecutar solo aprendizaje no supervisado:**
```bash
cd league-project
kedro run --pipeline unsupervised_learning
```

### **Ejecutar pipeline completo (incluye EP3):**
```bash
cd league-project
kedro run
```

### **Ver pipeline en Kedro Viz:**
```bash
cd league-project
kedro viz
```

---

## 📊 CHECKLIST DE ENTREGABLES EP3

### ✅ **Clustering (OBLIGATORIO)**
- [x] K-Means: `src/.../clustering/nodes.py` (línea 60-121)
- [x] DBSCAN: `src/.../clustering/nodes.py` (línea 123-186)
- [x] Hierarchical: `src/.../clustering/nodes.py` (línea 188-246)
- [x] GMM: `src/.../clustering/nodes.py` (línea 248-299)
- [x] Métricas: Silhouette, Davies-Bouldin, Calinski-Harabasz
- [x] Método del codo: `src/.../clustering/nodes.py` (línea 20-67)
- [x] Análisis de patrones: `src/.../clustering/nodes.py` (línea 301-356)

### ✅ **Reducción de Dimensionalidad (OBLIGATORIO)**
- [x] PCA: `src/.../dimensionality_reduction/nodes.py` (línea 20-78)
- [x] t-SNE: `src/.../dimensionality_reduction/nodes.py` (línea 80-144)
- [x] UMAP: `src/.../dimensionality_reduction/nodes.py` (línea 146-201)
- [x] Análisis de componentes: `src/.../dimensionality_reduction/nodes.py` (línea 203-249)

### ✅ **Detección de Anomalías (OPCIONAL - Puntos extra)**
- [x] Isolation Forest: `src/.../anomaly_detection/nodes.py` (línea 20-72)
- [x] LOF: `src/.../anomaly_detection/nodes.py` (línea 74-126)
- [x] Comparación: `src/.../anomaly_detection/nodes.py` (línea 194-230)

### ✅ **Integración con Supervisados**
- [x] Features de clustering: `src/.../unsupervised_learning/nodes.py` (línea 14-54)
- [x] Pipeline integrado: `src/.../unsupervised_learning/pipeline.py`

### ✅ **Orquestación**
- [x] Airflow DAG actualizado: `airflow/dags/kedro_league_ml_dag.py`
- [x] DVC actualizado: `dvc.yaml`

### ✅ **Documentación**
- [x] README actualizado: `README.md`
- [x] Notebook de análisis: `notebooks/05_unsupervised_learning.ipynb`

---

## 📝 NOTAS IMPORTANTES

1. **Todos los modelos se guardan en:** `data/06_models/*.pkl`
2. **Todas las métricas se guardan en:** `data/08_reporting/*.json`
3. **Todas las comparaciones se guardan en:** `data/08_reporting/*.csv`
4. **Features integradas se guardan en:** `data/04_feature/X_*_with_clusters.parquet`

---

## 🔍 VERIFICACIÓN RÁPIDA

Para verificar que todo está implementado:

```bash
# Verificar que existen los pipelines
ls league-project/src/league_project/pipelines/unsupervised_learning/

# Verificar modelos generados
ls league-project/data/06_models/*.pkl

# Verificar métricas generadas
ls league-project/data/08_reporting/*.json

# Verificar que el pipeline está registrado
grep -n "unsupervised_learning" league-project/src/league_project/pipeline_registry.py
```

---

**Última actualización:** 26 de Noviembre, 2025  
**Estado:** ✅ COMPLETO - Listo para evaluación


