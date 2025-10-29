"""
Script de demostración para cargar y usar los resultados del pipeline de Kedro.
Ejecutar después de: kedro run
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    """Función principal para demostrar los resultados."""
    
    print("="*80)
    print("DEMOSTRACIÓN DE RESULTADOS DEL PIPELINE KEDRO")
    print("="*80)
    
    # =========================================================================
    # 1. CARGAR REPORTES
    # =========================================================================
    
    print("\n📊 1. REPORTE DE REGRESIÓN")
    print("-"*80)
    
    with open('data/08_reporting/regression_report.json', 'r') as f:
        reg_report = json.load(f)
    
    print(f"Mejor modelo: {reg_report['best_model']}")
    print(f"R² Score: {reg_report['best_r2']:.4f}")
    print(f"RMSE: {reg_report['best_rmse']:.2f} minutos")
    print(f"MAE: {reg_report['best_mae']:.2f} minutos")
    
    if reg_report['top_features']:
        print(f"\nTop 5 features más importantes:")
        for i, feature in enumerate(reg_report['top_features'], 1):
            print(f"   {i}. {feature}")
    
    print("\n📊 2. REPORTE DE CLASIFICACIÓN")
    print("-"*80)
    
    with open('data/08_reporting/classification_report.json', 'r') as f:
        cls_report = json.load(f)
    
    print(f"Mejor modelo: {cls_report['best_model']}")
    print(f"Accuracy: {cls_report['best_accuracy']:.4f}")
    print(f"F1-Score: {cls_report['best_f1']:.4f}")
    print(f"AUC-ROC: {cls_report['best_auc']:.4f}")
    
    if cls_report['top_features']:
        print(f"\nTop 5 features más importantes:")
        for i, feature in enumerate(cls_report['top_features'], 1):
            print(f"   {i}. {feature}")
    
    # =========================================================================
    # 2. CARGAR MÉTRICAS DETALLADAS
    # =========================================================================
    
    print("\n\n📈 3. COMPARACIÓN DE TODOS LOS MODELOS")
    print("-"*80)
    
    # Métricas de regresión
    reg_metrics = pd.read_parquet('data/08_reporting/regression_metrics.parquet')
    print("\nModelos de Regresión (ordenados por R²):")
    print(reg_metrics[['model', 'test_r2', 'test_rmse', 'test_mae']].to_string(index=False))
    
    # Métricas de clasificación
    cls_metrics = pd.read_parquet('data/08_reporting/classification_metrics.parquet')
    print("\nModelos de Clasificación (ordenados por F1-Score):")
    print(cls_metrics[['model', 'test_accuracy', 'f1_score', 'auc_roc']].to_string(index=False))
    
    # =========================================================================
    # 3. FEATURE IMPORTANCE
    # =========================================================================
    
    print("\n\n🎯 4. IMPORTANCIA DE FEATURES")
    print("-"*80)
    
    reg_importance = pd.read_parquet('data/08_reporting/regression_feature_importance.parquet')
    
    # Top features de Random Forest para regresión
    rf_reg_importance = reg_importance[reg_importance['model'] == 'random_forest'].nlargest(10, 'importance')
    print("\nTop 10 Features - Random Forest Regressor:")
    for _, row in rf_reg_importance.iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    cls_importance = pd.read_parquet('data/08_reporting/classification_feature_importance.parquet')
    
    # Top features de Random Forest para clasificación
    rf_cls_importance = cls_importance[cls_importance['model'] == 'random_forest'].nlargest(10, 'importance')
    print("\nTop 10 Features - Random Forest Classifier:")
    for _, row in rf_cls_importance.iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    # =========================================================================
    # 4. CARGAR MODELOS Y HACER PREDICCIONES
    # =========================================================================
    
    print("\n\n🤖 5. EJEMPLO DE PREDICCIÓN CON MODELOS ENTRENADOS")
    print("-"*80)
    
    # Cargar modelos
    with open('data/06_models/regression_models.pkl', 'rb') as f:
        reg_models = pickle.load(f)
    
    with open('data/06_models/classification_models.pkl', 'rb') as f:
        cls_models = pickle.load(f)
    
    # Cargar scaler y datos de test
    with open('data/06_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    X_test = pd.read_parquet('data/04_feature/X_test_scaled.parquet')
    
    # Hacer predicción de ejemplo con una partida
    sample_idx = 0
    sample_features = X_test.iloc[sample_idx:sample_idx+1]
    
    print(f"\nEjemplo de partida (índice {sample_idx}):")
    print(f"Features: {sample_features.to_dict('records')[0]}")
    
    # Predicción de duración (mejor modelo de regresión)
    best_reg_model_name = reg_report['best_model']
    best_reg_model = reg_models[best_reg_model_name]
    predicted_duration = best_reg_model.predict(sample_features)[0]
    
    print(f"\n🎯 Predicción de duración ({best_reg_model_name}):")
    print(f"   Duración predicha: {predicted_duration:.1f} minutos")
    
    # Predicción de resultado (mejor modelo de clasificación)
    best_cls_model_name = cls_report['best_model']
    best_cls_model = cls_models[best_cls_model_name]
    predicted_result = best_cls_model.predict(sample_features)[0]
    predicted_proba = best_cls_model.predict_proba(sample_features)[0]
    
    print(f"\n🎯 Predicción de resultado ({best_cls_model_name}):")
    print(f"   Resultado predicho: {'Victoria Azul' if predicted_result == 1 else 'Derrota Azul'}")
    print(f"   Probabilidad Victoria Azul: {predicted_proba[1]:.2%}")
    print(f"   Probabilidad Derrota Azul: {predicted_proba[0]:.2%}")
    
    # =========================================================================
    # 5. ESTADÍSTICAS GENERALES
    # =========================================================================
    
    print("\n\n📊 6. ESTADÍSTICAS DEL DATASET")
    print("-"*80)
    
    model_input = pd.read_parquet('data/03_primary/model_input_table.parquet')
    
    print(f"Total de partidas: {len(model_input):,}")
    print(f"Duración promedio: {model_input['gamelength'].mean():.1f} minutos")
    print(f"Duración mínima: {model_input['gamelength'].min():.1f} minutos")
    print(f"Duración máxima: {model_input['gamelength'].max():.1f} minutos")
    
    win_rate = model_input['bResult'].mean()
    print(f"\nWin rate equipo azul: {win_rate:.2%}")
    
    print("\n" + "="*80)
    print("✓ DEMOSTRACIÓN COMPLETADA")
    print("="*80)
    
    return {
        'regression_report': reg_report,
        'classification_report': cls_report,
        'regression_metrics': reg_metrics,
        'classification_metrics': cls_metrics,
    }


if __name__ == "__main__":
    results = main()




