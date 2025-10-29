"""
Script rapido para visualizar resultados de modelos de Kedro
Compatible con PowerShell (sin emojis)
"""

import json
import pandas as pd
from pathlib import Path

def mostrar_resultados_clasificacion():
    """Muestra resultados de modelos de clasificacion"""
    path = Path("data/08_reporting/classification_report.json")
    
    if not path.exists():
        print("[X] No se encontraron resultados de clasificacion")
        print("    Ejecuta: kedro run")
        return
    
    with open(path) as f:
        data = json.load(f)
    
    print("\n" + "="*70)
    print("RESULTADOS DE CLASIFICACION (Prediccion de Ganador)")
    print("="*70)
    
    # Mejor modelo
    print(f"\n[*] Mejor Modelo: {data['best_model'].upper()}")
    print(f"    Accuracy:  {data['best_accuracy']:.4f} ({data['best_accuracy']*100:.2f}%)")
    print(f"    F1-Score:  {data['best_f1']:.4f}")
    print(f"    AUC-ROC:   {data['best_auc']:.4f}")
    
    # Tabla de todos los modelos
    print(f"\n[+] Comparacion de Todos los Modelos:")
    print("-" * 70)
    
    df = pd.DataFrame(data['all_metrics'])
    df = df[['model', 'test_accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']]
    df.columns = ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    # Formatear numeros
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")
    
    print(df.to_string(index=False))
    
    # Top features
    print(f"\n[!] Top Features Importantes:")
    for i, feature in enumerate(data['top_features'][:5], 1):
        print(f"    {i}. {feature}")


def mostrar_resultados_regresion():
    """Muestra resultados de modelos de regresion"""
    path = Path("data/08_reporting/regression_report.json")
    
    if not path.exists():
        print("[X] No se encontraron resultados de regresion")
        print("    Ejecuta: kedro run")
        return
    
    with open(path) as f:
        data = json.load(f)
    
    print("\n" + "="*70)
    print("RESULTADOS DE REGRESION (Prediccion de Duracion)")
    print("="*70)
    
    # Mejor modelo
    print(f"\n[*] Mejor Modelo: {data['best_model'].upper()}")
    print(f"    R2 Score:  {data['best_r2']:.4f} (explica {data['best_r2']*100:.1f}% de la varianza)")
    print(f"    RMSE:      {data['best_rmse']:.2f} minutos (error cuadratico)")
    print(f"    MAE:       {data['best_mae']:.2f} minutos (error promedio)")
    
    # Tabla de todos los modelos
    print(f"\n[+] Comparacion de Todos los Modelos:")
    print("-" * 70)
    
    df = pd.DataFrame(data['all_metrics'])
    df = df[['model', 'test_r2', 'test_rmse', 'test_mae']]
    df.columns = ['Modelo', 'R2 Score', 'RMSE', 'MAE']
    
    # Formatear numeros
    df['R2 Score'] = df['R2 Score'].apply(lambda x: f"{x:.4f}")
    df['RMSE'] = df['RMSE'].apply(lambda x: f"{x:.2f} min")
    df['MAE'] = df['MAE'].apply(lambda x: f"{x:.2f} min")
    
    print(df.to_string(index=False))
    
    # Top features
    print(f"\n[!] Top Features Importantes:")
    for i, feature in enumerate(data['top_features'][:5], 1):
        print(f"    {i}. {feature}")
    
    # Interpretacion
    print(f"\n[i] Interpretacion:")
    print(f"    Si una partida real dura 35 minutos,")
    print(f"    el modelo predecira entre {35 - data['best_mae']:.0f}-{35 + data['best_mae']:.0f} minutos.")


def mostrar_resumen():
    """Muestra resumen general"""
    print("\n" + "="*70)
    print("RESUMEN GENERAL DEL PROYECTO")
    print("="*70)
    
    print("\n[+] Archivos de resultados encontrados:")
    
    files = [
        ("data/08_reporting/classification_report.json", "Clasificacion"),
        ("data/08_reporting/regression_report.json", "Regresion"),
        ("data/08_reporting/eda_complete_report.json", "EDA"),
        ("data/08_reporting/team_performance_analysis.csv", "Analisis de Equipos")
    ]
    
    for filepath, name in files:
        if Path(filepath).exists():
            print(f"    [OK] {name}")
        else:
            print(f"    [X] {name} (falta)")
    
    print("\n[+] Modelos entrenados:")
    models_dir = Path("data/06_models")
    if models_dir.exists():
        models = list(models_dir.glob("*.pkl"))
        if models:
            for model in models:
                print(f"    [OK] {model.name}")
        else:
            print("    [X] No hay modelos entrenados")
            print("         Ejecuta: kedro run")
    else:
        print("    [X] Directorio de modelos no existe")


def main():
    """Funcion principal"""
    print("\n" + "="*70)
    print("LEAGUE OF LEGENDS ML - VISUALIZACION DE RESULTADOS")
    print("="*70)
    
    mostrar_resumen()
    mostrar_resultados_clasificacion()
    mostrar_resultados_regresion()
    
    print("\n" + "="*70)
    print("VISUALIZACION COMPLETADA")
    print("="*70)
    
    print("\n[i] Otras formas de visualizar:")
    print("    1. Dashboard interactivo: streamlit run dashboard_ml.py")
    print("    2. Jupyter notebook: kedro jupyter notebook")
    print("    3. Archivos JSON: cat data/08_reporting/*.json")
    print()


if __name__ == "__main__":
    main()
