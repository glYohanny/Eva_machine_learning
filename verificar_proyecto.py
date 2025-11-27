#!/usr/bin/env python
"""
Script de verificación del proyecto League of Legends ML.
Verifica que todos los componentes estén correctamente implementados.
"""

import os
import json
from pathlib import Path
import pandas as pd

def verificar_estructura():
    """Verifica que la estructura de carpetas esté correcta."""
    print("="*80)
    print("VERIFICANDO ESTRUCTURA DE CARPETAS")
    print("="*80)
    
    carpetas_requeridas = [
        "src/league_project/pipelines",
        "data/01_raw",
        "data/02_intermediate",
        "data/04_feature",
        "data/06_models",
        "data/08_reporting",
        "airflow/dags",
        "conf/base",
    ]
    
    errores = []
    for carpeta in carpetas_requeridas:
        if not os.path.exists(carpeta):
            errores.append(f"❌ Falta: {carpeta}")
        else:
            print(f"✅ {carpeta}")
    
    if errores:
        print("\n❌ ERRORES ENCONTRADOS:")
        for error in errores:
            print(f"   {error}")
        return False
    else:
        print("\n✅ Todas las carpetas requeridas existen")
        return True


def verificar_pipelines():
    """Verifica que todos los pipelines estén implementados."""
    print("\n" + "="*80)
    print("VERIFICANDO PIPELINES")
    print("="*80)
    
    pipelines_requeridos = [
        "data_cleaning",
        "data_exploration",
        "data_processing",
        "data_science",
        "evaluation",
        "unsupervised_learning",
    ]
    
    errores = []
    for pipeline in pipelines_requeridos:
        ruta = f"src/league_project/pipelines/{pipeline}"
        if not os.path.exists(ruta):
            errores.append(f"❌ Falta pipeline: {pipeline}")
        else:
            print(f"✅ Pipeline: {pipeline}")
    
    if errores:
        print("\n❌ ERRORES ENCONTRADOS:")
        for error in errores:
            print(f"   {error}")
        return False
    else:
        print("\n✅ Todos los pipelines están implementados")
        return True


def verificar_archivos_clave():
    """Verifica que los archivos clave existan."""
    print("\n" + "="*80)
    print("VERIFICANDO ARCHIVOS CLAVE")
    print("="*80)
    
    archivos_requeridos = [
        "requirements.txt",
        "pyproject.toml",
        "dvc.yaml",
        "docker-compose.yml",
        "Dockerfile",
        "Dockerfile.airflow",
        "airflow/dags/kedro_league_ml_dag.py",
        "src/league_project/pipeline_registry.py",
    ]
    
    errores = []
    for archivo in archivos_requeridos:
        if not os.path.exists(archivo):
            errores.append(f"❌ Falta: {archivo}")
        else:
            print(f"✅ {archivo}")
    
    if errores:
        print("\n❌ ERRORES ENCONTRADOS:")
        for error in errores:
            print(f"   {error}")
        return False
    else:
        print("\n✅ Todos los archivos clave existen")
        return True


def verificar_resultados():
    """Verifica que los resultados principales existan."""
    print("\n" + "="*80)
    print("VERIFICANDO RESULTADOS GENERADOS")
    print("="*80)
    
    resultados_importantes = [
        "data/08_reporting/classification_report.json",
        "data/08_reporting/regression_report.json",
        "data/08_reporting/classification_cv_comparison_table.csv",
        "data/08_reporting/regression_cv_comparison_table.csv",
        "data/08_reporting/kmeans_metrics.json",
        "data/08_reporting/pca_metrics.json",
    ]
    
    encontrados = 0
    faltantes = []
    
    for resultado in resultados_importantes:
        if os.path.exists(resultado):
            print(f"✅ {resultado}")
            encontrados += 1
        else:
            print(f"⚠️  No encontrado (ejecutar pipeline): {resultado}")
            faltantes.append(resultado)
    
    print(f"\n📊 Resultados encontrados: {encontrados}/{len(resultados_importantes)}")
    
    if faltantes:
        print("\n⚠️  Algunos resultados no están generados. Ejecuta:")
        print("   kedro run")
        return False
    else:
        print("\n✅ Todos los resultados principales están generados")
        return True


def verificar_dependencias():
    """Verifica que las dependencias principales estén en requirements.txt."""
    print("\n" + "="*80)
    print("VERIFICANDO DEPENDENCIAS")
    print("="*80)
    
    dependencias_requeridas = [
        "kedro",
        "scikit-learn",
        "pandas",
        "numpy",
        "dvc",
        "shap",  # Para interpretabilidad
    ]
    
    if not os.path.exists("requirements.txt"):
        print("❌ No se encontró requirements.txt")
        return False
    
    with open("requirements.txt", "r") as f:
        contenido = f.read().lower()
    
    faltantes = []
    for dep in dependencias_requeridas:
        if dep.lower() in contenido:
            print(f"✅ {dep}")
        else:
            print(f"⚠️  {dep} no encontrado en requirements.txt")
            faltantes.append(dep)
    
    if faltantes:
        print(f"\n⚠️  Dependencias faltantes: {', '.join(faltantes)}")
        return False
    else:
        print("\n✅ Todas las dependencias principales están en requirements.txt")
        return True


def main():
    """Ejecuta todas las verificaciones."""
    print("\n" + "="*80)
    print("VERIFICACIÓN COMPLETA DEL PROYECTO")
    print("League of Legends ML Project")
    print("="*80 + "\n")
    
    resultados = []
    
    resultados.append(("Estructura", verificar_estructura()))
    resultados.append(("Pipelines", verificar_pipelines()))
    resultados.append(("Archivos Clave", verificar_archivos_clave()))
    resultados.append(("Dependencias", verificar_dependencias()))
    resultados.append(("Resultados", verificar_resultados()))
    
    print("\n" + "="*80)
    print("RESUMEN DE VERIFICACIÓN")
    print("="*80)
    
    total = len(resultados)
    exitosos = sum(1 for _, resultado in resultados if resultado)
    
    for nombre, resultado in resultados:
        estado = "✅ PASS" if resultado else "❌ FAIL"
        print(f"{estado} - {nombre}")
    
    print("\n" + "="*80)
    print(f"TOTAL: {exitosos}/{total} verificaciones exitosas")
    print("="*80)
    
    if exitosos == total:
        print("\n🎉 ¡PROYECTO VERIFICADO CORRECTAMENTE!")
        print("✅ El proyecto está listo para la defensa técnica")
    else:
        print("\n⚠️  Algunas verificaciones fallaron")
        print("   Revisa los errores arriba y corrige los problemas")
    
    return exitosos == total


if __name__ == "__main__":
    exit(0 if main() else 1)

