"""
Script de Verificación de Pipelines
Verifica que todos los archivos fueron generados correctamente después de ejecutar los pipelines.
"""

import os
import pandas as pd
import json


def verificar_archivos():
    """Verifica que todos los archivos fueron generados."""
    
    archivos_esperados = {
        'Datos Limpios (02_intermediate)': [
            'data/02_intermediate/main_clean.csv',
            'data/02_intermediate/matchinfo_clean.csv',
            'data/02_intermediate/bans_clean.csv',
            'data/02_intermediate/gold_clean.csv',
            'data/02_intermediate/kills_clean.csv',
            'data/02_intermediate/monsters_clean.csv',
            'data/02_intermediate/structures_clean.csv',
        ],
        'Reportes de Análisis (08_reporting)': [
            'data/08_reporting/data_quality_report_cleaning.csv',
            'data/08_reporting/descriptive_statistics.csv',
            'data/08_reporting/team_performance_analysis.csv',
            'data/08_reporting/champion_bans_analysis.csv',
            'data/08_reporting/neutral_objectives_analysis.csv',
            'data/08_reporting/structures_analysis.csv',
            'data/08_reporting/correlations_analysis.csv',
            'data/08_reporting/game_duration_analysis.csv',
            'data/08_reporting/eda_complete_report.json',
        ]
    }
    
    print("="*70)
    print(" "*15 + "VERIFICACIÓN DE PIPELINES")
    print("="*70)
    
    todos_existen = True
    total_archivos = 0
    total_encontrados = 0
    
    for categoria, archivos in archivos_esperados.items():
        print(f"\n📁 {categoria}")
        print("-"*70)
        
        for archivo in archivos:
            total_archivos += 1
            existe = os.path.exists(archivo)
            
            if existe:
                total_encontrados += 1
                estado = "✅"
            else:
                estado = "❌"
                todos_existen = False
            
            nombre_archivo = os.path.basename(archivo)
            print(f"{estado} {nombre_archivo:40s}", end="")
            
            if existe:
                try:
                    if archivo.endswith('.csv'):
                        df = pd.read_csv(archivo)
                        print(f" [{len(df):>6,} filas, {len(df.columns):>2} cols]")
                    elif archivo.endswith('.json'):
                        with open(archivo, 'r') as f:
                            data = json.load(f)
                        print(f" [JSON con {len(data)} keys]")
                except Exception as e:
                    print(f" [⚠️  Error: {str(e)[:30]}]")
            else:
                print(" [NO ENCONTRADO]")
    
    print("\n" + "="*70)
    print(f"📊 Resumen: {total_encontrados}/{total_archivos} archivos encontrados")
    
    if todos_existen:
        print("✅ ¡TODOS LOS ARCHIVOS FUERON GENERADOS CORRECTAMENTE!")
    else:
        print("❌ FALTAN ALGUNOS ARCHIVOS")
        print("\n💡 Solución:")
        print("   1. Ejecutar: kedro run --pipeline data_cleaning")
        print("   2. Ejecutar: kedro run --pipeline data_exploration")
        print("   O simplemente: kedro run --pipeline eda")
    
    print("="*70)
    
    return todos_existen


def mostrar_insights():
    """Muestra insights rápidos de los datos si existen."""
    
    print("\n" + "="*70)
    print(" "*20 + "INSIGHTS RÁPIDOS")
    print("="*70)
    
    try:
        # Top 5 equipos
        teams_file = 'data/08_reporting/team_performance_analysis.csv'
        if os.path.exists(teams_file):
            teams = pd.read_csv(teams_file)
            if 'win_rate' in teams.columns and 'team' in teams.columns:
                print("\n🏆 Top 5 Equipos por Win Rate:")
                top5 = teams.nlargest(5, 'win_rate')[['team', 'win_rate', 'total_games']]
                for idx, row in top5.iterrows():
                    print(f"   {row['team']:20s} - {row['win_rate']:.2%} ({int(row['total_games'])} juegos)")
        
        # Top 5 campeones baneados
        bans_file = 'data/08_reporting/champion_bans_analysis.csv'
        if os.path.exists(bans_file):
            bans = pd.read_csv(bans_file)
            if 'champion' in bans.columns and 'ban_count' in bans.columns:
                print("\n🚫 Top 5 Campeones Más Baneados:")
                top5 = bans.head(5)[['champion', 'ban_count', 'ban_percentage']]
                for idx, row in top5.iterrows():
                    print(f"   {row['champion']:20s} - {int(row['ban_count']):>4} bans ({row['ban_percentage']:.1f}%)")
        
        # Duración promedio
        duration_file = 'data/08_reporting/game_duration_analysis.csv'
        if os.path.exists(duration_file):
            duration = pd.read_csv(duration_file)
            mean_row = duration[duration['metric'] == 'mean']
            if not mean_row.empty:
                mean_duration = mean_row['value'].iloc[0]
                print(f"\n⏱️  Duración Promedio de Partidos: {mean_duration:.1f} minutos")
        
        # Reporte completo
        report_file = 'data/08_reporting/eda_complete_report.json'
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            if 'key_insights' in report and report['key_insights']:
                print("\n📈 Insights Clave:")
                for insight in report['key_insights']:
                    print(f"   • {insight}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n⚠️  Error al cargar insights: {str(e)}")


if __name__ == "__main__":
    print("\n")
    exito = verificar_archivos()
    
    if exito:
        mostrar_insights()
    
    print("\n💡 Para más detalles, revisa los archivos en data/08_reporting/\n")

