"""
Script para ver y explorar los datos procesados del pipeline de Kedro.
"""

import pandas as pd
import pickle

print("="*80)
print("VISUALIZACIÓN DE DATOS PROCESADOS")
print("="*80)

# 1. DATOS LIMPIOS FINALES (con todas las features)
print("\n📊 1. DATOS LIMPIOS FINALES (model_input_table)")
print("-"*80)
df_clean = pd.read_parquet('../data/03_primary/model_input_table.parquet')
print(f"Shape: {df_clean.shape}")
print(f"\nPrimeras 10 filas:")
print(df_clean.head(10))
print(f"\nColumnas: {list(df_clean.columns)}")
print(f"\nEstadísticas descriptivas:")
print(df_clean.describe())

# 2. DATOS DE ENTRENAMIENTO
print("\n\n📊 2. DATOS DE ENTRENAMIENTO (X_train)")
print("-"*80)
X_train = pd.read_parquet('../data/04_feature/X_train.parquet')
print(f"Shape: {X_train.shape}")
print(f"\nPrimeras 5 filas:")
print(X_train.head())

# 3. DATOS DE TEST
print("\n\n📊 3. DATOS DE TEST (X_test)")
print("-"*80)
X_test = pd.read_parquet('../data/04_feature/X_test.parquet')
print(f"Shape: {X_test.shape}")
print(f"\nPrimeras 5 filas:")
print(X_test.head())

# 4. TARGETS DE REGRESIÓN
print("\n\n🎯 4. TARGETS DE REGRESIÓN (gamelength)")
print("-"*80)
with open('../data/04_feature/y_reg_train.pkl', 'rb') as f:
    y_reg_train = pickle.load(f)
print(f"Train - Shape: {y_reg_train.shape}")
print(f"Train - Primeros 10 valores: {y_reg_train.head(10).values}")
print(f"Train - Estadísticas: Min={y_reg_train.min():.1f}, Max={y_reg_train.max():.1f}, Media={y_reg_train.mean():.1f}")

# 5. TARGETS DE CLASIFICACIÓN
print("\n\n🎯 5. TARGETS DE CLASIFICACIÓN (bResult)")
print("-"*80)
with open('../data/04_feature/y_cls_train.pkl', 'rb') as f:
    y_cls_train = pickle.load(f)
print(f"Train - Shape: {y_cls_train.shape}")
print(f"Train - Distribución:")
print(y_cls_train.value_counts())
print(f"Train - Porcentaje de victorias: {y_cls_train.mean()*100:.2f}%")

# 6. OPCIÓN: EXPORTAR A CSV
print("\n\n💾 ¿DESEAS EXPORTAR A CSV?")
print("-"*80)
respuesta = input("¿Exportar datos limpios a CSV? (s/n): ")

if respuesta.lower() == 's':
    # Exportar datos limpios
    csv_path = '../data/03_primary/model_input_table.csv'
    df_clean.to_csv(csv_path, index=False)
    print(f"✅ Datos exportados a: {csv_path}")
    
    # Exportar train/test
    X_train.to_csv('../data/04_feature/X_train.csv', index=False)
    X_test.to_csv('../data/04_feature/X_test.csv', index=False)
    print(f"✅ X_train exportado a: ../data/04_feature/X_train.csv")
    print(f"✅ X_test exportado a: ../data/04_feature/X_test.csv")
    
    # Exportar targets
    pd.DataFrame({'gamelength': y_reg_train}).to_csv('../data/04_feature/y_reg_train.csv', index=False)
    pd.DataFrame({'bResult': y_cls_train}).to_csv('../data/04_feature/y_cls_train.csv', index=False)
    print(f"✅ Targets exportados")
    
    print("\n🎉 ¡Todos los datos exportados a CSV exitosamente!")
    print("   Ahora puedes abrirlos con Excel o cualquier editor de texto.")
else:
    print("❌ Exportación cancelada")

print("\n" + "="*80)
print("✓ SCRIPT COMPLETADO")
print("="*80)

import pandas as pd

# Ver datos limpios
df = pd.read_parquet('../data/03_primary/model_input_table.parquet')
df.head(20)  # Ver primeras 20 filas




