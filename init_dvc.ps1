# ========================================================================
# Script de Inicializacion de DVC para League of Legends ML Project
# ========================================================================
# Este script configura DVC para versionar datos, modelos y metricas

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "INICIALIZACION DE DVC - League of Legends ML Project" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

# Verificar si DVC esta instalado
Write-Host "`n[1/6] Verificando instalacion de DVC..." -ForegroundColor Yellow
$dvcVersion = dvc version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "DVC no esta instalado. Instalando..." -ForegroundColor Red
    pip install dvc
} else {
    Write-Host "DVC ya esta instalado: $dvcVersion" -ForegroundColor Green
}

# Verificar si DVC ya esta inicializado
Write-Host "`n[2/6] Verificando si DVC ya esta inicializado..." -ForegroundColor Yellow
if (Test-Path ".dvc") {
    Write-Host "DVC ya esta inicializado en este proyecto." -ForegroundColor Green
} else {
    Write-Host "Inicializando DVC..." -ForegroundColor Yellow
    dvc init
    Write-Host "DVC inicializado correctamente." -ForegroundColor Green
}

# Configurar remote storage (local por defecto)
Write-Host "`n[3/6] Configurando remote storage..." -ForegroundColor Yellow
$remotePath = "../dvc_storage"
if (!(Test-Path $remotePath)) {
    New-Item -ItemType Directory -Path $remotePath -Force | Out-Null
    Write-Host "Directorio de storage creado: $remotePath" -ForegroundColor Green
}

# Verificar si el remote ya existe
$existingRemote = dvc remote list 2>$null | Select-String "local_storage"
if ($existingRemote) {
    Write-Host "Remote 'local_storage' ya existe." -ForegroundColor Green
} else {
    dvc remote add -d local_storage $remotePath
    Write-Host "Remote 'local_storage' configurado en: $remotePath" -ForegroundColor Green
}

# Trackear archivos de datos raw
Write-Host "`n[4/6] Trackeando archivos de datos raw con DVC..." -ForegroundColor Yellow
$rawFiles = @(
    "data/01_raw/LeagueofLegends.csv",
    "data/01_raw/matchinfo.csv",
    "data/01_raw/kills.csv",
    "data/01_raw/gold.csv",
    "data/01_raw/bans.csv",
    "data/01_raw/monsters.csv",
    "data/01_raw/structures.csv",
    "data/01_raw/_columns.csv"
)

foreach ($file in $rawFiles) {
    if (Test-Path $file) {
        if (!(Test-Path "$file.dvc")) {
            Write-Host "  Trackeando: $file" -ForegroundColor Cyan
            dvc add $file
        } else {
            Write-Host "  Ya trackeado: $file" -ForegroundColor Gray
        }
    } else {
        Write-Host "  Archivo no encontrado: $file" -ForegroundColor Red
    }
}

# Agregar archivos .dvc a Git
Write-Host "`n[5/6] Agregando archivos .dvc y .gitignore a Git..." -ForegroundColor Yellow
git add data/01_raw/*.dvc .gitignore .dvc/config dvc.yaml 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Archivos DVC agregados a Git." -ForegroundColor Green
} else {
    Write-Host "No hay cambios nuevos para agregar a Git." -ForegroundColor Yellow
}

# Mostrar resumen
Write-Host "`n[6/6] Resumen de configuracion DVC:" -ForegroundColor Yellow
Write-Host "========================================================================" -ForegroundColor Cyan

# Verificar archivos trackeados
$dvcFiles = Get-ChildItem -Path "data/01_raw" -Filter "*.dvc" -ErrorAction SilentlyContinue
Write-Host "`nArchivos trackeados con DVC: $($dvcFiles.Count)" -ForegroundColor Green
foreach ($file in $dvcFiles) {
    Write-Host "  - $($file.Name)" -ForegroundColor White
}

# Mostrar comandos utiles
Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host "COMANDOS UTILES DE DVC:" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  dvc dag                  # Ver grafo de dependencias" -ForegroundColor White
Write-Host "  dvc repro                # Reproducir pipeline completo" -ForegroundColor White
Write-Host "  dvc metrics show         # Ver metricas" -ForegroundColor White
Write-Host "  dvc metrics diff         # Comparar metricas entre versiones" -ForegroundColor White
Write-Host "  dvc push                 # Subir datos a remote" -ForegroundColor White
Write-Host "  dvc pull                 # Descargar datos desde remote" -ForegroundColor White
Write-Host "  dvc status               # Ver estado de archivos" -ForegroundColor White
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "DVC CONFIGURADO CORRECTAMENTE" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "PROXIMO PASO: Ejecutar 'dvc repro' para reproducir el pipeline" -ForegroundColor Yellow
Write-Host ""

