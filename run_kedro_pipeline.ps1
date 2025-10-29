# ============================================================================
# Script para ejecutar pipelines de Kedro en Docker
# ============================================================================

param(
    [Parameter(Mandatory=$false)]
    [string]$Pipeline = "default",
    [switch]$Help
)

if ($Help) {
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host "EJECUTAR PIPELINES DE KEDRO EN DOCKER" -ForegroundColor Cyan
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "USO:" -ForegroundColor Yellow
    Write-Host "  .\run_kedro_pipeline.ps1 [-Pipeline <nombre>]" -ForegroundColor White
    Write-Host ""
    Write-Host "PIPELINES DISPONIBLES:" -ForegroundColor Yellow
    Write-Host "  default             - Pipeline completo (todos los pasos)" -ForegroundColor White
    Write-Host "  eda                 - Limpieza + Exploración" -ForegroundColor White
    Write-Host "  data_cleaning       - Solo limpieza de datos" -ForegroundColor White
    Write-Host "  data_exploration    - Solo exploración" -ForegroundColor White
    Write-Host "  data_processing     - Feature engineering" -ForegroundColor White
    Write-Host "  data_science        - Entrenamiento de modelos" -ForegroundColor White
    Write-Host "  evaluation          - Evaluación de modelos" -ForegroundColor White
    Write-Host ""
    Write-Host "EJEMPLOS:" -ForegroundColor Yellow
    Write-Host "  .\run_kedro_pipeline.ps1" -ForegroundColor Cyan
    Write-Host "  .\run_kedro_pipeline.ps1 -Pipeline eda" -ForegroundColor Cyan
    Write-Host "  .\run_kedro_pipeline.ps1 -Pipeline data_science" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "EJECUTANDO PIPELINE DE KEDRO: $Pipeline" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Construir comando
if ($Pipeline -eq "default") {
    $command = "kedro run"
} else {
    $command = "kedro run --pipeline $Pipeline"
}

Write-Host "Comando: $command" -ForegroundColor Yellow
Write-Host ""

# Ejecutar en Docker
docker-compose run --rm kedro-app $command

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host "✅ PIPELINE EJECUTADO EXITOSAMENTE" -ForegroundColor Green
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host "❌ ERROR EN LA EJECUCIÓN DEL PIPELINE" -ForegroundColor Red
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Revisa los logs para más detalles:" -ForegroundColor Yellow
    Write-Host "  docker-compose logs kedro-app" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}


