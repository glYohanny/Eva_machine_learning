# Setup Script para Airflow + Kedro en Windows
# Sin emojis para compatibilidad con PowerShell

Write-Host "=========================================================================="
Write-Host "CONFIGURACION DE AIRFLOW + KEDRO EN WINDOWS"
Write-Host "=========================================================================="
Write-Host ""

# Verificar Docker Desktop
Write-Host "1. Verificando Docker Desktop..."
try {
    docker --version
    docker-compose --version
    Write-Host "   [OK] Docker instalado correctamente" -ForegroundColor Green
} catch {
    Write-Host "   [ERROR] Docker no esta instalado" -ForegroundColor Red
    Write-Host "   Descarga: https://www.docker.com/products/docker-desktop"
    exit 1
}

Write-Host ""
Write-Host "2. Creando variables de entorno..."
# Crear archivo .env
if (-not (Test-Path ".env")) {
    $envContent = "AIRFLOW_UID=50000`nAIRFLOW_PROJ_DIR=.`n_AIRFLOW_WWW_USER_USERNAME=admin`n_AIRFLOW_WWW_USER_PASSWORD=admin`nPOSTGRES_USER=airflow`nPOSTGRES_PASSWORD=airflow`nPOSTGRES_DB=airflow`nKEDRO_ENV=local"
    Set-Content -Path ".env" -Value $envContent
    Write-Host "   [OK] Archivo .env creado" -ForegroundColor Green
} else {
    Write-Host "   [INFO] Archivo .env ya existe" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "3. Creando directorios de Airflow..."
$directories = @("airflow\dags", "airflow\logs", "airflow\plugins", "airflow\config")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   [OK] Directorio creado: $dir" -ForegroundColor Green
    } else {
        Write-Host "   [INFO] Ya existe: $dir" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "4. Construyendo imagen de Kedro..."
Write-Host "   (Esto puede tomar varios minutos...)"
docker build -t league-kedro-ml:latest .
if ($LASTEXITCODE -eq 0) {
    Write-Host "   [OK] Imagen construida exitosamente" -ForegroundColor Green
} else {
    Write-Host "   [ERROR] Error construyendo imagen" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "5. Inicializando Airflow..."
Write-Host "   (Esto puede tomar unos minutos...)"
docker-compose up airflow-init
if ($LASTEXITCODE -eq 0) {
    Write-Host "   [OK] Airflow inicializado" -ForegroundColor Green
} else {
    Write-Host "   [ERROR] Error inicializando Airflow" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=========================================================================="
Write-Host "[COMPLETADO] Configuracion exitosa" -ForegroundColor Green
Write-Host "=========================================================================="
Write-Host ""
Write-Host "PROXIMOS PASOS:"
Write-Host ""
Write-Host "1. Iniciar servicios:"
Write-Host "   docker-compose up -d"
Write-Host ""
Write-Host "2. Acceder a Airflow:"
Write-Host "   URL: http://localhost:8080"
Write-Host "   Usuario: admin"
Write-Host "   Password: admin"
Write-Host ""
Write-Host "3. Ver logs:"
Write-Host "   docker-compose logs -f"
Write-Host ""
Write-Host "4. Detener servicios:"
Write-Host "   docker-compose down"
Write-Host ""
