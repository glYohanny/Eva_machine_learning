# ============================================================================
# Dockerfile para el Proyecto Kedro de League of Legends ML
# ============================================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="League of Legends ML Team"
LABEL description="Kedro ML Pipeline for League of Legends Worlds Analysis"

# Configurar variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KEDRO_LOGGING_CONFIG=/app/conf/logging.yml

# Crear directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requerimientos primero (para aprovechar cache de Docker)
COPY requirements.txt .
COPY pyproject.toml .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install kedro-airflow kedro-docker

# Copiar el código del proyecto
COPY . .

# Crear directorios de datos si no existen
RUN mkdir -p data/01_raw \
             data/02_intermediate \
             data/03_primary \
             data/04_feature \
             data/05_model_input \
             data/06_models \
             data/07_model_output \
             data/08_reporting \
             logs

# Dar permisos de ejecución
RUN chmod -R 755 /app

# Exponer puerto (si se necesita en el futuro)
EXPOSE 8080

# Usuario no-root para seguridad
RUN useradd -m -u 1000 kedro_user && \
    chown -R kedro_user:kedro_user /app

USER kedro_user

# Comando por defecto: mostrar ayuda de Kedro
CMD ["kedro", "--help"]


