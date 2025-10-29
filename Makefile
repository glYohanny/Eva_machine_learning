# ============================================================================
# Makefile para League of Legends ML Project
# Para usar en Windows con GNU Make o Make.exe
# ============================================================================

.PHONY: help build up down restart logs clean test lint format

# Variables
PROJECT_NAME = league-kedro-ml
COMPOSE_FILE = docker-compose.yml

help: ## Mostrar esta ayuda
	@echo "============================================================================"
	@echo "League of Legends ML Project - Comandos Disponibles"
	@echo "============================================================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

setup: ## Configurar proyecto (primera vez)
	powershell -ExecutionPolicy Bypass -File setup_airflow_windows.ps1

build: ## Construir imágenes Docker
	docker-compose build

up: ## Iniciar todos los servicios
	docker-compose up -d
	@echo "Servicios iniciados. Airflow UI: http://localhost:8080"

down: ## Detener todos los servicios
	docker-compose down

restart: ## Reiniciar todos los servicios
	docker-compose restart

logs: ## Ver logs de todos los servicios
	docker-compose logs -f

ps: ## Ver estado de contenedores
	docker-compose ps

# Pipelines de Kedro
run: ## Ejecutar pipeline completo de Kedro
	docker-compose run --rm kedro-app kedro run

run-eda: ## Ejecutar pipeline de EDA
	docker-compose run --rm kedro-app kedro run --pipeline eda

run-clean: ## Ejecutar pipeline de limpieza
	docker-compose run --rm kedro-app kedro run --pipeline data_cleaning

run-explore: ## Ejecutar pipeline de exploración
	docker-compose run --rm kedro-app kedro run --pipeline data_exploration

run-process: ## Ejecutar pipeline de procesamiento
	docker-compose run --rm kedro-app kedro run --pipeline data_processing

run-train: ## Ejecutar pipeline de entrenamiento
	docker-compose run --rm kedro-app kedro run --pipeline data_science

run-eval: ## Ejecutar pipeline de evaluación
	docker-compose run --rm kedro-app kedro run --pipeline evaluation

# Utilidades
shell: ## Abrir shell en contenedor de Kedro
	docker-compose run --rm kedro-app bash

airflow-shell: ## Abrir shell en contenedor de Airflow
	docker-compose exec airflow-webserver bash

clean: ## Limpiar contenedores y volúmenes
	docker-compose down -v
	docker system prune -f

clean-all: ## Limpieza completa (¡cuidado!)
	docker-compose down -v
	docker system prune -a -f --volumes

test: ## Ejecutar tests
	docker-compose run --rm kedro-app pytest

lint: ## Ejecutar linters
	docker-compose run --rm kedro-app flake8 src/

format: ## Formatear código con black
	docker-compose run --rm kedro-app black src/

catalog: ## Ver catálogo de datos
	docker-compose run --rm kedro-app kedro catalog list

pipeline-list: ## Listar pipelines disponibles
	docker-compose run --rm kedro-app kedro pipeline list

# Logs específicos
logs-airflow: ## Ver logs de Airflow webserver
	docker-compose logs -f airflow-webserver

logs-scheduler: ## Ver logs de Airflow scheduler
	docker-compose logs -f airflow-scheduler

logs-kedro: ## Ver logs de Kedro
	docker-compose logs -f kedro-app

# Info
version: ## Mostrar versiones
	@echo "Docker:"
	@docker --version
	@echo "Docker Compose:"
	@docker-compose --version
	@echo ""
	@echo "Kedro:"
	@docker-compose run --rm kedro-app kedro --version


