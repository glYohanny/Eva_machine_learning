"""
Pipeline de Limpieza de Datos (Data Cleaning)

Este pipeline se encarga de limpiar y preparar los datos raw para análisis.
Sigue la metodología CRISP-DM en la fase de Preparación de Datos.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

