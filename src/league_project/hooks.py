"""
Hooks del proyecto League of Legends ML.

Nota: PySpark no se utiliza en este proyecto, por lo que los hooks de Spark
están comentados. Si en el futuro se necesita Spark, descomentar y agregar
pyspark a requirements.txt.
"""

from kedro.framework.hooks import hook_impl

# PySpark no se usa en este proyecto
# from pyspark import SparkConf
# from pyspark.sql import SparkSession


# class SparkHooks:
#     @hook_impl
#     def after_context_created(self, context) -> None:
#         """Initialises a SparkSession using the config
#         defined in project's conf folder.
#         """
#
#         # Load the spark configuration in spark.yaml using the config loader
#         parameters = context.config_loader["spark"]
#         spark_conf = SparkConf().setAll(parameters.items())
#
#         # Initialise the spark session
#         spark_session_conf = (
#             SparkSession.builder.appName(context.project_path.name)
#             .enableHiveSupport()
#             .config(conf=spark_conf)
#         )
#         _spark_session = spark_session_conf.getOrCreate()
#         _spark_session.sparkContext.setLogLevel("WARN")
