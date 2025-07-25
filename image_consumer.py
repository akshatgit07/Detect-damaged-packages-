from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType

spark = SparkSession.builder \
    .appName("KafkaTestConsumer") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()

schema = StructType().add("path", StringType()).add("label", StringType())

df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "test_2") \
    .load()

json_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# For testing: print output to console
query = json_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .start()

query.awaitTermination()

