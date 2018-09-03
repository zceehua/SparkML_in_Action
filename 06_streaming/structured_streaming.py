from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split

spark = SparkSession \
    .builder \
    .appName("StructuredNetworkWordCount") \
    .getOrCreate()



lines = spark \
    .readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Split the lines into words
#words = lines.select(explode(split(lines.value,"")).alias("value"))

# Generate running word count
wordCounts = lines.groupBy("value").count()

split_col = split(wordCounts['value'], ',')
wordCounts = wordCounts.withColumn('NAME1', split_col.getItem(0))
wordCounts = wordCounts.withColumn('NAME2', split_col.getItem(1))
wordCounts = wordCounts.withColumn('NAME3', split_col.getItem(2))

query = wordCounts \
    .writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

#query.awaitTermination()