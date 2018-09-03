from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer,OneHotEncoder
from pyspark.ml.feature import SQLTransformer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType,StringType
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import FloatType
from pyspark.ml.clustering import KMeans,BisectingKMeans
from pyspark.ml.linalg import Vectors, VectorUDT

spark = SparkSession \
        .builder \
        .appName("CorrelationExample") \
        .getOrCreate()
sc=spark.sparkContext


category=spark.read.text("../ml-100k/u.genre").rdd
movies=spark.read.text("../ml-100k/u.item").rdd
rating = spark.read.text("../ml-100k/u.data").rdd

#filter the last empty line
category=category.filter(lambda x : x.value !="")
genreMap=category.map(lambda x: x.value.split("|")).map(lambda x:(x[1],x[0])).collectAsMap()
#transform binary into corresponding movie categories
#[1, 'Toy Story (1995)', ['Animation', "Children's", 'Comedy']]
titlesAndGenres=movies.map(lambda x:x.value.split("|")).map(lambda x:[(int(x[0])),x[1],[genreMap[str(idx)] for idx in np.where(np.array(x[5:])=="1")[0]]])#(array([0, 2], dtype=int64),)[0]
rating=rating.map(lambda x:x.value.split("\t")[:3]).map(lambda x: Row(userId=int(x[0]),movieId=int(x[1]),rating=float(x[2])))
rate_frame=spark.createDataFrame(rating)


als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(rate_frame)

#check the distribution to see if normalization is needed

mean=udf(lambda x: float(np.mean(x)),FloatType())
array2Vec=udf(lambda x :Vectors.dense(x), VectorUDT())
userFactors=model.userFactors.withColumn("features",array2Vec("features"))#transform array to Vector in pyspark DataFrame
itemFactors=model.itemFactors.withColumn("features",array2Vec("features"))
#looks like there is no need to apply normalization
userFactors=userFactors.withColumn("f_mean",mean("features"))
itemFactors=itemFactors.withColumn("f_mean",mean("features"))
# +-------+--------------------+
# |summary|              f_mean|
# +-------+--------------------+
# |  count|                 943|
# |   mean|-0.20458285266955917|
# | stddev| 0.16544009872953944|
# |    min|         -0.85233325|
# |    max|           0.6468454|
# +-------+--------------------+
userFactors.describe(["f_mean"]).show()
# +-------+-------------------+
# |summary|             f_mean|
# +-------+-------------------+
# |  count|               1682|
# |   mean|-0.1940755807113511|
# | stddev|0.22787624832705886|
# |    min|         -1.0772276|
# |    max|          0.6794037|
# +-------+-------------------+
itemFactors.describe(["f_mean"]).show()

#-------------------------clustering------------------------------
kmeans = KMeans().setK(5).setSeed(1)
model = kmeans.fit(itemFactors)
centers = model.clusterCenters()
wssse = model.computeCost(itemFactors)
print("Within Set Sum of Squared Errors = " + str(wssse))#2161.7
# +---+--------------------+------------+----------+
# | id|            features|      f_mean|prediction|
# +---+--------------------+------------+----------+
# | 10|[-1.1613303422927...| -0.49788338|         1|
# | 20|[-1.0665705204010...| -0.39318347|         1|
# | 30|[-1.9666320085525...| -0.51597494|         1|
# | 40|[-0.2027567178010...| 0.056338817|         2|
# | 50|[-0.3597640395164...| -0.25277916|         1|
# | 60|[-1.1981514692306...| -0.56160897|         1|
# | 70|[-0.7702391147613...| -0.43908575|         3|
# .....
itemFactors=model.transform(itemFactors)


bkm = BisectingKMeans().setK(5).setSeed(1)
model = bkm.fit(itemFactors)
cost = model.computeCost(itemFactors)
print("Within Set Sum of Squared Errors = " + str(cost))#2181.4

centers=[[i,center] for i,center in enumerate(centers)]
centers = sc.parallelize(centers)
centers=centers.map(lambda x:Row(prediction=x[0],center=Vectors.dense(x[1])))
centers=spark.createDataFrame(centers)


#-------------------------explain-------------------------------
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
itemFactors_center=itemFactors.join(centers,["prediction"])
#take top 20 from each cluster with their title and category
dist=udf(lambda x,y:float(np.sum(np.power(x-y,2))),FloatType())

# +----------+---+--------------------+-----------+--------------------+----------+
# |prediction| id|            features|     f_mean|              center|      dist|
# +----------+---+--------------------+-----------+--------------------+----------+
# |         0| 10|[-0.8583860993385...|-0.25816193|[-0.5481244971372...| 0.5258225|
# |         0| 60|[-0.4993891119956...|-0.30285546|[-0.5481244971372...|0.31626022|
# |         0| 70|[-0.6186761856079...|-0.26710632|[-0.5481244971372...|0.21415813|
# |         0| 90|[-0.5056012868881...|-0.25839493|[-0.5481244971372...|0.70033497|
# |         0|150|[-1.1845953464508...|-0.15942442|[-0.5481244971372...| 1.5181916|
# |         0|160|[-0.9759202003479...|-0.50211096|[-0.5481244971372...| 1.3931999|
# |  ......
itemFactors_center=itemFactors_center.withColumn("dist",dist("features","center"))

# +----------+---+--------------------+------------+--------------------+---------+
# |prediction| id|            features|      f_mean|              center|     dist|
# +----------+---+--------------------+------------+--------------------+---------+
# |         4|127|[-0.6819288730621...|  0.32100606|[-0.6124414238147...|13.272518|
# |         4| 93|[0.91705816984176...|   0.6468454|[-0.6124414238147...|12.958792|
# |         4| 50|[-1.7279316186904...|-0.034507066|[-0.6124414238147...|10.938511|
# |         4|917|[-1.8949651718139...| 0.047153354|[-0.6124414238147...| 9.118178|
# |         4|434|[-0.8833367228507...| -0.15495218|[-0.6124414238147...| 9.084182|
# |         4|202|[-0.1081635206937...|-0.052518647|[-0.6124414238147...| 8.707626|
# |         4|604|[0.36510351300239...|  0.45913818|[-0.6124414238147...| 8.687587|
itemFactors_center.sort("prediction","dist",ascending=False).show()#sort according to prediction then dist

# +----------+-----+
# |prediction|count|
# +----------+-----+
# |         1|  182|
# |         3|   90|
# |         4|  197|
# |         2|  122|
# |         0|  352|
# +----------+-----+
itemFactors_center.groupBy("prediction").count().show()
titlesAndGenres_f=titlesAndGenres.map(lambda x:Row(id=x[0],title=x[1],category=x[2]))
titlesAndGenres_f=spark.createDataFrame(titlesAndGenres_f)

#DataFrame[id: int, prediction: int, features: vector, f_mean: float, center: vector, dist: float, category: array<string>, title: string]
all=itemFactors_center.join(titlesAndGenres_f,["id"])

#use window functions to rank dist in each cluster,here we only show top 5 for convenience
# +---+----------+--------------------+-----------+--------------------+----------+--------------------+--------------------+----+
# | id|prediction|            features|     f_mean|              center|      dist|            category|               title|rank|
# +---+----------+--------------------+-----------+--------------------+----------+--------------------+--------------------+----+
# |207|         1|[-0.3477138280868...|-0.16462141|[-0.1126436584444...|0.26383346|[Action, Drama, R...|Cyrano de Bergera...|   1|
# | 95|         1|[0.02467109821736...|-0.23603973|[-0.1126436584444...|0.29004848|[Animation, Child...|      Aladdin (1992)|   2|
# |178|         1|[-0.1564631313085...|-0.12570915|[-0.1126436584444...|0.29872712|             [Drama]| 12 Angry Men (1957)|   3|
# | 87|         1|[-0.3330552279949...|-0.11615199|[-0.1126436584444...| 0.3192487|             [Drama]|Searching for Bob...|   4|
# |144|         1|[-0.4688196182250...| -0.1482518|[-0.1126436584444...|0.36593416|  [Action, Thriller]|     Die Hard (1988)|   5|
# |504|         3|[-0.5722782611846...|-0.21572395|[-0.6785494033247...|0.55665404|      [Crime, Drama]|Bonnie and Clyde ...|   1|
# |530|         3|[-0.7186721563339...|-0.16036995|[-0.6785494033247...| 0.5603168|         [Adventure]|Man Who Would Be ...|   2|
# |835|         3|[-1.0369261503219...|-0.33052975|[-0.6785494033247...|0.56044173|[Comedy, Musical,...|Gay Divorcee, The...|   3|
# |376|         3|[-0.8201395869255...|-0.37102145|[-0.6785494033247...| 0.5718177|            [Comedy]|   Houseguest (1994)|   4|
# |585|         3|[-0.4923697710037...|  -0.300104|[-0.6785494033247...| 0.5745846|            [Comedy]|   Son in Law (1993)|   5|
# |552|         4|[-0.5532200932502...|-0.07915019|[-0.6124414238147...|0.21844864|            [Sci-Fi]|      Species (1995)|   1|
# |436|         4|[-0.8437770009040...|-0.21150774|[-0.6124414238147...| 0.4613527|            [Horror]|American Werewolf...|   2|
# |399|         4|[-0.3762356042861...|-0.09822665|[-0.6124414238147...|   0.49395|[Action, Adventur...|Three Musketeers,...|   3|
# |646|         4|[-0.4437087178230...|-0.03998735|[-0.6124414238147...| 0.5035491|           [Western]|Once Upon a Time ...|   4|
# |624|         4|[-0.8322196602821...|-0.16311903|[-0.6124414238147...| 0.5183173|[Animation, Child...|Three Caballeros,...|   5|
# |313|         2|[-0.4044930040836...|-0.27094376|[-0.5422254357817...| 0.4365328|[Action, Drama, R...|      Titanic (1997)|   1|
# |  7|         2|[-0.5057435035705...|-0.23938975|[-0.5422254357817...|0.47840604|     [Drama, Sci-Fi]|Twelve Monkeys (1...|   2|
# | 26|         2|[-0.7005702853202...|-0.26516035|[-0.5422254357817...|0.49615732|            [Comedy]|Brothers McMullen...|   3|
# |616|         2|[-0.6528839468955...| -0.2530402|[-0.5422254357817...|0.54316664|    [Horror, Sci-Fi]|Night of the Livi...|   4|
# |860|         2|[-0.5837879180908...|-0.21495089|[-0.5422254357817...|0.57241887|  [Horror, Thriller]|Believers, The (1...|   5|
# +---+----------+--------------------+-----------+--------------------+----------+--------------------+--------------------+----+
window = Window.partitionBy(all['prediction']).orderBy(all['dist'].asc())
ranked_all=all.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 5).show()

#to get a more clear view of top 10 in each cluster
ranked_all.filter(col("prediction")==0).select("prediction","category","title").show(truncate=False)#most are drama and comedy
ranked_all.filter(col("prediction")==1).select("prediction","category","title").show(truncate=False)
ranked_all.filter(col("prediction")==2).select("prediction","category","title").show(truncate=False)
ranked_all.filter(col("prediction")==3).select("prediction","category","title").show(truncate=False)
