from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType,StringType,DoubleType
import re

spark = SparkSession \
        .builder \
        .appName("CorrelationExample") \
        .getOrCreate()

#---------------------data preprocessing-----------------------------
sc=spark.sparkContext
#spark.read.load results in Row format
rawData = spark.read.load("./train.tsv",format="text",header=False).rdd
#rawData = sc.textFile("./train.tsv")
#remove header
header=rawData.first()
rawData=rawData.filter(lambda x : x != header)
records =rawData.map(lambda x : x.value.split("\t"))

#transfer rdd to dataframe with specified column names
records=records.toDF(list(map(lambda x:re.sub("\"","",x),header.value.split())))


def preprocess(x):
        x = re.sub(r"\"", "", x)
        if "?" in x:
                x=0.0
        else:
                x=float(x)
        if x<0:
                x=0.0
        return x

clean=udf(preprocess,DoubleType())
# records.withColumn(records.columns[4],clean(records.columns[4])).show()
for col_ in records.columns[4:]:
        records=records.withColumn(col_,clean(col_))

#string -> int
records=records.withColumn("label",records["label"].cast(IntegerType()))

records.cache
records.count()
#-----------------------------------------feature engineering-------------------------------------
from pyspark.ml.classification import LogisticRegression,LinearSVC,NaiveBayes,RandomForestClassifier
from pyspark.ml.feature import VectorAssembler,StandardScaler,StringIndexer,OneHotEncoder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Imputer
# +--------------------+
# |    alchemy_category|
# +--------------------+
# |         "law_crime"|
# |"arts_entertainment"|
# |        "recreation"|
# |            "gaming"|
# |          "religion"|
# |  "culture_politics"|
# |          "business"|
# |           "weather"|
# |"science_technology"|
# |            "health"|
# |           "unknown"|
# | "computer_internet"|
# |                 "?"|
# |            "sports"|
# +--------------------+
records.select("alchemy_category").distinct().show()
indexer = StringIndexer(inputCol="alchemy_category", outputCol="categoryIndex",handleInvalid='error')
indexed = indexer.fit(records).transform(records)

encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
records = encoder.transform(indexed)

feature_cols=records.columns[4:-3]
#fill in missing value
imputer = Imputer(inputCols=[col for col in feature_cols[:7]], outputCols=[col for col in feature_cols[:7]],missingValue=0)
records2=records
records2=imputer.fit(records2).transform(records2)
#assemble feature vectors
feature_cols.extend(["categoryVec"])
assembler = VectorAssembler(
    inputCols=[col for col in feature_cols],
    outputCol="features_raw")
data = assembler.transform(records2)

feature_cols.extend(["label","features_raw"])
cols=feature_cols
data=data.select([col for col in cols])
scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                        withStd=True, withMean=True)

data = scaler.fit(data).transform(data)
(training, test) = data.randomSplit([0.8, 0.2])

#----------------------------classification--------------------------------
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.2)
lrModel = lr.fit(training)

predictions = lrModel.transform(test)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
#accuracy = evaluator.evaluate(predictions,{evaluator.metricName: "accuracy"})
print("Accuracy = %g " % (accuracy))#0.52 drop to 0.483 if imputed ,then increase to 0.58 if scaled, to  0.62 if category is used

#耗时
lsvc = LinearSVC(maxIter=10, regParam=0.1)
lsvcModel = lsvc.fit(training)
predictions = lsvcModel.transform(test)
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % (accuracy))#0.607 increase to 0.619 if imputed, then to 0.631854 if scaled, to 0.65482 if category is used


nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = nb.fit(training)
predictions = model.transform(test)
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % (accuracy))#0.582 increase to  0.591781 if imputed

#we usually do not scale(StandardScaler) input vectors we using tree models
#as decision depends on probability rather than feature value, scaling may cause information loss
rf = RandomForestClassifier( numTrees=150,subsamplingRate=0.8,maxDepth=10)
model = rf.fit(training)
predictions = model.transform(test)
accuracy = evaluator.evaluate(predictions)
#accuracy = evaluator.evaluate(predictions,{evaluator.metricName: "weightedRecall"})
print("Accuracy = %g " % (accuracy))#0.68643 increase to 0.690411 if imputed, then to 0.685 if scaled, to 0.701743 if category is used


