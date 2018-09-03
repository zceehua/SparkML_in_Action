from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType,DateType
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

spark = SparkSession \
        .builder \
        .appName("CorrelationExample") \
        .getOrCreate()

#load data in csv format with header
rawData = spark.read.load("D:\PycharmWorkSpace\My_Project\pyspark\pyspark_in_action\\03_regression\\hour.csv",format="csv",header=True)
rawData.count()#17379
data=rawData
#casual+registered=cnt
rawData=rawData.drop("casual","registered")#注意如何drop列
rawData=rawData.withColumnRenamed("cnt","label")#注意如何reanme列
cat_features=rawData.columns[2:10]

for col in cat_features:
    #must give a new column name
    indexer = StringIndexer(inputCol=col, outputCol=col+"_indexed",handleInvalid='error')
    indexed = indexer.fit(rawData).transform(rawData)
    encoder = OneHotEncoder(inputCol=col+"_indexed", outputCol=col+"Vec")
    rawData = encoder.transform(indexed)


#cast columns to float
for col in rawData.columns[2:15]:
    rawData=rawData.withColumn(col,rawData[col].cast(FloatType()))

#convert date to date format and extract week day
from pyspark.sql.functions import date_format
rawData=rawData.withColumn("dteday",rawData["dteday"].cast(DateType()))
rawData=rawData.withColumn('dteday', date_format('dteday', 'u'))

isweekend=udf(lambda x:1.0 if int(x) > 5 else 0.0,FloatType())
rawData=rawData.withColumn("isWeekend",isweekend("dteday"))#whether it is weekend
rawData=rawData.drop("dteday")


#feature and label for liner regression
cols=[col+'Vec' for col in rawData.columns[2:10]]
cols+=[col for col in rawData.columns[10:14]+['label']]
linear=rawData.select(cols)
assembler = VectorAssembler(
    inputCols=[col for col in linear.columns[:-1]],
    outputCol="features")
linear = assembler.transform(linear)
#feature and label for decision tree
dtree=rawData.select([col for col in rawData.columns[2:15]])
assembler = VectorAssembler(
    inputCols=[col for col in dtree.columns[:-1]],
    outputCol="features")
dtree = assembler.transform(dtree)
#----------------------regression model------------------------
from pyspark.ml.regression import LinearRegression,DecisionTreeRegressor,RandomForestRegressor,GBTRegressor

#see improve data section below
# def log_t(x):
#     return float(np.log(x))
#
# log=udf(log_t,FloatType())
# linear=linear.withColumn("label",log("label"))

train,test=linear.randomSplit([0.8, 0.2],seed=42)
#rmse increases to  142.322 if we don't use one-hot features
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(train)
predictions = lrModel.transform(test)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

# def exp_t(x):
#     return float(np.exp(x))
#
# exp=udf(exp_t,FloatType())
# predictions=predictions.withColumn("label",exp("label"))
# predictions=predictions.withColumn("prediction",exp("prediction"))

rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)#with one-hot: 101.355


train,test=dtree.randomSplit([0.8, 0.2],seed=42)

models=[DecisionTreeRegressor(),RandomForestRegressor(),GBTRegressor()]

#Without one-hot on category features:
# DecisionTreeRegressor (RMSE) on test data = 107.854
# RandomForestRegressor (RMSE) on test data = 115.121
# GBTRegressor (RMSE) on test data = 60.633
#with one-hot :
# DecisionTreeRegressor (RMSE) on test data = 128.919
# RandomForestRegressor (RMSE) on test data = 120.957
# GBTRegressor (RMSE) on test data = 84.7877
#for tree models, often we dont use one hot, this would usually degrade their performance
for reg in models:
    model=reg.fit(train)
    predictions=model.transform(test)
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("%s (RMSE) on test data = %g" % (str(reg).split("_")[0],rmse))

def squared_log_error(pred, actual):
    return (np.log(pred + 1) - np.log(actual + 1))**2

#--------------------improve model-------------------------------
#many ML models will some kind of assume data distributions, for example, liner models will assume data is normal distribution
# however here, target is not normal distribution
target=list(map(lambda x:float(x.cnt),data.select("cnt").rdd.collect()))
plt.hist(target,bins=40,color='lightblue', normed=True)
plt.show()

#applying the transformation below could improve the performance of liner models
#so we can apply log transform to target to approach normal distribution
log_targets=list(map(lambda x:np.log(float(x.cnt)),data.select("cnt").rdd.collect()))
plt.hist(log_targets,bins=40,color='lightblue', normed=True)
plt.show()

#when target is non-negative and the range is large, we can apply square root
sqrt_targets = list(map(lambda x:np.sqrt(float(x.cnt)),data.select("cnt").rdd.collect()))
plt.hist(sqrt_targets, bins=40, color='lightblue', normed=True)
plt.show()
