from pyspark.sql import SparkSession
from pyspark.sql import Row

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.streaming import StreamingContext
import time
import sys
from pyspark.ml.regression import LinearRegression


spark = SparkSession \
        .builder \
        .appName("CorrelationExample") \
        .getOrCreate()

sc=spark.sparkContext
ssc=StreamingContext(sc, 10)
ssc.checkpoint("checkpoint")
lines = ssc.socketTextStream("localhost", 9999)


def getSparkSessionInstance(sparkConf):
    if ("sparkSessionSingletonInstance" not in globals()):
        globals()["sparkSessionSingletonInstance"] = SparkSession \
            .builder \
            .config(conf=sparkConf) \
            .getOrCreate()
    return globals()["sparkSessionSingletonInstance"]

def getSampleCounter(sparkContext):
    if ('SampleCounter' not in globals()):
        globals()['SampleCounter'] = sparkContext.accumulator(0)
    return globals()['SampleCounter']



#---------------------------------------------streaming calculation------------------------------------------------------------------
def process(time,rdd):
    try:
        numPurchases = rdd.count()
        uniqueUsers=rdd.map(lambda x:x[0]).distinct().count()
        totalRevenue=rdd.map(lambda x:int(x[2])).sum()
        #sort by value
        productsByPopularity=rdd.map(lambda x:(x[1],1)).reduceByKey(lambda x,y:x+y).sortBy(lambda x:x[1],ascending=False).collect()
        print("========= %s =========")
        print("========= %s =========")
        print("========= %s =========" % str(time))
        print("Total purchases: " , numPurchases)
        print("Unique users: " , uniqueUsers)
        print("Total revenue: " ,totalRevenue)
        print(productsByPopularity)
        print("========= %s =========")
        print("========= %s =========")
        print("========= %s =========")
    except:
        pass

events=lines.map(lambda x: (x.split(",")[0],x.split(",")[1],float(x.split(",")[2])))
#events.foreachRDD(process) #enable this to see the result
#events.foreachRDD(lambda rdd: rdd.foreachPartition(process))#no output and why ??
#---------------------------------------------streaming analysis------------------------------------------------------------------
#streaming analysis,calculate num products and total money spent by each user
users=events.map(lambda x:(x[0],(x[1],x[2])))#(user,(product,price))

#careful the input is a turple with form (user,(product,price))
#so prices has form of (product,price) and currentTotal is set to  (num_prodcut,total_revenue)
def updateFunction(prices, currentTotal):
    currentRevenue=np.sum(list(map(lambda x:x[1],prices)))
    currentNumberPurchases=len(prices)
    state=currentTotal or (0,0.0)
    if len(prices)==0:
        return state
    return (currentNumberPurchases+state[0],currentRevenue+state[1])

revenuePerUser=users.updateStateByKey(updateFunction)#updateFunction(currentkeyvalue,totalvalue)
#revenuePerUser.pprint() #enable this to see the result
#---------------------------------------------streaming regression------------------------------------------------------------------

labeledStream=lines.map(lambda x : (float(x.split("\t")[0]),Vectors.dense([float(f) for f in x.split("\t")[1].split(",")])))
#y=labeledStream.map(lambda x:x[0])
#x=labeledStream.map(lambda x:Vectors.dense([float(f) for f in x[1].split(",")]))



def regression(time,rdd):
    try:
        spark = getSparkSessionInstance(rdd.context.getConf())
        SampleCounter=getSampleCounter(rdd.context)
        rowRdd = rdd.map(lambda x: Row(features=x[1],label=x[0]))
        training=spark.createDataFrame(rowRdd)
        # x=rdd.map(lambda x:Vectors.dense([float(f) for f in x[1].split(",")]))
        # y=rdd.map(lambda x:x[0])
        lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
        lrModel = lr.fit(training)
        predictions = lrModel.transform(training)
        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        SampleCounter.add(training.count())
        print("========= %s =========")
        print("========= %s =========")
        print("========= %s =========" % str(time))
        #RMSE wont change a lot as data generated is random, plz change y and x in  gen_data.py if you wnat to see a more consistent result
        print("Samples=%d , Root Mean Squared Error (RMSE) on training data = %g" % (SampleCounter.value,rmse))
        f=open("result.log",'a')
        f.write("Samples=%d , Root Mean Squared Error (RMSE) on training data = %g " % (SampleCounter.value,rmse) +str(lr)+"\n")
        print("========= %s =========")
        print("========= %s =========")
        print("========= %s =========")
        if SampleCounter.value==100000:
            model_path = "./"
            lrModel.save(model_path )
    except:
        print("***********************NO RDD***************************")


labeledStream.foreachRDD(regression)
#lines.pprint()
ssc.start()             # Start the computation
ssc.awaitTermination()

