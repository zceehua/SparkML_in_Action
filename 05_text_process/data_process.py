from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import Tokenizer, RegexTokenizer,StopWordsRemover,StringIndexer,OneHotEncoder,Word2Vec
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType,StringType,ArrayType
import numpy as np

spark = SparkSession \
        .builder \
        .appName("CorrelationExample") \
        .getOrCreate()
sc=spark.sparkContext

path="../20news-bydate-train2/*"
rdd = sc.wholeTextFiles(path)#rdd contains (path,text)
rdd.first()#('file:/D:/PycharmWorkSpace/My_Project/pyspark/pyspark_in_action/20news-bydate-train2/comp.os.ms-windows.misc/10000', 'From: yeoy@a.cs.okstate.edu (YEO YEK CHONG)\nSubject: Re: Is "Kermit" available for Windows 3.0/3.1?\nOrganization: Oklahoma State University\nLines: 7\n\nFrom article <a4Fm3B1w165w@vicuna.ocunix.on.ca>, by Steve Frampton <frampton@vicuna.ocunix.on.ca>:\n> I was wondering, is the "Kermit" package (the actual package, not a\n\nYes!  In the usual ftp sites.\n\nYek CHong\n\n')
rdd.map(lambda x:x[1]).count()#52+78+90=220

newsgroups=rdd.map(lambda x: x[0].split("/")[-2]).map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y).map(lambda x:(x[1],x[0])).sortByKey().collect()
text=rdd.map(lambda x:x[1])
text_frame=text.map(lambda x : Row(sentence=x))
text_frame=spark.createDataFrame(text_frame)

countTokens = udf(lambda words: len(words), IntegerType())
regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern=r"(?u)\b\w\w+\b",gaps=False)
regexTokenized = regexTokenizer.transform(text_frame)

regexTokenized=regexTokenized.select("sentence", "words").withColumn("tokens", countTokens(col("words")))

# +--------------------+--------------------+------+--------------+
# |               words|            filtered|tokens|filteredtokens|
# +--------------------+--------------------+------+--------------+
# |[from, yeoy, cs, ...|[yeoy, cs, okstat...|    53|            39|
# |[subject, win, st...|[subject, win, st...|    81|            64|
# |[subject, re, win...|[subject, re, win...|   126|            73|
# |[from, phoenix, p...|[phoenix, princet...|   115|            78|
# |[from, holmes7000...|[holmes7000, iscs...|    49|            32|
# |[subject, is, sma...|[subject, smartdr...|   434|           256|
# |[from, kmembry, v...|[kmembry, viamar,...|   140|            91|
# |[organization, un...|[organization, un...|   130|            80|
# |[subject, help, w...|[subject, help, c...|   169|           111|
# |[from, farley, ac...|[farley, access, ...|    98|            75|
# |[from, cm, cavsys...|[cm, cavsys, demo...|    99|            71|
# .......
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
removed=remover.transform(regexTokenized)
removed=removed.select("words", "filtered","tokens").withColumn("filteredtokens", countTokens(col("filtered")))
#removed.select("filteredtokens").rdd.map(lambda x:x.filteredtokens).reduce(lambda x,y:x+y)#33953
frequency=removed.select("filtered").rdd.flatMap(lambda x:x.filtered).map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y).collect()
countdistinct=removed.select("filtered").rdd.flatMap(lambda x:x.filtered).distinct().count()#7590
#filter top 20 words with lowest frequency
lf_words=list(map(lambda x:x[0],sorted(frequency,key=lambda x:x[1])[:20]))

filter_low=udf(lambda x: [w for w in x if w not in lf_words],ArrayType(StringType()))#注意如何返回string类型的数组
#[organization, un...|[organization, un...|   130|            79|   where 80--->79
removed=removed.withColumn("filtered",filter_low("filtered")).withColumn("filteredtokens", countTokens(col("filtered"))).show()

#stem the words
stemmer=PorterStemmer()
stem=udf(lambda x: [stemmer.stem(y) for y in x], ArrayType(StringType()))
stemmed=removed.withColumn("filtered", stem("filtered"))
#The number of distinct words get fewer
countdistinct=stemmed.select("filtered").rdd.flatMap(lambda x:x.filtered).distinct().count()#6062

#word similarity using word2vec
from pyspark.sql.functions import format_number as fmt
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="filtered", outputCol="result")
model = word2Vec.fit(stemmed)
word_vec=model.getVectors()
#because the corpus here is very small, much better result could be seen if the whole 20 news group data set is loaded
model.findSynonyms("robot", 10).select("word", fmt("similarity", 5).alias("similarity")).show()
