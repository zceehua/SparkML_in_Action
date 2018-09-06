
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer,OneHotEncoder
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType,StringType
import numpy as np
import matplotlib.pyplot as plt


spark = SparkSession \
        .builder \
        .appName("whatever") \
        .getOrCreate()

#---------------------User Data------------------------------
sc=spark.sparkContext
#spark.read.text returns dataframe，while sc.textFile returns RDD
user_data = sc.textFile("../ml-100k/u.user")#u.user MapPartitionsRDD[12]
user_data.first()#'1|24|M|technician|85711'

# user_data = spark.read.text("/ml-100k/u.user")#DataFrame[value: string]
# user_data.first()#Row(value='1|24|M|technician|85711')

#dataframe can not use map() directly, use dataframe.rdd to get RDD
#user_data=user_data.rdd
#becareful we have to use x.value rather than x directly if using dataframe.rdd
#user_fields=user_data.map(lambda x : x.value.split("|"))

user_fields=user_data.map(lambda x : x.split("|"))
num_users=user_fields.map(lambda x:x[0]).count()
num_genders=user_fields.map(lambda x : x[2]).distinct()#['M', 'F']
num_occupations = user_fields.map(lambda x:x[3]).distinct().count()
num_zipcodes = user_fields.map(lambda x:x[4]).distinct().count()
print ("Users: %d, genders: %d, occupations: %d, ZIP codes: %d" % (num_users, num_genders,num_occupations, num_zipcodes))

# [24, 53, 23, 24, 33, 42, 57, 36, 29, 53, 39, 28, 47, 45, 49, 21, 30, 35, 40, 42, 26, 25, 30, 21,
#  39, 49, 40, 32, 41, 7, 24, 28, 23, 38, 20, 19, 23, 28, 41, 38, 33, 30, 29, 26, 29, 27, 53, 45, 23, 21, 28, 18, 26, 22, 37, 25, 16, 27, 49, 50, 36, 27, 31, 32, 51, 23, 17, 19, 24, 27, 39, 48, 24, 39, 24, 20, 30, 26, 39, 34, 21, 50, 40, 32, 51, 26, 47, 49, 43, 60, 55, 32, 48, 26, 31, 25, 43, 49, 20, 36, 15, 38, 26, 27, 24, 61, 39, 44, 29, 19, 57, 30, 47, 27, 31, 40, 20, 21, 32, 47, 54, 32, 48, 34, 30, 28, 33, 24, 36, 20, 59, 24, 53, 31, 23, 51,
ages = user_fields.map(lambda x: int(x[1])).collect()
#x:range of ages, y:counts of ages in different bin
plt.hist(ages, bins=20, color='lightblue', normed=True)
plt.xlabel("ages")
plt.show()

#career distribution
#[('engineer', 67), ('homemaker', 7), ('doctor', 7), ('administrator', 79), ('student', 196), ('programmer', 66), ('other', 105), ('executive', 32), ('retired', 14), ('artist', 28), ('none', 9), ('educator', 95), ('scientist', 31), ('lawyer', 12), ('writer', 45), ('technician', 27), ('librarian', 51), ('salesman', 12), ('healthcare', 16), ('marketing', 26), ('entertainment', 18)]
count_by_occupation = user_fields.map(lambda x: (x[3], 1)).reduceByKey(lambda x,y:x+y)
#we can use countByValue instead of reduceByKey
#count_by_occupation2 = user_fields.map(lambda fields: fields[3]).countByValue()
x_axis1 = np.array([c[0] for c in count_by_occupation])
y_axis1 = np.array([c[1] for c in count_by_occupation])
x_axis = x_axis1[np.argsort(y_axis1)]#name
y_axis = y_axis1[np.argsort(y_axis1)]#number
pos = np.arange(len(x_axis))
width=1
plt.bar(pos+0.5, y_axis, width, color='lightblue')
plt.xticks(pos,x_axis,rotation=30)
plt.show()

#------------------------------movie data--------------------------------
movie_data = sc.textFile("../ml-100k/u.item")
#1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
movie_data.first()
num_movies = movie_data.count()#1682
titles2=movie_data.map(lambda x: x.split("|")[:2]).map(lambda x: (int(x[0]),x[1])).collectAsMap()
titles=movie_data.map(lambda x: x.split("|")[:2]).map(lambda x: Row(itemid=int(x[0]),title=x[1]))
# +------+--------------------+
# |itemid|               title|
# +------+--------------------+
# |     1|    Toy Story (1995)|
# |     2|    GoldenEye (1995)|
# |     3|   Four Rooms (1995)|
# |     4|   Get Shorty (1995)|
# |     5|      Copycat (1995)|
# |     6|Shanghai Triad (Y...|
# |     7|Twelve Monkeys (1...|
# |     8|         Babe (1995)|
# ...
titles=spark.createDataFrame(titles)
def convert_year(x):
    try:
        return int(x[-4:])
    except:
        return 1900

movie_fields=movie_data.map(lambda x:x.split("|"))
years=movie_fields.map(lambda x: x[2]).map(lambda x: convert_year(x))
#filter out year=1900
years_filtered = years.filter(lambda x: x != 1900)
#calculate movie age
movie_ages = years_filtered.map(lambda yr: 1998-yr).countByValue()

values = list(movie_ages.values())
bins = list(movie_ages.keys())
#plt.hist(values,bins=bins,color='lightblue', normed=True)
plt.bar(range(len(bins)),values,width,color='lightblue')
plt.show()

#------------------------------rating data--------------------------------
rating_data = sc.textFile("../ml-100k/u.data")
print (rating_data.first())#196\t242\t3\t881250949
num_ratings = rating_data.count()
rating_data=rating_data.map(lambda x : x.split("\t"))
ratings=rating_data.map(lambda x:int(x[2]))
max_rating=ratings.reduce(lambda x,y:max(x,y))
min_rating=ratings.reduce(lambda x,y:min(x,y))
mean_rating=ratings.reduce(lambda x,y:x+y)/num_ratings
median_rating=np.median(ratings.collect())
ratings_per_user=num_ratings/num_users
ratings_per_movie = num_ratings / num_movies

ratings.stats()
#(count: 100000, mean: 3.52986, stdev: 1.12566797076, max: 5.0, min: 1.0)
#rating distribution
count_by_rating=dict(ratings.countByValue())
x=[c[0] for c in count_by_rating.items()]
y=[c[1] for c in count_by_rating.items()]

plt.bar(range(len(x)),y,width,color='lightblue')
plt.show()

user_ratings_grouped=rating_data.map(lambda x:(int(x[0]),int(x[2]))).groupByKey().sortByKey()
#use len to count how many films are rated by each user
user_ratings_byuser=user_ratings_grouped.map(lambda x :(x[0],len(x[1])))
user_ratings_byuser.take(5)
user_ratings_byuser_local = user_ratings_byuser.map(lambda x: x[1]).collect()
plt.hist(user_ratings_byuser_local,bins=200)
plt.show()


#------------------------------------data processing and transforming-------------------------------


mean_year=np.mean(years_filtered.collect())
median_year=np.median(years_filtered.collect())
raw_year=np.array(years.collect())
#filling null value
null_year_idx=np.where( raw_year== 1900)[0][0]#(array([266], dtype=int64),)
raw_year[null_year_idx]=mean_year


all_occupations = user_fields.map(lambda fields: fields[3]). distinct().collect().sort()

#test= spark.read.load("test.csv",format="csv",header=True)
#transfer rdd to dataframe
fields=user_fields.map(lambda x: Row(id=x[0],age=x[1],gender=x[2],occupations=x[3],postcode=x[4]))
users=spark.createDataFrame(fields)
# +---+------+---+-------------+--------+
# |age|gender| id|  occupations|postcode|
# +---+------+---+-------------+--------+
# | 24|     M|  1|   technician|   85711|
# | 53|     F|  2|        other|   94043|
# | 23|     M|  3|       writer|   32067|
# | 24|     M|  4|   technician|   43537|
# | 33|     F|  5|        other|   15213|
# | 42|     M|  6|    executive|   98101|
# | 57|     M|  7|administrator|   91344|
# | 36|     M|  8|administrator|   05201|
# | 29|     M|  9|      student|   01002|
# | 53|     M| 10|       lawyer|   90703|
# | 39|     F| 11|        other|   30329|
indexer = StringIndexer(inputCol="occupations", outputCol="occupationsIndex",handleInvalid='error')
indexed=indexer.fit(users).transform(users)
#transfer dataframe to rdd by ".rdd"
all_occupations = set(indexed.select("occupations","occupationsIndex").rdd.map(lambda x:(x[0],x[1])).collect())
encoder = OneHotEncoder(inputCol="occupationsIndex", outputCol="occupationsVec")
encoded = encoder.transform(indexed)
encoded.select("occupations","occupationsVec").show()
# +-------------+---------------+
# |  occupations| occupationsVec|
# +-------------+---------------+
# |   technician|(20,[11],[1.0])|
# |        other| (20,[1],[1.0])|
# |       writer| (20,[7],[1.0])|
# |   technician|(20,[11],[1.0])|
# |        other| (20,[1],[1.0])|
# |    executive| (20,[8],[1.0])|
# |administrator| (20,[3],[1.0])|
# |administrator| (20,[3],[1.0])|
# |      student| (20,[0],[1.0])|


def change(x):
    return x/2

#use withColumn to perform simple column transformation
users=users.withColumn("postcode",change(col("postcode")))
# +---+------+---+-------------+--------+
# |age|gender| id|  occupations|postcode|
# +---+------+---+-------------+--------+
# | 24|     M|  1|   technician| 42856.0|
# | 53|     F|  2|        other| 47022.0|
# | 23|     M|  3|       writer| 16034.0|
# | 24|     M|  4|   technician| 21769.0|
# | 33|     F|  5|        other|  7607.0|
# | 42|     M|  6|    executive| 49051.0|
# | 57|     M|  7|administrator| 45672.5|
# | 36|     M|  8|administrator|  2601.0|
# | 29|     M|  9|      student|   501.5|

rate_frame2=rating_data.map(lambda x: Row(userid=x[0],itemid=x[1],rating=x[2],timestep=int(x[3])))
rate_frame=spark.createDataFrame(rate_frame2)

def extract_datetime(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts)

#get_hour=udf(lambda x : extract_datetime(x),IntegerType())
timestamps = rating_data.map(lambda fields: np.float(fields[3]))
hour_of_day = timestamps.map(lambda ts: np.float(extract_datetime(ts).hour))
hour = hour_of_day.map(lambda x: Row(hour=int(x)))
hours= spark.createDataFrame(hour)
# timestamps = rating_data.map(lambda fields: (np.float(fields[0]),np.float(fields[3])))
# hour_of_day = timestamps.map(lambda ts: (ts[0],np.float(extract_datetime(ts[1]).hour)))
# hour = hour_of_day.map(lambda x: Row(id=x[0],hour=x[1]))
# hours= spark.createDataFrame(hour)
# users.join(hours,["id"]).show()

#merge two dataframe by row
# +-----+
# |myCol|
# +-----+
# |    0|
# |    1|
# |    2|
# |   20|
# +-----+
firstDF = spark.range(3).toDF("myCol")
newRow = spark.createDataFrame([[20]])#spark.createDataFrame([(20,)])
appended = firstDF.union(newRow)

#merge two dataframe by column
# +-----+----------+--------+------+
# | name|      date|duration|upload|
# +-----+----------+--------+------+
# |alice|2015-04-23|      10|   100|
# |  bob|2015-01-13|       4|    23|
# |  bob|2015-01-13|       5|    23|
# +-----+----------+--------+------+
llist = [('bob', '2015-01-13', 4), ('alice', '2015-04-23',10),('bob', '2015-01-13', 5)]
left = spark.createDataFrame(llist, ['name','date','duration'])
right = spark.createDataFrame([('alice', 100),('bob', 23)],['name','upload'])
df = left.join(right, ["name"])


#define udf function to carry out complex logic on column of dataframe
def assign_tod(hr):
    if hr>7: return 'morning'
    elif hr>12: return 'lunch'
    elif hr>14: return 'afternoon'
    elif hr>18: return 'evening'
    else: return 'night'
    # times_of_day = {
    # 'morning' : range(7, 12),
    # 'lunch' : range(12, 14),
    # 'afternoon' : range(14, 18),
    # 'evening' : range(18, 23),
    # 'night' : range(23, 7)
    # }
    # for k, v in times_of_day.items():
    #     if hr in v:
    #         return k

period=udf(assign_tod,StringType())
hours.withColumn("period",period("hour")).show()

#-----------------------------------text feature----------------------------------
raw_titles = movie_fields.map(lambda fields: Row(title=fields[1]))
raw_titles=spark.createDataFrame(raw_titles)

#extract film's name
def extract_title(raw):
    import re##a="Seven (Se7en) (1995)"
    grps=re.search("\((\w+)\)",raw)#<_sre.SRE_Match object; span=(6, 13), match='(Se7en)'>
    if grps:
        return raw[:grps.start()].strip()#grps.start()=6
    else:
        return raw

title=udf(extract_title,StringType())
raw_titles.withColumn("raw_title",title("title")).show()


raw_titles2 = movie_fields.map(lambda fields: fields[1])
movie_titles = raw_titles2.map(lambda m: extract_title(m))
title_terms = movie_titles.map(lambda t: t.split(" "))

#zipWithIndex map value->(value,idx):[('Henry', 2625), ('V\ufffdronique,', 2626), ('Mystery', 2627), ('Exotica', 2628), ('Somewhere', 2629), ('Rangers', 2630),...]
#collectAsMap returns result as dict:{'(Pred': 1208, 'Ado': 1324, 'Hill': 1326, 'Jackie': 1566, 'Rich': 1229, 'V\ufffdronique,': 2626, 'Horseman': 1834, 'Don': 1329,...}
all_terms_dict2 = title_terms.flatMap(lambda x: x).distinct().zipWithIndex().collectAsMap()
#print "Index of term 'Dead': %d" % all_terms_dict2['Dead'] --->Index of term 'Dead': 147

def create_vector(terms,dicts):
    from scipy import sparse as sp
    num_len=len(dicts)
    matrix=sp.csr_matrix((1,num_len))
    for x in terms:
        if x in dicts:
            matrix[0,dicts[x]]=1
    return matrix

#all_terms_dict2 can be very large in real case, broadcast it to reduce memory usage.
all_terms_bcast = sc.broadcast(all_terms_dict2)
#use '.value' to get the value of broadcast value
term_vectors = title_terms.map(lambda terms: create_vector(terms,all_terms_bcast.value))


#-----------------------------------------data exploration-------------------------------
#keyBy:('378', Row(itemid='78', rating='3', timestep=880056976, userid='378')), ('880', Row(itemid='476', rating='3', timestep=880175444, userid='880')), ('716', Row(itemid='204', rating='5', timestep=879795543, userid='716')), ('276', Row(itemid='1090', rating='1', timestep=874795795, userid='276')), ('13', Row(itemid='225', rating='2', timestep=882399156, userid='13')),moviesForUser=rate_frame2.keyBy(lambda x: x.userid).lookup("789")
#lookup:find all items that belong to a key
moviesForUser=rate_frame2.keyBy(lambda x: x.userid).lookup("789")
#sort Row rdd by sorted()
list(map(lambda x : (titles2[int(x.itemid)],x.rating),sorted(moviesForUser, key=lambda x : x.rating,reverse=True)[:10]))

#alternatively, use filter when the data format is Dataframe
# +------+------+---------+------+
# |itemid|rating| timestep|userid|
# +------+------+---------+------+
# |  1012|     4|880332169|   789|
# |   127|     5|880332039|   789|
# |   475|     5|880332063|   789|
# |    93|     4|880332063|   789|
# |  1161|     3|880332189|   789|
# |   286|     1|880332039|   789|
#......
rate_frame.filter(rate_frame["userid"]=="789").show()
#use Dateframe to find top rated movies by user 789
rate_frame.join(titles,["itemid"],"left").filter(rate_frame["userid"]=="789").sort(rate_frame.rating.desc()).show()
