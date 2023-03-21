import time
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from graphframes import GraphFrame
import os

# os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11"


threshold = 7
in_file = 'ub_sample_data.csv'
out_file = 'myoutput.txt'
#
# sc = SparkContext('local[*]', 'task1')
# sqlContext = SQLContext(sc)
conf = SparkConf().setMaster("local").setAppName("task1")
sc = SparkContext.getOrCreate(conf=conf)
sqlContext = SQLContext(sc)
text_rdd = sc.textFile(in_file)

raw_rdd = text_rdd.map(lambda x: x.split(',')).filter(lambda x: 'user_id' not in x).map(lambda x: (x[1], x[0]))
# (businessID, userID)
# eliminate business that only has review from one user
bus_userpair = raw_rdd.join(raw_rdd).filter(lambda x: x[1][0] != x[1][1])
# find the user pairs that have the number of common reviewed businesses greater than the threshold
userpair_bus_set = bus_userpair.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).\
                    filter(lambda x: len(x[1]) >= threshold)
user_bus_set = raw_rdd.map(lambda x:(x[1], x[0])).groupByKey()  # user number: 3374

userpair_bus_set_lst = userpair_bus_set.collect()
userpair = userpair_bus_set.map(lambda x: (x[0][0], x[0][1])).collect()
temp = set()  # store all the user/nodes
for i in userpair_bus_set_lst:
    for j in i[0]:
        if j not in temp:
            temp.add(j)
# (user, (bus1, bus2))
filter_user_bus_set = user_bus_set.filter(lambda x: x[0] in temp)
# vertices = sqlContext.createDataFrame(filter_user_bus_set, ["userid", "businessSet"])
# edges = sqlContext.createDataFrame(userpair, ["src", "dst"])
vertices = sc.parallelize(list(temp)).map(lambda x: (x,)).toDF(['id'])
edges = sc.parallelize(userpair).toDF(["src", "dst"])
g = GraphFrame(vertices, edges)
result = g.labelPropagation(maxIter=5)
res_rdd = result.rdd
res = res_rdd.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: sorted(list(x[1]))).sortBy(lambda x: len(x))
print(res)
with open(out_file, 'w') as f:
    for i in res:
        f.writelines(str(i)[1:-1] + '\n')
f.close()

