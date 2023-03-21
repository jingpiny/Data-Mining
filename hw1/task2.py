from pyspark.sql import SparkSession
from pyspark import SparkContext
import sys
import json
from operator import add
import time

# spark = SparkSession.builder \
#       .master("local[*]") \
#       .appName("task2") \
#       .getOrCreate()
# only one partition
sc = SparkContext('local[*]', 'task2')
# two partitions

in_file = sys.argv[1]
out_file = sys.argv[2]
n_partition_customize = int(sys.argv[3])

rdd_text = sc.textFile(in_file)
rdd = rdd_text.map(lambda x: json.loads(x))

# The number of Partitions
n_partition_default = rdd.getNumPartitions()
# The number of items in each Partition
n_items_default = rdd.glom().map(len).collect()
# Time for execution
start_default = time.time()
top10_business_default = rdd.map(lambda x: (x['business_id'], 1)).reduceByKey(add).sortBy(lambda x: (x[1], x[0])). \
    top(10, key=lambda x: x[1])
end_default = time.time()
time_default = end_default - start_default

# customized:
rdd_text = sc.textFile('test_review.json')
rdd = rdd_text.map(lambda x: json.loads(x))
rdd = rdd.map(lambda x: (x['business_id'], 1)).partitionBy(n_partition_customize)
# # number of items in each partition
n_items_customize = rdd.glom().map(len).collect()
# execution time
start_customize = time.time()
top10_business_customize = rdd.reduceByKey(add).sortBy(lambda x: (x[1], x[0])). \
    top(10, key=lambda x: x[1])
end_customize = time.time()
time_customize = end_customize - start_customize

output_dict = {
    "default": {
        "n_partition": n_partition_default,
        "n_items": n_items_default,
        "exe_time": time_default
    },
    "customized": {
        "n_partition": n_partition_customize,
        "n_items": n_items_customize,
        "exe_time": time_customize
    }
}
with open(out_file, 'w') as f:
    f.write(json.dumps(output_dict))

f.close()
