# Import SparkSession
from pyspark.sql import SparkSession
from operator import add
import sys
import json

spark = SparkSession.builder \
      .master("local[*]") \
      .appName("task1") \
      .getOrCreate()

in_file = sys.argv[1]
out_file = sys.argv[2]

review = spark.read.json('in_file')

rdd = review.rdd

output = dict()

# The number of reviews:
n_review = rdd.count()
output['n_review'] = n_review
# The number of reviews in 2018:
n_review_2018 = rdd.filter(lambda row: '2018' in row[2]).count()
output['n_review_2018'] = n_review_2018
# The number of distinct users
n_user = rdd.map(lambda x: x[8]).distinct().count()
output['n_user'] = n_user
# Top 10 users with largest number of reviews and the largest number of reviews they have (list of tuples)
top10_user = rdd.map(lambda x: (x[8], 1)).reduceByKey(add).sortBy(lambda x: (x[1],x[0])).top(10, key=lambda x: x[1])
output['top10_user'] = top10_user
# The number of distinct businesses
n_business = rdd.map(lambda x: x[0]).distinct().count()
output['n_business'] = n_business
# Top 10 businesses with largest number of reviews and the largest number of reviews they have
top10_business = rdd.map(lambda x: (x[0],1)).reduceByKey(add).sortBy(lambda x: (x[1],x[0])).top(10, key=lambda x: x[1])
output['top10_business'] = top10_business

json_object = json.dumps(output, indent=2)


with open(out_file, 'w') as f:
    f.write(json_object)
f.close()
