from pyspark import SparkContext
import sys
import json
import time

sc = SparkContext('local[*]', 'task3')

in1_file = sys.argv[1]
in2_file = sys.argv[2]
out1_file = sys.argv[3]
out2_file = sys.argv[4]

review = sc.textFile(in1_file)
rdd_review = review.map(lambda x: json.loads(x))
business = sc.textFile(in2_file)
rdd_business = business.map(lambda x: json.loads(x))

# map task: (city, (sum, cnt))
# reduce task: (city, (all_sum, all_cnt)) --> (city, all_sum/all_cnt)
rdd1 = rdd_review.map(lambda x: (x['business_id'], x['stars']))
rdd2 = rdd_business.map(lambda x: (x['business_id'], x['city']))
# only join the business that appears in the test review
rdd_join = rdd1.join(rdd2).map(lambda x: (x[1][1], (x[1][0], 1)))  # (city, (stars, 1))
agg_rdd = rdd_join.reduceByKey(lambda u, v: (u[0] + v[0], u[1] + v[1]))  # (city, (sum, cnt))
avg_rdd = agg_rdd.map(lambda x: (x[0], x[1][0]/x[1][1]))
sorted_avg_rdd = avg_rdd.sortBy(lambda x: [-x[1], x[0]])

with open(out1_file, 'w') as f:
    f.write('city,stars')
    for i in sorted_avg_rdd:
        f.write(i[0] + ',' + str(i[1]) + '\n')
f.close()

####################
# print top 10 cities with highest stars
# method 1: sort and then print the 10
start1 = time.time()
avg_sort1 = sorted(avg_rdd.collect(), key=(lambda x: [-x[1], x[0]]))
cnt = 0
for i in avg_sort1:
    if cnt < 10:
        print(i)
        cnt += 1
    else:
        break
end1 = time.time()
m1 = end1 - start1

# method 2: sort in spark and then take the first 10 and print out
start2 = time.time()
avg_sort2 = avg_rdd.sortBy(lambda x: [-x[1], x[0]]).take(10)
for i in avg_sort2:
    print(i)
end2 = time.time()
m2 = end2 - start2

result = {
    "m1": m1,
    "m2": m2,
    "reason": "The method 1 is much faster than method 2. I think the reason behind is that for spark sorting, it needs to do the shuffling and sorting for each partition, which is more expensive when we are sorting small datasets. However, if the dataset is huge, then SPARK as a distributed processing system, it will be faster than the pure python sorting."
}

with open(out2_file, 'w') as f2:
    f2.write(json.dumps(result, indent=2))
f2.close()






