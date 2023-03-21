from pyspark import SparkContext
import sys
import time
import math
from itertools import islice


start = time.time()
sc = SparkContext('local[*]', 'task21')

train_file = 'yelp_train.csv'
test_file = 'yelp_val.csv'
out_file = 'task21output.csv'
text_train_file = sc.textFile(train_file)
text_test_file = sc.textFile(test_file)
# create train and test rdd  (userID, businessID, rating)
train_rdd = text_train_file.map(lambda x: x.split(',')).filter(lambda x: 'business_id' not in x) \
            .map(lambda x: (x[0], x[1], float(x[2])))
test_rdd = text_test_file.map(lambda x: x.split(',')).filter(lambda x: 'business_id' not in x) \
            .map(lambda x: (x[0], x[1]))
# calculate the overall average rating of all business by each user
init = (0, 0)
user_rdd = train_rdd.map(lambda x: (x[0], x[2]))  # (userID, rating)
user_sum_rating = user_rdd.aggregateByKey(init, lambda x, y: (x[0]+y, x[1]+1), lambda x, y: (x[0]+y[0], x[1]+y[1]))
user_avg_rating = user_sum_rating.mapValues(lambda value: value[0]/value[1]).collectAsMap()  # (userID, average_rating)
# calculate the average rating of all users on each business
bus_rdd = train_rdd.map(lambda x: (x[1], x[2]))  # (businessID, rating)
bus_sum_rating = bus_rdd.aggregateByKey(init, lambda x, y: (x[0]+y, x[1]+1), lambda x, y: (x[0]+y[0], x[1]+y[1]))
bus_avg_rating = bus_sum_rating.mapValues(lambda value: value[0]/value[1]).collectAsMap()  # (businessID, average_rating)

# different users and businesses
user_lst = user_rdd.groupByKey().map(lambda x: x[0]).collect()
bus_lst = bus_rdd.groupByKey().map(lambda x: x[0]).collect()
# storing userID/businessID with its corresponding businessIDs/userIDs in a bucket
bus_user_bucket = train_rdd.map(lambda x: (x[1], x[0])).groupByKey()
user_bus_bucket = train_rdd.map(lambda x: (x[0], x[1])).groupByKey()
# create ((user, business), rating) to locate each rating: access rating using (userID, busID)
user_bus_idx_rating = train_rdd.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()
# access the user set/business set by businessID/userID
bus_user_dict = bus_user_bucket.mapValues(set).collectAsMap()
user_bus_dict = user_bus_bucket.mapValues(set).collectAsMap()

def calculate_w(bus_set, user_id, bus_id):
    w_list = dict()
    # find the co-users of each candidate business pair
    for bus in bus_set:
        couser = bus_user_dict[bus_id].intersection(bus_user_dict[bus])
        # business j: have number of co-users with i greater than N = 20
        # because there will be too many pairs if no constraint is set
        if len(couser) >= 20:
            # loop through each business pair and calculate W
            numerator = 0
            deno_i = 0
            deno_j = 0
            for i in couser:
                numerator = numerator + (user_bus_idx_rating[(i, bus_id)] - bus_avg_rating[bus_id]) * \
                            (user_bus_idx_rating[(i, bus)] - bus_avg_rating[bus])
                deno_i = pow((user_bus_idx_rating[(i, bus_id)] - bus_avg_rating[bus_id]), 2) + deno_i
                deno_j = pow((user_bus_idx_rating[(i, bus)] - bus_avg_rating[bus]), 2) + deno_j
            # after loop through all the co-user in the U set, square the denominator and do the division
            if deno_i == 0:
                return bus_avg_rating[bus_id]
            elif deno_j == 0:
                continue  # skip this business
            else:
                sqr_deno_i = math.sqrt(deno_i)
                sqr_deno_j = math.sqrt(deno_j)
                denominator = sqr_deno_i * sqr_deno_j
                w = numerator / denominator
                w_list[bus] = w
        else:
            continue
    return w_list

def predict(user_id, bus_id):
    if user_id in user_lst and bus_id in bus_lst:
        # all other rated business by this user
        bus_set = user_bus_dict[user_id]
        w_lst = calculate_w(bus_set, user_id, bus_id)
        # calculate the prediction
        # only calculate the pearson correlation if there are at least 30 weights, otherwise, return the average
        top_n = 30
        if len(w_lst) >= top_n:
            # sort the weight in order
            sort_w_dict = dict(sorted(w_lst.items(), key = lambda item: item[1]))
            # get the top 30 weights
            top_w_dict = dict(islice(sort_w_dict.items(), top_n))
            # get the keys of the key-value pairs
            sort_busid = list(top_w_dict.keys())
            p_numerator = 0
            p_denomenator = 0
            for bus in sort_busid:
                p_numerator = p_numerator + user_bus_idx_rating[(user_id, bus)] * top_w_dict[bus]
                p_denomenator = p_denomenator + top_w_dict[bus]
                return p_numerator/p_denomenator
        else:
            return bus_avg_rating[bus_id]
        return None
    elif user_id in user_lst: # user exist but business doesn't, return average rating of that user
        return user_avg_rating[user_id]
    elif bus_id in bus_lst: # business exist but user doesn't, return average rating of that business by all users
        return bus_avg_rating[bus_id]

def root_mean_squared_error(actual, predict):
    sum_square = 0
    for i in range(len(actual)):
        sum_square = sum_square + (actual[i] - predict[i]) ** 2
    mean = sum_square / len(actual)
    root_mean = math.sqrt(mean)
    return root_mean


# input the test data and generate the prediction
pred = test_rdd.map(lambda x: (x[0], x[1], predict(x[0], x[1])))
final = pred.collect()
# write out the results to csv
with open(out_file, 'w') as f:
    f.write('user_id, business_id, prediction\n')
    for i in final:
        f.write(i[0] + ',' + i[1] + ',' + str(i[2]) + '\n')
f.close()

# test = text_test_file.map(lambda x: x.split(',')).filter(lambda x: 'business_id' not in x) \
#             .map(lambda x: float(x[2])).collect()
# error = root_mean_squared_error(test, pred.map(lambda x: x[2]).collect())
# print(error)

# print("Duration: {0:.2f}".format(time.time() - start))