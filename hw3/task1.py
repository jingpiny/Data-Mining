from pyspark import SparkContext
from itertools import islice
import sys
import time
from itertools import combinations

start = time.time()
sc = SparkContext('local[*]', 'task1')
in_file = 'yelp_train.csv'
out_file = 'task1_result.csv'

# define the parameters
similarity = 0.5
num_hash = 50
num_r = 2
num_b = 25

# read the file
rdd_text = sc.textFile(in_file)
# get rid of header: user_id, business_id, star
rdd = rdd_text.map(lambda line: line.split(',')).filter(lambda line: 'business_id' not in line).map(lambda x: (x[0], x[1]))
user_rdd = rdd.map(lambda x: x[0]).distinct().zipWithIndex()  # (userID, userIndex) order by x[1] the index
# user_dict = user_rdd.collectAsMap()  # can access x[1] index value using x[0] user_id
num_user = user_rdd.count()  # find number of users = 11270
record_user = rdd.map(lambda x: x[0]).zipWithIndex().join(user_rdd)
record_user_idx = record_user.map(lambda x: x[1])   # (recordIndex, userIndex)

bus_rdd = rdd.map(lambda x: x[1]).distinct().zipWithIndex()
# bus_dict = bus_rdd.map(lambda x: (x[1], x[0])).collectAsMap()  # (BusinessIndex, businessID): access id using index
num_bus = bus_rdd.count()  # find number of business = 24732
record_bus = rdd.map(lambda x: x[1]).zipWithIndex().join(bus_rdd)  # give each record/row an index
record_bus_idx = record_bus.map(lambda x: x[1])  # (recordIndex, businessIndex)

record_bus_user = record_bus_idx.join(record_user_idx)
bus_user_idx = record_bus_user.map(lambda x: x[1])
# create business buckets (business, (user1, user2...))
bus_user_bucket = bus_user_idx.groupByKey()
# print(bus_user_bucket.map(lambda x: (x[0], list(x[1]))).collect())
bus_user_set = bus_user_bucket.mapValues(set)  # number: 407
bus_user_dict = bus_user_set.collectAsMap()  # access each user set using business id
# print(list(islice(user_dict.items(), 5)))


def prime(x, y):
    prime_list = []
    for i in range(x, y):
        if i == 0 or i == 1:
            continue
        else:
            for j in range(2, int(i / 2) + 1):
                if i % j == 0:
                    break
            else:
                prime_list.append(i)
        if len(prime_list) == 50:
            break
    return prime_list


def min_hash(userId_set):
    hashid_lst = list()  # a list of set that stores
    hashed_min = list()
    for i in range(num_hash):
        hashid_lst.append(set(map(lambda x: (a_lst[i] * x + b_lst[i]) % m, userId_set)))
    for i in hashid_lst:
        hashed_min.append(min(i))
    return hashed_min  # return the min index of each businessID from each hash function


def divide(sig_lst):
    divide_lst = list()
    for i in range(num_b):
        divide_lst.append((i, tuple(sig_lst[num_r * i:num_r * (i+1)])))
    return divide_lst  # [(band0, [h1min, h2min]), (band1, [h3min, h4min]), ....]

# def find_candidate(bus_id1, bus_id2):
#     for band in range(num_b):
#         if bus_sig_div_dict[bus_id1][band] == bus_sig_div_dict[bus_id2][band]:
#             return True
#     return False

def jaccard_sim(bus1, bus2):
    # find intersection of user sets between two businesses
    intersect = set(bus_user_dict[bus1]).intersection(set(bus_user_dict[bus2]))
    # find the union of user sets of two businesses
    union = set(bus_user_dict[bus1]).union(set(bus_user_dict[bus2]))
    # divide the intersection with the union
    return len(intersect)/len(union)


def sorting(lst):
    if lst[0] > lst[1]:
        return (lst[1], lst[0], lst[2])
    else:
        return (lst[0], lst[1], lst[2])

# generate list for parameter a and b.
startA = 100
endA = 700
a_lst = prime(startA, endA)
# num_user = 11270
startB = 8000
endB = 10000
b_lst = prime(startB, endB)
m = num_user

bus_sig = bus_user_set.mapValues(lambda x: min_hash(x))  # (businessID, [h1, h2, ...])
band_rdd = bus_sig.flatMap(lambda x: [(bandidx_sig, x[0]) for bandidx_sig in divide(x[1])])
# ((bandIdx, [h1, h2]), businessIdx)
# group business index if they are identical in the same band with the same two hashed indice
band_bus_bucket = band_rdd.groupByKey()  # 51w
# filter out the band bucket with only one business index, which means there is no other business same with it
band_bus_bucket_filter = band_bus_bucket.filter(lambda x: len(x[1]) > 1)  # 5w
# generate pair combination out of each business index bucket and filter out duplicated combinations
bus_pair = band_bus_bucket_filter.map(lambda x: x[1]).flatMap(lambda x: [p for p in list(combinations(x, 2))])
bus_unique_pair = bus_pair.distinct()

##############################
# bus_sig_div = bus_sig.mapValues(lambda x: divide(x))  # (businessID, [[h1, h2],[h3, h4]...])
# bus_sig_div_dict = bus_sig_div.collectAsMap()
#
# # generate pair combinations of business index
# bus_idx_rdd = sc.parallelize([*range(num_bus)])
# # [(bus1, bus2), (bus1, bus3), ...]
# bus_pair_rdd = bus_idx_rdd.flatMap(lambda busid: [(busid,nextbusid) for nextbusid in range(busid + 1, num_bus)])
# print(bus_pair_rdd.count())
#
# # filter out pair of business ids that are identical in at least one basket
# bus_pair_filter = bus_pair_rdd.filter(lambda pair: find_candidate(pair[0], pair[1]))
###############################

# compute Jaccard Similarity
result = bus_unique_pair.map(lambda x: (x[0], x[1], jaccard_sim(x[0], x[1])))
result_filter = result.filter(lambda x: x[2] >= similarity)  # (busIdx, busIdx, sim)
bus_dict = bus_rdd.map(lambda x: (x[1], x[0])).collectAsMap()  # (busIdx, busId)
convert_res = result_filter.map(lambda x: (bus_dict[x[0]], bus_dict[x[1]], x[2]))
# convert_res_list = convert_res.collect()

# sort the answer
res_sort = convert_res.map(lambda x: sorting(x)).sortBy(lambda x: x[0])
res = res_sort.collect()
# sorting_id = list()
# for i in convert_res_list:
#     # get rid of similarity number and sort the id
#     sorting_id.append(sorted(list(i).pop(2)))
# print("Duration7: {0:.2f}".format(time.time() - start))
# # add the similarity back to the list
# for i in range(len(convert_res_list)):
#     sorting_id[i].append(convert_res_list[i][2])
# res = sorted(sorting_id)
# write out the answer
with open(out_file, 'w') as f:
    f.write('business_id_1, business_id_2, similarity\n')
    for i in res:
        f.write(i[0] + ',' + i[1] + ',' + str(i[2]) + '\n')
f.close()

print("Duration: {0:.2f}".format(time.time() - start))