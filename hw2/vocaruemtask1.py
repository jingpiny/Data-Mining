import sys
from pyspark import SparkContext
import time
from operator import add
import math

"""
A function that processes all the baskets and extracts all distinct items
@ param: baskets
a list storing all the baskets
@ return: unique_items
a set of unique items
"""
def single_item(baskets):
    unique_items = set()
    for basket in baskets:
        for item in basket:
            unique_items.add(frozenset({item}))  # storing each item in a immutable frozen set frozenset({x,y,z})
    return unique_items

"""
A function tha creates all set combinations of items
@ param: frequentSet
a set of all frequent sets that is passed from the last round
@ param: num
value k, the size of the combination
@ return: candidate 
a set of sets that stores all combinations of items
"""
def generate(frequentSet, num):
    candidate = set()
    for i in frequentSet:
        for j in frequentSet:
            # find the union of two sets that have a size of next k
            if len(i.union(j)) == num:
                candidate.add(i.union(j))
    return candidate  # these are candidates that might have subsets that are not frequent

"""
The function that counts each unique item-set in each basket and filter out item sets that pass the support
@ param: k_filtered
a list that each of its element having a size of k items
@ param: threshold
local support
@ param: baskets
A list of baskets
# return: frequent_k
A set that stores all the frequent item sets with a size of k items
"""
def count_frequent(k_filtered, threshold, baskets):
    # a dictionary that stores the count of each combination
    k_count = dict.fromkeys(k_filtered, 0)
    # check the count of k combination of id
    for basket in baskets:  # loop through each basket in the basket list
        for k_com in k_filtered:  # loop through each combination
            if set(k_com).issubset(basket):  # check if the combination exists in the basket
                k_count[k_com] += 1  # add one if it exists
    frequent_k = set()  # store all the combinations that exceed the threshold {{}, {}, ...}
    for k_com, count in k_count.items():
        if count >= threshold:
            frequent_k.add(k_com)
    return frequent_k

"""
The main part of a-priori algorithm that finds the frequent item set
@ param: threshold
A fraction of the support based on the size of the chunk: p*s
@ param: baskets
A list of baskets
"""
def aprior(threshold, baskets):
    # store all business_id values in a list to prepare for the counting
    items_unique = single_item(baskets)
    frequent_one = dict()  # didn't used
    # check if the single item is greater than the threshold, and then store the count {"id":count}
    outputSet = count_frequent(items_unique, threshold, baskets)
    frequentSet= outputSet  # serve as a boolean to check if there are any frequent sets
    k = 1  # the number of items of the combination sets
    # store all the frequent id without count in a list to prepare for the pair combination
    phase_one_candidates = dict() # store the results {k: a set of sets}
    phase_one_candidates[1] = outputSet

    # loop until there is no frequent sets in all the baskets
    while (frequentSet):
        k += 1
        # find item sets combination with size of k
        k_possible = generate(frequentSet, k)
        # check what pairs are frequent: FILTER
        frequent_k = count_frequent(k_possible, threshold, baskets)
        phase_one_candidates[k] = frequent_k
        frequentSet = frequent_k

    # extract tuples and store them into a list
    ap_candidate_list = set()
    # loop through each
    for value in phase_one_candidates.values():
        for ids in value:
            # check if it is a single item. If it is, then store it as a tuple ('1'), ('2')
            if isinstance(ids, str):
                ap_candidate_list.add(tuple([ids]))
            else:  # store all non-singleton item sets as it is {'1', '2'}, {'2', '3'}
                ap_candidate_list.add(ids)
    return ap_candidate_list  # a set

"""
First phase of Apriori algorithm: mapping all the cadidates
@ param: len_basket
The total number of baskets of the entire item set
@ param: part_baskets
The baskets in one of the partitions
@ return: candi
A set with all candidate item sets
"""
def phase_one(len_baskets, part_baskets):
    # take the ceiling of the threshold
    part_baskets = list(part_baskets)
    threshold = math.ceil((len(part_baskets)/len_baskets)*support)
    candi = aprior(threshold, part_baskets)
    return candi

"""
Count the frequency of each combination in the partition
@ param: call_candidate
a list of all candidate item-sets 
@ param: part_basket
@ return: a list of tuples with its key is the item set and its value is the count
"""
def phase_two(all_candidate, part_basket):
    candi_count = dict.fromkeys(all_candidate, 0)
    for basket in part_basket:
        for ids in all_candidate:
            if set(ids).issubset(basket):
                candi_count[ids] += 1
    return [(key, value) for key, value in candi_count.items()]


"""
Sort each element within the list
@ return: sorted_list
return a list that stores all sorted elements
"""
def sorting(unsorted):
    sorted_list = list()
    for i in unsorted:
        sorted_list.append(sorted(i))
    return sorted_list


"""
Sort the list in general
@ return: sorted_list
return a sorted list
"""
def sorting_general(unsorted):
    # sort the list based on its length first, and then based on its value
    sorted_list = sorted(unsorted, key=lambda item: (len(item), item))
    return sorted_list


def write_out(itemsets, file):
    separator = ','
    item_len = 1
    start_point = 0
    for item in itemsets:
        if len(item) > item_len and len(item) == 1:
            idx = itemsets.index(item)
            file.write(separator.join("('{}')".format(i[0]) for i in itemsets[start_point:idx]))
            file.write('\n\n')
            start_point = idx
            item_len += 1
        elif len(item) > item_len and len(item) > 1:
            idx = itemsets.index(item)
            file.write(separator.join(str(i) for i in itemsets[start_point:idx]).replace('[', '(').replace(']', ')'))
            file.write('\n\n')
            start_point = idx
            item_len += 1
    file.write(separator.join(str(i) for i in itemsets[start_point:len(itemsets)]).replace('[', '(').replace(']', ')'))
    file.write('\n\n')


if __name__ == '__main__':

    start = time.time()

    sc = SparkContext('local[*]', 'task1')
    case = int(1)
    support = int(4)
    in_file = 'small1.csv'
    out_file = 'mytest2.csv'
    rdd_text = sc.textFile(in_file)  # read the csv file
# identify the case number to create baskets for each user
    if case == 1:  # group by user's id
        rdd_rows = rdd_text.map(lambda line: line.split(','))  # get rid of the deliminator, return a list of list
        header = rdd_rows.first()
        rows = rdd_rows.filter(lambda x: x != header)
        # rows = rdd_rows.filter(lambda x: 'user_id' not in x)  # get rid of the header
        row_cnt = rows.count()
        print(row_cnt)
        rows = rows.map(lambda x: (x[0], x[1])).distinct() # make the list a tuple and get rid of the duplicate
        # create basket
        baskets_rdd = rows.groupByKey().mapValues(list).cache()
        # group by the userid and map all values (businessid) into a list
    else:  # group by business's id
        rdd_rows = rdd_text.map(lambda line: line.split(','))
        rows = rdd_rows.filter(lambda x: 'user_id' not in x)
        rows = rows.map(lambda x: (x[1], x[0])).distinct()
        baskets_rdd = rows.groupByKey().mapValues(list).cache()

    baskets_rdd = baskets_rdd.map(lambda x: x[1]).cache()  # a list of lists storing all the baskets
    length = baskets_rdd.count()  # total number of baskets

    # mapPartitions(f): apply function f on each partition of this rdd
    # distinct(): because the rdd is processed on the basis of partitions, there might be duplicates between partitions
    candidates = baskets_rdd.mapPartitions(lambda basket: phase_one(length, basket)).distinct()
    candidatesList = candidates.collect()  # convert the set into a list of sets
    # sort the candidate list of sets
    # these are candidates because Phase1 return only lists that are frequent in each partition, but not the entire set
    sorted_candi = sorting(candidatesList)
    candid = sorting_general(sorted_candi)

    # Phase two
    # check all candidates in each baskets
    # count the total number of frequency in the entire
    result = baskets_rdd.mapPartitions(lambda basket: phase_two(candidatesList, basket))
    # reduce all the result
    # the support threshold used here is the original support rather than a fraction of it
    final_result = result.reduceByKey(add).filter(lambda x: x[1] >= support).map(lambda x: x[0])

    # sort the final result
    sorted_final = sorting(final_result.collect())
    final = sorting_general(sorted_final)


    # write out the result
    with open(out_file, 'w') as f:
        f.write('Candidates:'+'\n')
        write_out(candid, f)
        f.write('Frequent Itemsets:'+'\n')
        write_out(final, f)
    f.close()

    end = time.time()
    duration = end - start
    print("Duration: {0:.2f}".format(duration))
    print("{0:.3f}".format(4))