import sys
from pyspark import SparkContext
import time
from operator import add
import math

def single_item(baskets):
    unique_items = set()
    for basket in baskets:
        for item in basket:
            unique_items.add(frozenset({item}))
    return unique_items

def generate(frequentSet, num):
    candidate = set()
    for i in frequentSet:
        for j in frequentSet:
            if len(i.union(j)) == num:
                candidate.add(i.union(j))
    return candidate

def count_frequent(k_filtered, thre, baskets):
    k_count = dict.fromkeys(k_filtered, 0)
    # check the count of k combination of id
    frequent_k = set()
    for basket in baskets:
        for k_com in k_filtered:
            if set(k_com).issubset(basket):
                k_count[k_com] += 1
    frequent_k = set()
    for k_com, count in k_count.items():
        if count >= thre:
            frequent_k.add(k_com)
    return frequent_k

def aprior(thre, baskets):
    # store all business_id values in a list to prepare for the counting
    items_unique = single_item(baskets)
    frequent_one = dict()
    # check if the single item is greater than the threshold
    # and then store the count {"id":count}
    outputSet = count_frequent(items_unique, thre, baskets)
    frequentSet= outputSet
    k = 1
    # store all the frequent id without count in a list to prepare for the pair combination
    phase_one_candidates = dict() # store the results
    phase_one_candidates[1] = outputSet

    while (frequentSet):
        k += 1
        # find possible pairs
        # print(k)
        #print("Duration1: {0:.2f}".format(time.time() - start))
        k_possible = generate(frequentSet, k)
        # # check what pairs are frequent: FILTER
        #print("Duration2: {0:.2f}".format(time.time() - start))
        frequent_k = count_frequent(k_possible, thre, baskets)
        #print("Duration3: {0:.2f}".format(time.time() - start))
        phase_one_candidates[k] = frequent_k
        frequentSet = frequent_k

    # extract tuples and store them into a list
    ap_candidate_list = set()
    for value in phase_one_candidates.values():
        for ids in value:
            if isinstance(ids, str):
                ap_candidate_list.add(tuple([ids]))
            else:
                ap_candidate_list.add(ids)
    # print("ap_candidateList is:", ap_candidate_list)
    # print(len(ap_candidate_list))
    return ap_candidate_list  # a dictionary of this form{k: []}


def phase_one(len_baskets, part_baskets):
    # take the ceiling of the threshold
    part_baskets = list(part_baskets)
    thre = math.ceil((len(part_baskets)/len_baskets)*support)
    # print("threshold:", threshold)
    candi = aprior(thre, part_baskets)

    return candi


def phase_two(all_candidate, part_basket):
    candi_count = dict.fromkeys(all_candidate, 0)
    for basket in part_basket:
        for ids in all_candidate:
            if set(ids).issubset(basket):
                candi_count[ids] += 1
    return [(key, value) for key, value in candi_count.items()]


def sorting(unsorted):
    sorted_list = list()
    for i in unsorted:
        sorted_list.append(sorted(i))
    return sorted_list


def sorting_general(unsorted):
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

    sc = SparkContext('local[*]', 'task2')
    threshold = 20
    support = 50
    in_file = 'ta_feng_all_months_merged.csv'
    out_file = 'mytest1.csv'

    # read and process the data
    rdd_text = sc.textFile(in_file)
    cusDatePro = rdd_text.filter(lambda x: "CUSTOMER_ID" not in x).map(lambda x: x.split(',')).map(lambda x: [str(x[0][1:-5]+x[0][-3:-1]+'-'+ x[1][1:-1]), int(x[5][1:-1])])
    cusDatePro = cusDatePro.collect()
    # write out the csv file
    with open('Customer_product.csv', 'w') as f1:
        f1.write('DATE-CUSTOMER_ID'+','+'PRODUCT_ID'+'\n')
        for i in cusDatePro:
            f1.write(i[0] + ',' +'{}'.format(i[1]) + '\n')
    f1.close()

    # # read the intermediate file
    # file = 'Customer_product.csv'
    # rdd_text_1 = sc.textFile(file)
    # rows = rdd_text_1.filter(lambda x: 'PRODUCT_ID' not in x).map(lambda x: x.split(',')).map(lambda x: (x[0], x[1]))
    # # print(rows.collect())
    # # create basket
    # baskets_rdd = rows.groupByKey().mapValues(set).filter(lambda x: len(x[1])>threshold).cache()
    # baskets_rdd = baskets_rdd.map(lambda x: x[1]).cache()
    # # print(baskets_rdd.collect())
    # length = baskets_rdd.count()
    #
    # candidates = baskets_rdd.mapPartitions(lambda basket: phase_one(length, basket)).distinct()
    # candidatesList = candidates.collect()
    #
    # sorted_candi = sorting(candidatesList)
    # candid = sorting_general(sorted_candi)
    #
    # # phase two
    # # check all candidates in each baskets
    # result = baskets_rdd.mapPartitions(lambda basket: phase_two(candidatesList, basket))
    # final_result = result.reduceByKey(add).filter(lambda x: x[1] >= support).map(lambda x: x[0])
    #
    # # sort the final result
    # sorted_final = sorting(final_result.collect())
    # final = sorting_general(sorted_final)
    #
    #
    # # write out the result
    # with open(out_file, 'w') as f2:
    #     f2.write('Candidates:'+'\n')
    #     write_out(candid, f2)
    #     f2.write('Frequent Itemsets:'+'\n')
    #     write_out(final, f2)
    # f2.close()
    #
    # end = time.time()
    # duration = end - start
    # print("Duration: {0:.2f}".format(duration))

