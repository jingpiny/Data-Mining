import sys
from sklearn.cluster import KMeans
import numpy as np
import time
import random
from collections import Counter, defaultdict
from itertools import combinations

def mahalanobis_dist(point, center, sd):
    sum_square = 0
    for i in range(num_dim):
        upper = point[i] - center[i]
        term = (upper/sd[i]) ** 2
        sum_square = sum_square + term
    return (sum_square ** 0.5)


def get_summary_init(label_set):
    summary = dict()
    for i in label_set:
        summary[i] = dict()
        summary[i]['N'] = list()  # store the index of the record in the original data
        summary[i]['SUM'] = [0] * num_dim
        summary[i]['SUMSQ'] = [0] * num_dim

    return summary


def get_stand_devi(summary_dict):
    sd_list = list()
    n = len(summary_dict['N'])
    sum_sumsq = list(zip(summary_dict['SUM'], summary_dict['SUMSQ']))
    for i in sum_sumsq:
        fst_term = i[1] / n
        snd_term = (i[0] / n) ** 2
        variance = fst_term - snd_term
        sd = variance ** 0.5
        sd_list.append(sd)

    return sd_list

in_file = 'hw6_clustering.txt'
n_cluster = 10
out_file = 'output.txt'

# read the text file
with open(in_file, 'r') as f:
    lines = f.readlines()
f.close()
rows = list()
for l in lines:
    rows.append(l.rstrip('\n').split(','))  # [[record1], [record2], [record3]...]
data_dict = dict()
for r in rows:
    data_dict[r[0]] = tuple(map(lambda x: float(x), r[2:]))  # {index(str):(feature1(float), feature2...)}
# get the features dimensions
num_dim = len(rows[0]) - 2

# step 1: randomly select 20% of the data
feature_list = list()
for k, v in data_dict.items():
    feature_list.append(list(v))  # [[pnt], [pnt], [pnt]]
feature_ndarray = np.array(feature_list)
sample_length = int(len(feature_ndarray) * 0.2)
rnd = 1
np.random.shuffle(feature_ndarray)
sample = list(feature_ndarray[:sample_length, :])
feature_ndarray = np.delete(feature_ndarray, slice(0,sample_length), 0)

# step 2: run Kmeans
kmeans = KMeans(n_clusters=n_cluster * 5).fit(sample)

# step 3: put cluster with only 1 point to the RS
train_labels = list(kmeans.labels_)
# count the number of points in a cluster
cluster_count = dict(Counter(train_labels))  # {cluster_index: number of points}
# find clusters with only one point
idx_rs = list()
for key, value in cluster_count.items():
    if cluster_count[key] == 1:
        # add the index of the point to a list
        idx_rs.append(train_labels.index(key))
# add those points to the RS
RS = list()
for i in idx_rs:
    RS.append(sample[i].tolist())  # [[feature1, feature2...], [feature1, feature2...]]

# discard these points from the sample
reverse_idx_rs = sorted(idx_rs)
for i in reversed(sorted(idx_rs)):
    sample.pop(i)

# step 4: run Kmeans again with K = input clusters
kmeans = KMeans(n_clusters=n_cluster).fit(sample)

# step 5: generate DS, summarize the statistics
labels_ds = dict(enumerate(kmeans.labels_))  # {index: cluster number}
cnt_ds = len(list(kmeans.labels_))
idx_ds = defaultdict(list)
cluster_data_ds = defaultdict(list)
centroid_ds = dict()
sd_ds = dict()
DS = list()

cluster_distinct = set(list(kmeans.labels_))  # get the label of each cluster
feature_index_dict = {v: k for k, v in data_dict.items()}  # {(pnt): index}

DS_summary = get_summary_init(cluster_distinct)  # {cluster: {N:[sample_index, ...], SUM:(), SUMSQ:()}, ...}

for k, v in labels_ds.items():  # k = index, v = cluster label
    cluster_data_ds[v].append(sample[k].tolist())
    idx_ds[v].append(feature_index_dict[tuple(sample[k])])
    DS_summary[v]['N'].append(feature_index_dict[tuple(sample[k])])
    DS_summary[v]['SUM'] = [sum(i) for i in zip(DS_summary[v]['SUM'], sample[k])]
    DS.append(sample[k].tolist())
    for i in range(num_dim):
        DS_summary[v]['SUMSQ'][i] += sample[k][i] ** 2

for k, v in DS_summary.items():  # k = label, v = dictionary with N, SUM, SUMSQ as keys
    # calculate centroid for each DS cluster
    centroid_ds[k] = [feature_sum/len(DS_summary[k]['N']) for feature_sum in DS_summary[k]['SUM']]
    # calculate SD for each DS cluster
    sd_ds[k] = get_stand_devi(v)

# step 6: run Kmeans on the RS to generate another CS and RS
cluster_data_cs = defaultdict(list)  # {cluster: [ [], [], ... ], ...} it stores the vector
CS = list()  # storing points
idx_cs = defaultdict(list)  # {cluster: [index, index, ...]} it stores the index of the vector in the original set
centroid_cs = dict()
sd_cs = dict()

idx_rs = list()
# pt_rs = list()

if len(RS) > 1:
    kmeans = KMeans(n_clusters=int(len(RS) / 2)).fit(RS)
    labels = list(kmeans.labels_)
    label_cnt_dict = dict(Counter(labels))  # {label: cnt}
    labels_multi_pnt = set()  # store the cluster label with multiple points
    for k, v in label_cnt_dict.items():
        if label_cnt_dict[k] == 1:
            idx_rs.append(labels.index(k))  # the index of point in RS
        #             pt_rs.append(RS[labels.index(k)])
        else:
            labels_multi_pnt.add(k)
    # add the cluster with multiple points to the CS
    if labels_multi_pnt:
        cnt_idx = 0
        for l in labels:
            if l in labels_multi_pnt:
                idx_cs[l].append(feature_index_dict[tuple(RS[cnt_idx])])
                cluster_data_cs[l].append(RS[cnt_idx])
                CS.append(RS[cnt_idx])
            cnt_idx += 1

        # get summary for CS
        CS_summary = get_summary_init(labels_multi_pnt)
        for k, v in cluster_data_cs.items():
            CS_summary[k]['N'].extend(v)
            for vector in v:  # a list of vector points
                CS_summary[k]['SUM'] = [sum(i) for i in zip(CS_summary[k]['SUM'], vector)]
                for i in range(num_dim):
                    CS_summary[k]['SUMSQ'][i] += vector[i] ** 2

        for k, v in CS_summary.items():
            centroid_cs[k] = [feature_sum / len(CS_summary[k]['N']) for feature_sum in CS_summary[k]['SUM']]
            sd_cs[k] = get_stand_devi(v)

    # if there are points in the RS classified into a cluster with only one point
    new_rs = list()
    if idx_rs:
        for i in idx_rs:
            new_rs.append(RS[i])
        RS = new_rs
    else:
        RS = new_rs

cnt_ds = len(DS)
cnt_cs_cluster = len(idx_cs)
cnt_cs = len(CS)
cnt_rs = len(RS)

print_lst = [cnt_ds, cnt_cs_cluster, cnt_cs, cnt_rs]
with open(out_file, 'w') as f:
    f.write('The intermediate results:\n')
    f.write('round {}: '.format(rnd) + ','.join(str(n) for n in print_lst) + '\n')

################################################################
while feature_ndarray:
    rnd = rnd + 1
    if rnd == 5:
        np.random.shuffle(feature_ndarray)
        sample = list(feature_ndarray[:, :])
        feature_ndarray = []
    else:
        # step 7: load another 20% of data
        np.random.shuffle(feature_ndarray)
        sample = list(feature_ndarray[:sample_length, :])
        feature_ndarray = np.delete(feature_ndarray, slice(0,sample_length), 0)

    threshold = 2 * (num_dim ** 0.5)
    for point in sample:
        point = point.tolist()
        pnt_clus_dist = list()  # store the cluster label and the distance
        for clus_label, center in centroid_ds.items():
            distance = mahalanobis_dist(point, center, sd_ds[clus_label])
            pnt_clus_dist.append((clus_label, distance))
        # find the shortest distance
        sort_pnt_clus_dist = sorted(pnt_clus_dist, key = lambda x: x[1])
        shortest_dist = sort_pnt_clus_dist[0][1]

        # step 8: check if the shortest distance is smaller than the threshold and assign to RS
        if shortest_dist < threshold:
            nearest_clus = sort_pnt_clus_dist[0][0]
            DS.append(point)
            cluster_data_ds[nearest_clus].append(point)
            idx_ds[nearest_clus].append(feature_index_dict[tuple(point)])
            # recalculate the summary, centroid and sd for that specific cluster
            DS_summary[nearest_clus]['N'].append(point)
            DS_summary[nearest_clus]['SUM'] = [sum(i) for i in zip(DS_summary[nearest_clus]['SUM'], point)]
            for i in range(num_dim):
                DS_summary[nearest_clus]['SUMSQ'][i] += point[i] ** 2
            value_N = len(DS_summary[nearest_clus]['N'])
            centroid_ds[nearest_clus] = [feature_sum/value_N for feature_sum in DS_summary[nearest_clus]['SUM']]
            sd_ds[nearest_clus] = get_stand_devi(DS_summary[nearest_clus])
        else: # step 9: assign to the nearest CS
            pntclus_dist_cs = list()
            for clus_label, center in centroid_cs.items():
                distance = mahalanobis_dist(point, center, sd_cs[clus_label])
                pntclus_dist_cs.append((clus_label, distance))
            sort_pntclus_dist_cs = sorted(pntclus_dist_cs, key = lambda x: x[1])
            shortest_dist_cs = sort_pntclus_dist_cs[0][1]
            if shortest_dist_cs < threshold:
                nearest_clus = sort_pntclus_dist_cs[0][0]
                CS.append(point)
                cluster_data_cs[nearest_clus].append(point)
                idx_cs[nearest_clus].append(feature_index_dict[tuple(point)])
                # recalculate the summary, centroid and sd for that specific cluster
                CS_summary[nearest_clus]['N'].append(point)
                CS_summary[nearest_clus]['SUM'] = [sum(i) for i in zip(CS_summary[nearest_clus]['SUM'], point)]
                for i in range(num_dim):
                    CS_summary[nearest_clus]['SUMSQ'][i] += point[i] ** 2
                value_N = len(CS_summary[nearest_clus]['N'])
                centroid_cs[nearest_clus] = [feature_sum/value_N for feature_sum in CS_summary[nearest_clus]['SUM']]
                sd_cs[nearest_clus] = get_stand_devi(CS_summary[nearest_clus])
            else: # assign to the RS if the distance exceeds the threshold with CS clusters
                RS.append(point)

    # step 11: run K-means on RS to generate CS and RS
    # determine the number of points in the RS and adjust the n_clusters parameter
    if len(RS) >= 5 * num_dim:
        cluster_input = 5 * num_dim
    elif len(RS) == 1:
        cluster_input = 1
    else:
        cluster_input = int(len(RS)/2)

    if RS:
        '''
        RS doesn't need cluster label because it is a single point
        Already have several existing CS clusters and DS clusters, so the newly-generated clustered points from 
        the RS will be classified as new CS clusters (need to assign them with new cluster labels).
        Add them into the CS list, cluster_data_cs dict, idx_cs dict, and then remove it from the RS list.
        !!! remove an element from the RS will change the index of the list
        '''
        kmeans = KMeans(n_clusters= cluster_input).fit(RS)
        labels = list(kmeans.labels_)
        label_cnt_dict = dict(Counter(labels))
        labels_multipnt_set = set()
        index_label = dict(enumerate(kmeans.labels_)) #{index: label, ...}
        for k, v in label_cnt_dict.items():
            if v != 1:
                labels_multipnt_set.add(k)
        if labels_multipnt_set:
            new_labels_set = set()
            for i in list(zip(RS, kmeans.labels_)):
                if i[1] in labels_multipnt_set:
                    new_label = 60000*rnd + i[1]  # create a new label for the new CS cluster
                    new_labels_set.add(new_label) # add the new label to the set
                    idx_cs[new_label].append(feature_index_dict[tuple(i[0])])
                    cluster_data_cs[new_label].append(i[0])
                    CS.append(i[0])
                    RS.remove(i[0])

            # update the CS summary by adding summaries of new clusters
            new_summary = get_summary_init(new_labels_set)
            for i in new_labels_set:
                for pnt in cluster_data_cs[i]:  # loop through a list of vector[[], []...]
                    new_summary[i]['N'].append(feature_index_dict[tuple(pnt)])
                    new_summary[i]['SUM'] = [sum(i) for i in zip(new_summary[i]['SUM'], pnt)]
                    for j in range(num_dim):
                        new_summary[i]['SUMSQ'][j] += pnt[j] ** 2
            CS_summary.update(new_summary)
            # update the centroid and sd for the new clusters
            for i in new_labels_set:
                value_N = len(CS_summary[i]['N'])
                centroid_cs[i] = [feature_sum/value_N for feature_sum in CS_summary[i]['SUM']]
                sd_cs[i] = get_stand_devi(CS_summary[i])

    # step 12: merge CS clusters with distance shorter than the threshold
    merge = True
    while merge:
        if len(centroid_cs) > 1:
            # find combinations of cluster pair
            clus_pair_lst = list(combinations(centroid_cs.keys(), 2))
            # compute the distance between two centroids
            cluspair_dist_lst = list()
            for pair in clus_pair_lst:
                clus_dist = mahalanobis_dist(centroid_cs[pair[0]], centroid_cs[pair[1]], sd_cs[pair[0]])
                cluspair_dist_lst.append((pair, clus_dist))
            # sort the pair distances in ascending order and merge from the nearest cluster pairs
            sort_cluspair_dist_lst = sorted(cluspair_dist_lst, key=lambda x: x[1])
            # compare the distance with the threshold
            if sort_cluspair_dist_lst[0][1] < threshold:  # [((centroid1, centroid2), distance)]
                # merge the two clusters and assign it with the first cluster's label
                clus1_centroid = sort_cluspair_dist_lst[0][0][0]
                clus2_centroid = sort_cluspair_dist_lst[0][0][1]
                merge_clus_lst = cluster_data_cs[clus1_centroid] + cluster_data_cs[clus2_centroid]
                merge_idx_lst = idx_cs[clus1_centroid] + idx_cs[clus2_centroid]
                cluster_data_cs[clus1_centroid] = merge_clus_lst
                idx_cs[clus1_centroid] = merge_idx_lst
                # delete cluster 2
                del cluster_data_cs[clus2_centroid]
                del idx_cs[clus2_centroid]
                del CS_summary[clus2_centroid]
                del centroid_cs[clus2_centroid]
                del sd_cs[clus2_centroid]

                # get new summary
                # update the CS summary by adding summaries of new clusters
                summary = dict()
                summary[clus1_centroid] = dict()
                summary[clus1_centroid]['N'] = list()
                summary[clus1_centroid]['SUM'] = [0] * num_dim
                summary[clus1_centroid]['SUMSQ'] = [0] * num_dim

                summary[clus1_centroid]['N'].extend(merge_clus_lst)
                for pnt in cluster_data_cs[clus1_centroid]:
                    CS_summary[clus1_centroid]['SUM'] = [sum(i) for i in zip(CS_summary[clus1_centroid]['SUM'], pnt)]
                    for j in range(num_dim):
                        CS_summary[clus1_centroid]['SUMSQ'][j] += pnt[j] ** 2

                value_N = len(CS_summary[clus1_centroid]['N'])
                centroid_cs[clus1_centroid] = [feature_sum / value_N for feature_sum in
                                               CS_summary[clus1_centroid]['SUM']]
                sd_cs[clus1_centroid] = get_stand_devi(CS_summary[clus1_centroid])

            else:  # stop merge if all the remaining CS clusters have a distance greater than the threshold
                merge = False
        else:
            merge = False

    # merge CS with DS with distance smaller than the threshold
    if rnd == 5:
        for clus_cs, center_cs in centroid_cs.items():
            cs_ds_dist = list()
            # loop through each DS cluster to find the nearest and merge CS into this DS
            for clus_ds, center_ds in centroid_ds.items():
                distance = mahalanobis_dist(center_cs, center_ds, sd_ds[center_ds])
                cs_ds_dist.append(((clus_cs, clus_ds), distance))
            sort_cs_ds_dist = sorted(cs_ds_dist, key=lambda x: x[1])
            if sort_cs_ds_dist[0][1] < threshold:
                cluster_cs = sort_cs_ds_dist[0][0][0]
                cluster_ds = sort_cs_ds_dist[0][0][1]
                merge_clus_lst = cluster_data_cs[cluster_cs] + cluster_data_ds[cluster_ds]
                merge_idx_lst = idx_ds[cluster_ds] + idx_cs[cluster_cs]
                cluster_data_ds[cluster_ds] = merge_clus_lst
                idx_ds[cluster_ds] = merge_idx_lst
                # delete CS cluster
                del cluster_data_cs[cluster_cs]
                del idx_cs[cluster_cs]
                del centroid_cs[cluster_cs]
                del sd_cs[cluster_cs]
                del CS_summary[cluster_cs]
                # recalculate the summary for the new DS
                summary = dict()
                summary[cluster_ds] = dict()
                summary[cluster_ds]['N'] = list()
                summary[cluster_ds]['SUM'] = [0] * num_dim
                summary[cluster_ds]['SUMSQ'] = [0] * num_dim

                summary[cluster_ds]['N'].extend(merge_clus_lst)
                for pnt in cluster_data_ds[cluster_ds]:
                    DS_summary[cluster_ds]['SUM'] = [sum(i) for i in zip(DS_summary[cluster_ds]['SUM'], pnt)]
                    for j in range(num_dim):
                        DS_summary[cluster_ds]['SUMSQ'][j] += pnt[j] ** 2

                value_N = len(DS_summary[cluster_ds]['N'])
                centroid_ds[cluster_ds] = [feature_sum / value_N for feature_sum in DS_summary[cluster_ds]['SUM']]
                sd_ds[cluster_ds] = get_stand_devi(DS_summary[cluster_ds])

    # write the intermediate result
    cnt_ds = len(DS)
    cnt_cs_cluster = len(idx_cs)
    cnt_cs = len(CS)
    cnt_rs = len(RS)
    print_lst = [cnt_ds, cnt_cs_cluster, cnt_cs, cnt_rs]
    f.write('round {}: '.format(rnd) + ','.join(str(n) for n in print_lst) + '\n')

# output the clustering result
f.write('\n')
result = list()
for clus_label, idx_lst in idx_ds:
    for idx in idx_lst:
        result.append((idx, clus_label))
for clus_label, idx_lst in idx_cs:
    for idx in idx_lst:
        result.append((idx, -1))
for pnt in RS:
    result.append((feature_index_dict[tuple(pnt)], -1))
f.write('The clustering results:' + '\n')
for row in result:
    f.write(str(row[0]) + str(row[1]) + '\n')
f.close()