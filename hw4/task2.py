from pyspark import SparkContext
from collections import Counter, defaultdict
import operator
from operator import add
import random
import sys

sc = SparkContext('local[*]', 'task2')
sc.setLogLevel('WARN')

threshold = 7
in_file = 'ub_sample_data.csv'
out_file1 = 'myoutput1.txt'
out_file2 = 'myoutput2.txt'

# read the file
text_rdd = sc.textFile(in_file)
raw_rdd = text_rdd.map(lambda x: x.split(',')).filter(lambda x: 'user_id' not in x).map(lambda x: (x[1], x[0]))

# find the vertices and edges list (doesn't need both direction)
bus_userpair = raw_rdd.join(raw_rdd).filter(lambda x: x[1][0] != x[1][1])
userpair_bus_set = bus_userpair.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).\
                    filter(lambda x: len(x[1]) >= threshold)  # ((user1, user2), {bus1, bus2...})
# find all edges (user-pairs)
edges_rdd = userpair_bus_set.map(lambda x: (x[0][0], x[0][1]))
edges_lst = edges_rdd.collect()  # [(user1, user2), (user2, user1)...]
# find all vertices (users)
vertices_rdd = edges_rdd.flatMap(lambda x: x).distinct() # [user1, user2, user3]
vertices_lst = vertices_rdd.collect()
# create the graph dictionary {node: {adjacent1, adjacent2..}}
graph = edges_rdd.groupByKey().mapValues(set).collectAsMap()  # {user1, {adjacent_user1, adjacent_user2...}}


def calculate_bet(node_level, children, parent, path):
    betweeness = defaultdict(float)
    for j, v in node_level.items():
        # check the leaf node
        if not children[j]:
            for par in parent[j]:
                betweeness[tuple(sorted((j, par)))] = path[par]/path[j]
        else:
            for par in parent[j]:
                childpath = 0
                for child in children[j]:
                    childpath += betweeness[tuple(sorted((j, child)))]
                betweeness[tuple(sorted((j, par)))] = (1 + childpath) * path[par]/path[j]
    return betweeness


def GirvanNewman(adj_set, rootnode):
    visited = list()
    queue = list()
    parent = defaultdict(list)
    children = defaultdict(list)
    pathcnt = defaultdict(int)
    nodelevel = defaultdict(int)

    queue.append(rootnode)
    visited.append(rootnode)
    pathcnt[rootnode] = 1

    # step1: calculate the shortest path for each node using BFS
    while queue:
        node = queue.pop(0)
        for neighbor in adj_set[node]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.append(neighbor)
                parent[neighbor].append(node)
                children[node].append(neighbor)
                # because this neighbor is unvisited: it's on the next level, and the path is still 0
                pathcnt[neighbor] = pathcnt[node]
                nodelevel[neighbor] = nodelevel[node] + 1
            elif nodelevel[neighbor] == nodelevel[node] + 1:
                parent[neighbor].append(node)
                children[node].append(neighbor)
                pathcnt[neighbor] += pathcnt[node]

    # step2: sum up the shortest path started from the bottom
    sorted_node_level = dict(sorted(nodelevel.items(), key=operator.itemgetter(1), reverse=True))
    # [(node, level), (node, level)...]: sorted in descending order, leaf first root final
    result = calculate_bet(sorted_node_level, children, parent, pathcnt)
    return list(result.items())  # {(node, node), betweeness}


userpair_bet = vertices_rdd.map(lambda x: GirvanNewman(graph, x)).flatMap(lambda x: x)\
                .reduceByKey(add).map(lambda x: (x[0], round(x[1]/2, 5)))
sort_bet_lst = userpair_bet.sortBy(lambda x: -x[1]).collect()

with open(out_file1, 'w') as f1:
    for i in sort_bet_lst:
        f1.write(str(i[0]) + ',' + str(i[1]) + '\n')


# task 2.2
def depthFirstSearch(nod, adj, visit, community):
    if nod not in visit:
        visit.add(nod)
        community.append(nod)
        for neighbor in adj[nod]:
            depthFirstSearch(neighbor, adj, visit, community)
    return community

# unchanged variables: A, m, kij
adj_set = graph
m = int(len(edges_lst)/2)
k = defaultdict(int)
for node in vertices_lst:
    k[node] = len(graph[node])
# the range of Q is -1 < Q < 1
best_q = -2
final_result = list()
cut = 0

while m > cut:
    # find the highest betweeness and cut it
    highest_bet = sort_bet_lst[0][1]
    for i in sort_bet_lst:
        if i[1] == highest_bet:
            adj_set[i[0][0]].remove(i[0][1])
            adj_set[i[0][1]].remove(i[0][0])
            cut += 1
            node_i = i[0][0]
            node_j = i[0][1]

    com_lst = list()
    vertices = vertices_rdd.collect()

    # find null communities model using dfs
    while vertices:
        comm = []
        visited = set()
        # randomly select a vertice in the remaining list to find a new community
        n = random.choice(vertices)
        comm = sorted(depthFirstSearch(n, adj_set, visited, comm))
        com_lst.append(comm)
        # remove all vertices already classified
        for i in comm:
            vertices.remove(i)

    # calculate modularity for each cutting choice
    mod = 0
    for community in com_lst:
        for i in community:
            for j in community:
                if j in graph[i]:
                    mod = mod + 1 - (k[i] * k[j] / (2 * m))
                else:
                    mod = mod - (k[i] * k[j] / (2 * m))
    q = mod / (2 * m)

    # compare the best Q with the Q
    if q > best_q:
        best_q = q
        final_result = com_lst

    # compute the new betweeneess
    userpair_bet = vertices_rdd.map(lambda x: GirvanNewman(adj_set, x)).flatMap(lambda x: x) \
        .reduceByKey(add).map(lambda x: (x[0], round(x[1] / 2, 5)))
    sort_bet_lst = userpair_bet.sortBy(lambda x: -x[1]).collect()

sorted_final_result = sorted(final_result, key=lambda x: (len(x), x))

with open(out_file2, 'w') as f2:
    for i in sorted_final_result:
        f2.write(str(i)[1:-1] + '\n')
