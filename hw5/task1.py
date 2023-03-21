import sys
import time
from blackbox import BlackBox
import binascii
import random

'''
number of user id = 1518169  --> 1518191
'''

def myhashs(id):
    converted_id = int(binascii.hexlify(id.encode('utf8')), 16)
    results = list()
    for f in hash_function_list:
        value = ((f[0] * converted_id + f[1]) % 1518191) % m
        results.append(value)
    return results

def create_hash(num):
    hashes = list()
    for i in range(num):
        hashes.append([random.randint(1, 9999999999), random.randint(1, 9999999999)])
    return hashes

in_file = 'users.txt'
stream_size = 100
num_ask = 30
out_file = 'output.csv'

bit_array = [0 for i in range(69997)]
m = 69997
previous = list()
fp_rate = list()

bx = BlackBox()
# create hash functions
hash_function_list = create_hash(20)
for _ in range(num_ask):
    # calculate fpr for each stream
    fp = 0
    tn = 0
    stream_users = bx.ask(in_file, stream_size)
    while len(stream_users) != len(set(stream_users)):
        stream_users = bx.ask(in_file, stream_size)
    # loop through the object in the streaming set
    for s in stream_users:
        # hash the id using all hash functions and return a list
        hashed_values = myhashs(s)
        count = 0
        for v in hashed_values:
            # check if number at index v in the bit array
            if bit_array[v] == 0:
                bit_array[v] = 1
                tn += 1
            else:
                count += 1
        # if all index position are 1, check if it is false positive
        if count == len(hashed_values):
            if s not in previous:
                fp += 1
        # store the unhashed user id
        previous.append(s)
    fp_rate.append(fp/(fp+tn))

with open(out_file, 'w') as f:
    f.write('Time,FPR' + '\n')
    cnt = 0
    for i in fp_rate:
        f.write(str(cnt) + ',' + str(i))
        cnt += 1
f.close()

