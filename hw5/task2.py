import sys
import time
from blackbox import BlackBox
import binascii
import random
import statistics


def myhashs(id):
    converted_id = int(binascii.hexlify(id.encode('utf8')), 16)
    results = list()
    for func in hash_function_list:
        value = ((func[0] * converted_id + func[1]) % 1518191) % 491
        results.append(value)
    return results

def create_hash(num):
    hashes = list()
    for i in range(num):
        hashes.append([random.randint(1, 9999999999), random.randint(1, 9999999999)])
    return hashes

in_file = 'users.txt'
stream_size = 300
num_ask = 30
out_file = 'output2.csv'

hash_maxtailing = [0 for i in range(19)]
# stores all the max number of tailing zeros from each hash function
bx = BlackBox()
# create hash functions
hash_function_list = create_hash(19)
result = list()
total = 0
for _ in range(num_ask):
    stream_users = bx.ask(in_file, stream_size)
    ground_truth = len(set(stream_users))
    for s in stream_users:
        hashed_values = myhashs(s)
        # convert the hashed values into binary
        for i in hashed_values:
            binary = bin(i)
            max_zero = list()
            # find the tailing zero of the binary number
            zero_lst = binary.split('1')
            tailing_str = zero_lst[-1]
            num_zeros = len(tailing_str)
            if hash_maxtailing[hashed_values.index(i)] < num_zeros:
                hash_maxtailing[hashed_values.index(i)] = num_zeros
    # find the average of MAX of each group (# 3)
    group_avg = list()
    for i in hash_maxtailing:
        if i <= len(hash_maxtailing) - 3:
            avg = (hash_maxtailing[i] + hash_maxtailing[i+1] + hash_maxtailing[i+2]) / 3
            group_avg.append(avg)
    # take the median among all averages
    med = statistics.median(group_avg)
    count = pow(2, med)
    result.append([ground_truth, count])
    total += round(count)
print(total/(300*30))


with open(out_file, 'w') as f:
    f.write('Time,Ground Truth,Estimation' + '\n')
    cnt = 0
    for i in result:
        f.write(str(cnt) + ',' + str(i[0]) + ',' + str(i[1]) + '\n')
        cnt += 1
f.close()

