from blackbox import BlackBox
import sys
import random


if __name__ == '__main__':
    in_file = 'users.txt'
    stream_size = 100
    num_ask = 30
    out_file = 'output3.csv'
    random.seed(553)
    n = 100
    s = 100
    reservoir = list()
    result = list()
    bx = BlackBox()
    stream_users = bx.ask(in_file, stream_size)
    for user in stream_users:
        reservoir.append(user)
    result.append([1 * stream_size, reservoir[20], reservoir[40], reservoir[60], reservoir[80]])

    for ask in range(1, num_ask):
        stream_users = bx.ask(in_file, stream_size)
        for user in stream_users:
            n = n+1
            if random.random() < s/n:
                idx = random.randint(0,99)
                reservoir[idx] = user
            else:
                continue
        result.append([(ask+1) * stream_size, reservoir[20], reservoir[40], reservoir[60], reservoir[80]])

    with open(out_file, 'w') as f:
        f.write('seqnum,0_id,20_id,40_id,60_id,80_id\n')
        for i in result:
            f.write(str(i[0]) + ',' + i[1] + ',' + i[2] + ',' + i[3] + ',' + i[4] + '\n' )
    f.close()
