from pyspark import SparkContext
import sys
import time
import json
import numpy as np
import xgboost as xgb

start = time.time()
sc = SparkContext('local[*]', 'task2_2')
in_folder = 'data/'
in_file = 'data/yelp_val.csv'
out_file = 'output.csv'

train_file = sc.textFile(in_folder + 'yelp_train.csv')
test_file = sc.textFile(in_file)
user_file = sc.textFile(in_folder + 'user.json')
bus_file = sc.textFile(in_folder + 'business.json')

train_rdd = train_file.map(lambda x: x.split(',')).filter(lambda x: 'user_id' not in x)
test_rdd = test_file.map(lambda x: x.split(',')).filter(lambda x: 'user_id' not in x)
user_rdd = user_file.map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], (x['review_count'], x['average_stars'])))
bus_rdd = bus_file.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (x['review_count'], x['stars'])))

user_dict = user_rdd.collectAsMap()
bus_dict = bus_rdd.collectAsMap()
# get training data
xtrain = np.array(train_rdd.map(lambda x: user_dict[x[0]]).collect() +
                  train_rdd.map(lambda x: bus_dict[x[1]]).collect())
ytrain = np.array(list(train_rdd.map(lambda x: x[2]).collect()))
# get test data
xtest = np.array(test_rdd.map(lambda x: user_dict[x[0]]).collect() +
                 test_rdd.map(lambda x: bus_dict[x[1]]).collect())
# create the model
model = xgb.XGBRegressor()
# fit the model
model.fit(xtrain, ytrain)
# predict
ypred = model.predict(xtest)
test = test_rdd.map(lambda x: (x[0], x[1])).collect()

with open(out_file, 'w') as f:
    f.write('user_id, business_id, prediction\n')
    for i in range(len(test)):
        f.write(test[i][0] + test[i][1] + str(ypred[i]) + '\n')
f.close()
