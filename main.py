# -*- coding: utf-8 -*-
"""
Author: Andrew Floyd
Course: CS3001 - Intro to Data Science
Dr. Fu
November 2nd, 2018
"""

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import NMF
from surprise import KNNBasic
from surprise import evaluate, print_perf
import os
import pandas as pd

#Loading data from a file into a dataset
file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp',sep='\t')
data = Dataset.load_from_file(file_path=file_path, reader=reader)

#splitting the data for 3-folds cross-validation
data.split(n_folds=3)

#SVD Algorithm with MAE and RMSE output
algo = SVD()
perf = evaluate(algo, data, measures=['RMSE','MAE'])
print("SVD Algorithm:")
print()
print_perf(perf)
print()

#PMF Algorithm with MAE and RMSE output (PMF is the same as unbiased SVD)
algo = SVD(biased=False)
perf = evaluate(algo, data, measures=['RMSE','MAE'])
print("PMF Algorithm:")
print()
print_perf(perf)
print()

#NMF Algorithm with MAE and RMSE output
algo = NMF()
perf = evaluate(algo, data, measures=['RMSE','MAE'])
print("NMF Algorithm: ")
print()
print_perf(perf)
print()

#User-Based Collaborative Filtering Algorithm with MAE and RMSE output
algo = KNNBasic(sim_options={'user_based':True})
perf = evaluate(algo, data, measures=['RMSE','MAE'])
print("User-Based Collaborative Filtering Algorithm: ")
print()
print_perf(perf)
print()

#Item-Based Collaborative Filtering Algorithm with MAE and RMSE output
algo = KNNBasic(sim_options={'user_based':False})
perf = evaluate(algo, data, measures=['RMSE','MAE'])
print("Item-Based Collaborative Filtering Algorithm: ")
print()
print_perf(perf)
print()

#User-Based Collaborative Filtering Algorithm with Cosine, MSD and Pearson Similarities
algo = KNNBasic(sim_options={'name':'MSD','user_based':True})
perf = evaluate(algo, data, measures=['RMSE','MAE'])
print("User-Based Collaborative Filtering Algorithm (w/ MSD): ")
print()
print_perf(perf)
print()

algo = KNNBasic(sim_options={'name':'cosine','user_based':True})
perf = evaluate(algo, data, measures=['RMSE','MAE'])
print("User-Based Collaborative Filtering Algorithm (w/ cosine): ")
print()
print_perf(perf)
print()

algo = KNNBasic(sim_options={'name':'pearson','user_based':True})
perf = evaluate(algo, data, measures=['RMSE','MAE'])
print("User-Based Collaborative Filtering Algorithm (w/ Pearson): ")
print()
print_perf(perf)
print()

#Item-Based Collaborative Filtering Algorithm with Cosine, MSD and Pearson Similarities
algo = KNNBasic(sim_options={'name':'MSD','user_based':False})
perf = evaluate(algo, data, measures=['RMSE','MAE'])
print("Item-Based Collaborative Filtering Algorithm (w/ MSD): ")
print()
print_perf(perf)
print()

algo = KNNBasic(sim_options={'name':'cosine','user_based':False})
perf = evaluate(algo, data, measures=['RMSE','MAE'])
print("Item-Based Collaborative Filtering Algorithm (w/ cosine): ")
print()
print_perf(perf)
print()

algo = KNNBasic(sim_options={'name':'pearson','user_based':False})
perf = evaluate(algo, data, measures=['RMSE','MAE'])
print("Item-Based Collaborative Filtering Algorithm (w/ Pearson): ")
print()
print_perf(perf)
print()

#Plotting Results
c = {'filtering_type': ['user(msd)','user(cosine)','user(pearson)','item(msd)','item(cosine)','item(pearson)'],
     'test': ['ok','ok','ok','no','no','no'],
      'rmse_mean': [0.9881,1.0213,1.0206,0.9860,1.0360,1.0495]}
cf = pd.DataFrame(data=c)

plotc = cf.plot.bar(x='filtering_type', y='rmse_mean', legend=False)
plotc.set_title('User Based and Item Based Collaborative w/ MSD, Cosine and Pearson Similarities', size=13)
plotc.set_ylabel('Mean RMSE Value', size=12)
plotc.set_xlabel('Collaborative Filtering Type', size=12)
plotc.legend(False)


#User-Based Collaborative Filtering Algorithm with varying K values
for i in range(0,20):
    algo = KNNBasic(k=i, sim_options={'name':'MSD','user_based':True})
    perf = evaluate(algo, data, measures=['RMSE'])
    print("User-Based Collaborative Filtering Algorithm (k=", i)
    print()
    print_perf(perf)
    print()

d = {'k': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
     'mean': [1.3278,1.1489,1.087,1.0562,1.037,1.0239,1.0151,1.0082,1.0033,0.9996,0.9964,0.9939,0.9919,0.9906,0.9896,0.9888,0.9881,0.9876,0.9872]}

df = pd.DataFrame(data=d)

plot1 = df.plot.scatter(x='k', y='mean', c='DarkBlue')
plot1.set_title('User Based Collaborative Filtering K Values')
plot1.set_ylabel('Mean RMSE Value')
plot1.set_xlabel('K')

for i in range(0,20):
    algo = KNNBasic(k=i, sim_options={'name':'MSD','user_based':False})
    perf = evaluate(algo, data, measures=['RMSE'])
    print("Item-Based Collaborative Filtering Algorithm (k=", i,")")
    print()
    print_perf(perf)
    print()
    
d1 = {'k': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
     'mean': [1.4289,1.2317,1.1527,1.1094,1.0815,1.0626,1.0486,1.0374,1.0290,1.0225,1.0166,1.0122,1.0085,1.0050,1.0022,1.0003,0.9983,0.9969,0.9953]}
df1 = pd.DataFrame(data=d1)

plot2 = df.plot.scatter(x='k', y='mean', c='Green')
plot2.set_title('Item Based Collaborative Filtering K Values')
plot2.set_ylabel('Mean RMSE Value')
plot2.set_xlabel('K')

algo = KNNBasic(k=55, sim_options={'name':'MSD','user_based':True})
perf = evaluate(algo, data, measures=['RMSE'])
print("User-Based Collaborative Filtering Algorithm (k=", 55,")")
print()
print_perf(perf)
print()
