# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import re
from itertools import product

from src import config
from src.sub_module.get_attributes_merged import get_attributes_merged

############################
# Read DataSet
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(config.PATH_rating, sep='\t', names=r_cols,
                      encoding='latin-1')

# For test filtering
g_user_id = ratings.groupby(by='user_id')['user_id'].count()
main_user_id = g_user_id[g_user_id>10].index
main_user_id = np.random.choice(main_user_id, size=50, replace=False)

g_movie_id = ratings.groupby(by='movie_id')['movie_id'].count()
main_movie_id = g_movie_id[g_movie_id>10].index
main_movie_id = np.random.choice(main_movie_id, size=50, replace=False)

ratings = ratings.loc[ratings.user_id.isin(main_user_id)&ratings.movie_id.isin(main_movie_id), :]


############################
# Define equations as functions (10)
from src.sub_module.confidence import confidence

############################
# Input: 
"""
 the training set L with labeled (rated) examples, 
and the unlabeled (not rated) example set U;
"""
columns = ['user_id', 'movie_id', 'rating']
L = ratings.loc[:, columns]
## seprate L into train and test.
test_size = int(L.shape[0]*0.2)
L_test  = L.iloc[:test_size, :]
L_train = L.iloc[test_size:, :]

L_test = L_test.loc[L_test['user_id'].isin(L_train['user_id'].unique()) & L_test['movie_id'].isin(L_train['movie_id'].unique()), :]

U = pd.DataFrame([[user_id, movie_id, None] for user_id, movie_id in product(L_train['user_id'].unique(), L_train['movie_id'].unique())], columns=columns)


############################
# Step1: Construct multiple regressors
"""
Generate the two regressors h00 and h01 by mainpulating the training set or manipulating the attributes;
Create a pool U_ by randomly picking examples from U;
Initialize the teaching sets T00, T01;
"""
U_size = min(U.shape[0], L.shape[0]*10)
rand_index = np.random.choice(U.index, size=U_size, replace=False)
U_ = U.loc[rand_index, :]

# merge attributes of item and user
from src.sub_module.get_attributes import get_attributes
user_attribute_vectors, item_attribute_vectors = get_attributes()


# build regressor
from src.sub_module.User_Item_Bias_and_MFCF_SGD import User_Item_Bias_and_MFCF_SGD

T01 = L_train.copy()
T02 = L_train.copy()


############################
# Step2: Co-Training

rounds = 5
N = int(U_.shape[0] * 1/(rounds*2))

dict_T = {'T01': T01, 'T02': T02}

for i in range(rounds):
    # --- Train Regressor of co-training ---
    print("round {}".format(str(i)))
    for key, T0_ in dict_T.items():
        #T0_ = [T01, T02][0]
        if key == 'T01':
            h = User_Item_Bias_and_MFCF_SGD(np.array(T0_)[:, :3],
                                              user_attribute_vectors=user_attribute_vectors,
                                              item_attribute_vectors=None)
            h.fit(epochs=500, learning_rate=0.01, regularize_rate=0.1)
        elif key == 'T02':
            h = User_Item_Bias_and_MFCF_SGD(np.array(T0_)[:, :3],
                                              user_attribute_vectors=None,
                                              item_attribute_vectors=item_attribute_vectors)
            h.fit(epochs=500, learning_rate=0.01, regularize_rate=0.1)            

        # --- GET confidencees on each sample of U
        U_attributes_merged = get_attributes_merged(U_)    
        C = confidence(U_attributes_merged)
        confideces = [C.get(row['user_id'], row['movie_id']) for idx, row in U_attributes_merged.iterrows()]
        confideces = confideces / sum(confideces)
        
        # --- add other T0* rating estimated samples from U_
        picked_idx = np.random.choice(U_.index, size=N, p=confideces, replace=False)
        picked_U_ = U_.loc[picked_idx, :]
        U_ = U_.loc[~U_.index.isin(picked_idx), :]
        picked_U_['rating'] = [h.predict(row[0], row[1]) for row in picked_U_.values]
        
        other_key = list(dict_T.keys())[(list(dict_T.keys()).index(key)+1)%len(dict_T.keys())]
        dict_T[other_key] = T0_.append(picked_U_)


T_0102 = pd.concat([df for key, df in dict_T.items()]).drop_duplicates(subset=['user_id', 'movie_id'])

############################
# Step3: Assembling the results

CSEL = User_Item_Bias_and_MFCF_SGD(np.array(T_0102)[:, :3],
                                  user_attribute_vectors=user_attribute_vectors,
                                  item_attribute_vectors=item_attribute_vectors)
CSEL.fit(epochs=500, learning_rate=0.01, regularize_rate=0.1)

FactCF = User_Item_Bias_and_MFCF_SGD(np.array(L_train)[:, :3],
                                  user_attribute_vectors=None,
                                  item_attribute_vectors=None)
FactCF.fit(epochs=500, learning_rate=0.01, regularize_rate=0.1)

CSEL_no_semi = User_Item_Bias_and_MFCF_SGD(np.array(L_train)[:, :3],
                                  user_attribute_vectors=user_attribute_vectors,
                                  item_attribute_vectors=item_attribute_vectors)
CSEL_no_semi.fit(epochs=500, learning_rate=0.01, regularize_rate=0.1)

L_test['CSEL']   = [CSEL.predict(user_id, item_id) for user_id, item_id in L_test.values[:, :2]]
L_test['FactCF'] = [FactCF.predict(user_id, item_id) for user_id, item_id in L_test.values[:, :2]]
L_test['CSEL_no_semi'] = [CSEL_no_semi.predict(user_id, item_id) for user_id, item_id in L_test.values[:, :2]]

L_test.to_csv('test_result.tsv', sep='\t', index=False)


