# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import re
from itertools import product

from src import config

############################
# Read DataSet
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(config.PATH_rating, sep='\t', names=r_cols,
                      encoding='latin-1')


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
U = pd.DataFrame([[user_id, movie_id, None] for user_id, movie_id in product(L['user_id'].unique(), L['movie_id'].unique())], columns=columns)


############################
# Step1: Construct multiple regressors
"""
Generate the two regressors h00 and h01 by mainpulating the training set or manipulating the attributes;
Create a pool U_ by randomly picking examples from U;
Initialize the teaching sets T00, T01;
"""
rand_index = np.random.choice(U.index, size=L.shape[0]*10, replace=False)
U_ = U.loc[rand_index, :]

# merge attributes of item and user
from src.sub_module.get_merged import get_merged
L  = get_merged(L)
U_ = get_merged(U_) 

## seprate L into train and test.
test_size = int(L.shape[0]*0.2)
L_test = L.loc[:L.size, :]


from src.sub_module import MFCF_StochasticGradientDescent
h00 = MFCF_StochasticGradientDescent.MFCF_StochasticGradientDescent()
h01 = MFCF_StochasticGradientDescent.MFCF_StochasticGradientDescent()

T00 = L.copy()
T01 = L.copy()

############################
# Step2: Co-Training

rounds = 100
N = int(U_.shape[0] * 1/rounds)

for i in range(rounds):
    C00 = confidence(T00)
    C01 = confidence(T01)
    
    """
    Obtain the confidence c* by Eq(10)
    """
    c00s = [C00.get(row['user_id'], row['movie_id']) for idx, row in U_.iterrows()]
    c01s = [C01.get(row['user_id'], row['movie_id']) for idx, row in U_.iterrows()]

    """
    Select N examples to form T by the Roulette algorithm.
    with the probability Pr(user_id, movie_id) given by Eq. 11:
        Tplus  = Roulette(Pr(user_id, movie_id));
        U_     = U_ - Tplus ;
    """
    
    
    
    for i in U_.index:
        # i = 1
        user_id, movie_id  = U_.loc[i, 'user_id'], U_.loc[i, 'movie_id']
        
        
        c00 = C00.get(user_id, movie_id)
        c01 = C01.get(user_id, movie_id)    
        
        """
        Select N examples to form T by the Roulette algorithm.
        with the probability Pr(user_id, movie_id) given by Eq. 11:
            Tplus  = Roulette(Pr(user_id, movie_id));
            U_     = U_ - Tplus ;
        """
        
        



############################
# Step3: Assembling the results

