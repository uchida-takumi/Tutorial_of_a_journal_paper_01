# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

############################
# Read DataSet
# pass in column names for each CSV
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
                      encoding='latin-1')

# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')


############################
# Preprocessing
train_ratings = ratings.loc[:90000, :]
test_ratings  = ratings.loc[90000:, :]


############################
# Step1: Construct multiple regressors
from sklearn.linear_model.ridge import Ridge
#from sklearn.ensemble import GradientBoostingRegressor




############################
# Step2: Co-Training


############################
# Step3: Assembling the results

