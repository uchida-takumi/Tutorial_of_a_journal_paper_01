# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from src import config


def get_attributes():
    # pass in column names for each CSV
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv(config.PATH_user, sep='|', names=u_cols,
                        encoding='latin-1')
    users = users.loc[:, ['user_id', 'age', 'sex', 'occupation']]
    
    # Get dummies_df
    ## change age columns to dummy of bins.    
    bins = range(0, 200, 10)
    users['age']  = ["%02d-%02d"%(bins[i-1], bins[i]) for i in np.digitize(users['age'], bins)]
    cols_to_dummy = ['age', 'sex', 'occupation']
    dummy_users   =  pd.get_dummies(users[cols_to_dummy], drop_first=True)
    user_attribute_vectors = {users.loc[i, 'user_id']: list(dummy_users.values[i, :]) for i in range(users.shape[0])}
    
    # the movies file contains columns indicating the movie's genres
    # let's only load the first five columns of the file with usecols
    m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
    genre_cols = ['genre%02d'%i for i in range(19)]
    movies = pd.read_csv(config.PATH_item, sep='|', 
                         names=m_cols+genre_cols, 
                         encoding='latin-1')
    movies = movies.loc[:, ['movie_id']+genre_cols]
    item_attribute_vectors = {val[0]: list(val[1:]) for val in movies.values}

    return user_attribute_vectors, item_attribute_vectors
