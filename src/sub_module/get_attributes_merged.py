# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from src import config

# pass in column names for each CSV
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(config.PATH_user, sep='|', names=u_cols,
                    encoding='latin-1')
users = users.loc[:, ['user_id', 'age', 'sex', 'occupation']]


# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
genre_cols = ['genre%02d'%i for i in range(19)]
movies = pd.read_csv(config.PATH_item, sep='|', 
                     names=m_cols+genre_cols, 
                     encoding='latin-1')
movies = movies.loc[:, ['movie_id']+genre_cols]

def get_attributes_merged(ratings):
    ratings_ = ratings.loc[:, ['user_id', 'movie_id', 'rating']]
    ratings_ = ratings_.merge(users, how='inner', on='user_id')
    ratings_ = ratings_.merge(movies, how='inner', on='movie_id')
    ratings_ = ratings_.reset_index(drop=True)
    
    # Get info_df
    info_df = ratings_.loc[:, ['user_id', 'movie_id', 'rating']]

    # Get dummies_df
    ## change single vector to dummies matrix.    
    bins = range(0, 200, 10)
    ratings_.loc[:, 'age']  = ["%02d-%02d"%(bins[i-1], bins[i]) for i in np.digitize(ratings_['age'], bins)]
    maked_dummies_col = ['age', 'sex', 'occupation']
    maked_dummies_df =  pd.get_dummies(ratings_[maked_dummies_col], drop_first=True)
    
    dummies_col = [col for col in ratings_.columns if re.match(r'genre[0-9].*', col)]
    already_dummies_df = ratings_.loc[:, dummies_col]
    dummies_df = pd.concat([maked_dummies_df, already_dummies_df], axis=1)
    
    # concatenate info_df and dummies_df
    return pd.concat([info_df, dummies_df], axis=1)


if __name__ == '__main__':
    
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv(config.PATH_rating, sep='\t', names=r_cols,
                          encoding='latin-1')

    print( get_merged(ratings)[:5] )
    print( get_merged(ratings).shape )
    print( get_merged(ratings).columns )

