# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re


class confidence:
    def __init__(self, T, N=2**(-100)):
        # set list of user_id and movie_id.
        self.user_ids_list  = T['user_id'].unique()
        self.movie_ids_list = T['movie_id'].unique()
        
        # set the number of samples of user and item.
        self.user_d = {user_id : (T['user_id']==user_id).sum() for user_id in self.user_ids_list}
        self.movie_d = {movie_id : (T['movie_id']==movie_id).sum() for movie_id in self.movie_ids_list}

        # set the number of samples of each attribute of user and item.
        self.ds = {col : T[col].sum() for col in T.columns if re.match(r'age_|sex_|occupation_|genre', col)}

        # set the attributes of each user and item.
        attr = lambda user_id : [col for col in T.columns[T.loc[T['user_id']==user_id, :].iloc[0,:]>0] if re.match('age_|sex_|occupation_', col)]
        self.user_attributes = {user_id: attr(user_id) for user_id in T['user_id'].unique()}
        attr = lambda movie_id : [col for col in T.columns[T.loc[T['movie_id']==movie_id, :].iloc[0,:]>0] if re.match('genre', col)]
        self.movie_attributes = {movie_id: attr(movie_id) for movie_id in T['movie_id'].unique()}
        
        # set total products of user related number.
        total_product = lambda user_id : np.prod([self.user_d[user_id]] + [self.ds[attribute] for attribute in self.user_attributes[user_id]])
        self.user_total_product = {user_id : total_product(user_id) for user_id in self.user_ids_list}
        total_product = lambda movie_id : np.prod([self.movie_d[movie_id]] + [self.ds[attribute] for attribute in self.movie_attributes[movie_id]])
        self.item_total_product = {movie_id : total_product(movie_id) for movie_id in self.movie_ids_list}
        
        # set N (is just scaler).
        self.N = N

    def get(self, user_id, movie_id):
        """
        This is based on the equation (10).

        Belows code is too slow to run at set of user_id and movie_id.
        But, it is more readable. When we use on set of user_id, movie_id, use self.get_from_dataframe.
        """
        # the sample number of the user_id and movie_id.
        d_user = self.user_d[user_id]
        d_movie = self.movie_d[movie_id]

        # the sample number of each item's attribute.
        d_user_attributes  = [self.ds[attribute] for attribute in self.user_attributes[user_id]]
        d_movie_attributes = [self.ds[attribute] for attribute in self.movie_attributes[movie_id]]

        # set values in list to product (self.N is just scaler.)
        prod_list = [self.N, d_user, d_movie] + d_user_attributes + d_movie_attributes

        return np.prod(prod_list)
    
    def get_from_dataframe(T):
        """
         An argument T must be a DataFrame which has 
        'user_id' and 'movie_id' in columns of T.
        """
        pass

if __name__ == '__main__':
    from src import config
    from src.sub_module.get_merged import get_merged
    
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv(config.PATH_rating, sep='\t', names=r_cols,
                          encoding='latin-1')
    ratings = ratings.loc[:5, :]

    ratings  = get_merged(ratings)

    C = confidence(ratings)
    C.get(196, 242)

    C.user_d
    C.movie_d
    C.movie_attributes
    C.user_attributes
    C.ds