# -*- coding: utf-8 -*-
"""
I refferd following url to study SGD:
    https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/
"""

import numpy as np
import pandas as pd
import random
import json
from datetime import datetime

class User_Item_Bias_and_MFCF_SGD:
    
    def __init__(self, user_item_ratings, n_latent_factors=3, is_global_bias=True, user_attribute_vectors=None, item_attribute_vectors=None):
        """
        INPUTS:
            user_item_ratings: multiple arrays like as below.
                [
                    ['user01', 'item93', 2],
                    ['user01', 'item23', 5],
                    ...
                ]
            user_attribute_vectors: dictionary whose key is user_id, value is array like.
                {
                    'user01': [0, 1, 1, 0],
                    'user02': [1, 0, 0, 1],
                    ...
                }
            item_attribute_vectors: dictionary whose key is item_id, value is array like.
                {
                    'item01': [1, 1, 1, 0, 0, 0],
                    'item02': [1, 0, 0, 0, 0, 1],
                    ...
                }             
        NOTE:
            本当は、user_idとitem_idの属性に関係のない、そのIDのbiasを用意しないといけない。
            まぁ、user_attribute_vectorsに導入してもらえれば問題はない。（メモリに負荷をかけるダミー行列になるが）
        """
        
        # --- SET INITIALIZE VALUES ---
        ## Set user_id, item_id, rating as pandas.DataFrame
        columns = ['user_id', 'item_id', 'rating']
        self.user_item_ratings_df = pd.DataFrame(user_item_ratings, columns=columns)
        
        ## Unique set of user_ids and item_ids
        self.user_ids = set([row[0] for row in user_item_ratings])
        self.item_ids = set([row[1] for row in user_item_ratings])
        
        ## Set the factor values of each ids randomly
        self.n_latent_factors = n_latent_factors
        self.user_latent_factors = {id_ : np.random.uniform(size=n_latent_factors) for id_ in self.user_ids}
        self.item_latent_factors = {id_ : np.random.uniform(size=n_latent_factors) for id_ in self.item_ids}
        
        ## Set bias weights of each attributes vectors
        ## If user_attribute_vectors or item_attribute_vectors is None, Set zero vectors.
        self.user_attribute_vectors = user_attribute_vectors
        self.item_attribute_vectors = item_attribute_vectors
        if user_attribute_vectors:
            vec_length = len(list(self.user_attribute_vectors.values())[0])
            self.user_bias_weights =  np.random.uniform(size=vec_length)
        else:
            self.user_bias_weights = None

        if item_attribute_vectors:
            vec_length = len(list(self.item_attribute_vectors.values())[0])
            self.item_bias_weights =  np.random.uniform(size=vec_length)
        else:
            self.item_bias_weights = None
        
        ## Set global bias weight
        self.is_global_bias = is_global_bias
        if is_global_bias:
            self.global_bias_weight = np.mean([row[2] for row in user_item_ratings])
                
        ## Set lossHistory
        self.lossHistory = []

        # Set Debug flg to output log file
        self.DEBUG = True
        if self.DEBUG:
            self.log_file_path = str(datetime.now())+'.log'
        
        
    def fit(self, epochs=10, learning_rate=0.01, regularize_rate=0.1):
        """
        Inputs:
            epochs: the number of learning.
            learning_rate : learning rate.
            regularize_rate: just regularize rate.
        
        Results of fit:
            self.user_latent_factors  : the learned factors of users. {user_id : np.array([factor_1, fctor_2, ...]),}
            self.item_latent_factors  : the learned factors of items. {item_id : np.array([factor_1, fctor_2, ...]),}
            self.user_bias_weights    : the weights of user attributes.
            self.item_bias_weights    : the weights of item attributes.
            self.global_bias_weight   : the weights of global bias.
            
            
        After fit, self.predict(user_ids, item_ids) is available
        """
                    
        # Initialize        
        self.fit_params = {
                'epochs':epochs, 
                'learning_rate':learning_rate, 
                'regularize_rate':regularize_rate, 
                'n_latent_factors':self.n_latent_factors
                }
        
        # Define lambda func to get error
        get_error = lambda user_id, item_id, rating : self.predict(user_id, item_id) - rating

        # loop over the desired number of epochs
        for epoch in range(epochs):
            # update some weights and latent factors for each epochs using SGD            
            mean_error = np.mean([get_error(user_id, item_id, rating) for user_id, item_id, rating in self.user_item_ratings_df.values])
            self._gradient_update_global_bias(mean_error)
            
            """
            # update item bias weights
            if self.item_attribute_vectors:
                n_attribute = len(self.item_bias_weights)
                for attribute_index in range(n_attribute):
                    # update _bias_weights respectively
                    target_ids = [_id for _id, vec in self.item_attribute_vectors.items() if vec[attribute_index]!=0]
                    target_df = self.user_item_ratings_df.loc[self.user_item_ratings_df['item_id'].isin(target_ids), :]
                    if target_df.shape[0] == 0:
                        continue
                    mean_error = np.mean([get_error(user_id, item_id, rating) for user_id, item_id, rating in target_df.values])
                    self._gradient_update_item_attribute_bias(mean_error, attribute_index)
            """    
            # update user bias weights
            if self.user_attribute_vectors:
                n_attribute = len(self.user_bias_weights)
                for attribute_index in range(n_attribute):
                    # update _bias_weights respectively
                    target_ids = [_id for _id, vec in self.user_attribute_vectors.items() if vec[attribute_index]!=0]
                    target_df = self.user_item_ratings_df.loc[self.user_item_ratings_df['user_id'].isin(target_ids), :]
                    if target_df.shape[0] == 0:
                        continue
                    mean_error = np.mean([get_error(user_id, item_id, rating) for user_id, item_id, rating in target_df.values])
                    if np.isnan(mean_error):
                        continue
                    self._gradient_update_user_attribute_bias(mean_error, attribute_index)
                
            # update latent factors of user_id and item_id
            for factor_index in range(self.n_latent_factors):
                for user_id, item_id, rating in self.user_item_ratings_df.values:
                    error = get_error(user_id, item_id, rating)
                    if random.random() < 0.5:
                        self._gradient_update_item_latent_factor(error, user_id, item_id, factor_index)
                        self._gradient_update_user_latent_factor(error, user_id, item_id, factor_index)                             
                    else:
                        self._gradient_update_user_latent_factor(error, user_id, item_id, factor_index)                             
                        self._gradient_update_item_latent_factor(error, user_id, item_id, factor_index)
            
            # DEBUG.log output
            if self.DEBUG:
                self.__DEBUG_log(file_path=self.log_file_path, comment='epock%09d'%epoch)
            
            # Append lossHistory
            mean_abs_error = np.mean(np.abs(
                    [get_error(user_id, item_id, rating) for user_id, item_id, rating in self.user_item_ratings_df.values]
                    ))
            self.lossHistory.append(mean_abs_error)
            
            

    def predict(self, user_id, item_id):
        """
        return predicted rating of the interaction between user_id and item_id 
        """
        # global bias
        if self.is_global_bias:
            global_bias = self.global_bias_weight
        else:
            global_bias = 0
        
        # latent factor effection between user_id and item_id
        latent_factor_result = (self.user_latent_factors[user_id] * self.item_latent_factors[item_id]).sum()
            
        # user attributes bias
        if self.user_attribute_vectors:
            user_attribute_bias = (self.user_attribute_vectors[user_id] * self.user_bias_weights).sum()
        else:
            user_attribute_bias = 0

        # item attributes bias
        if self.item_attribute_vectors:
            item_attribute_bias = (self.item_attribute_vectors[item_id] * self.item_bias_weights).sum()
        else:
            item_attribute_bias = 0
        
        return global_bias + latent_factor_result + user_attribute_bias + item_attribute_bias
    
    def _gradient_update_global_bias(self, error):
        # Adopt Squre error with 2-norm regulization as loss function.
        the_partial_differential_of_error      = 2 * error
        the_partial_differential_of_regularize = 2 * self.fit_params['regularize_rate'] * self.global_bias_weight
        the_partial_differential = the_partial_differential_of_error + the_partial_differential_of_regularize
        self.global_bias_weight += (-1) * self.fit_params['learning_rate'] * the_partial_differential

    def _gradient_update_user_latent_factor(self, error, user_id, item_id, factor_index):
        # Adopt Squre error with 2-norm regulization as loss function.
        the_partial_differential_of_error      = 2 * error * self.item_latent_factors[item_id][factor_index]
        the_partial_differential_of_regularize = 2 * self.user_latent_factors[user_id][factor_index]
        the_partial_differential = the_partial_differential_of_error + the_partial_differential_of_regularize
        self.user_latent_factors[user_id][factor_index] += (-1) * self.fit_params['learning_rate'] * the_partial_differential

    def _gradient_update_item_latent_factor(self, error, user_id, item_id, factor_index):
        # Adopt Squre error with 2-norm regulization as loss function.
        the_partial_differential_of_error      = 2 * error * self.user_latent_factors[user_id][factor_index]
        the_partial_differential_of_regularize = 2 * self.item_latent_factors[item_id][factor_index]
        the_partial_differential = the_partial_differential_of_error + the_partial_differential_of_regularize
        self.item_latent_factors[item_id][factor_index] += (-1) * self.fit_params['learning_rate'] * the_partial_differential

    def _gradient_update_user_attribute_bias(self, error, attribute_index):
        # Adopt Squre error with 2-norm regulization as loss function.
        the_partial_differential_of_error      = 2 * error
        the_partial_differential_of_regularize = 2 * self.fit_params['regularize_rate'] * self.user_bias_weights[attribute_index]
        the_partial_differential = the_partial_differential_of_error + the_partial_differential_of_regularize
        self.user_bias_weights[attribute_index] += (-1) * self.fit_params['learning_rate'] * the_partial_differential

    def _gradient_update_item_attribute_bias(self, error, attribute_index):
        # Adopt Squre error with 2-norm regulization as loss function.
        the_partial_differential_of_error      = 2 * error
        the_partial_differential_of_regularize = 2 * self.fit_params['regularize_rate'] * self.item_bias_weights[attribute_index]
        the_partial_differential = the_partial_differential_of_error + the_partial_differential_of_regularize
        self.item_bias_weights[attribute_index] += (-1) * self.fit_params['learning_rate'] * the_partial_differential
    
    def __DEBUG_log(self, file_path, comment=''):        
        with open(file_path, 'a') as f:
            f.write('===START===\n')            
            log_dict = {
                    'timestamp' : str(datetime.now()),
                    'comment' : comment,
                    'self.user_latent_factors': str(self.user_latent_factors),
                    'self.item_latent_factors': str(self.item_latent_factors),
                    'self.user_bias_weights': str(self.user_bias_weights),
                    'self.item_bias_weights': str(self.item_bias_weights),
                    'self.global_bias_weight': str(self.global_bias_weight)
                    }
            f.write(json.dumps(log_dict, indent=True))
            f.write('===END===\n')


if __name__ == '__main__':
    # IMPORT
    import numpy as np
    import random
    from src.sub_module.User_Item_Bias_and_MFCF_SGD import User_Item_Bias_and_MFCF_SGD
    
    # INPUT
    n_sample = 50
    n_factors = 3
    
    user_item_ = [['u%02d'%d1, 'i%02d'%d2] for d1,d2 in np.random.randint(1,10,size=n_sample*2).reshape(-1,2)]
    
    user_ids = set([row[0] for row in user_item_])
    item_ids = set([row[1] for row in user_item_])

    answer_UserFactor = {user:np.random.rand(n_factors)*4 for user in set(user_ids)}
    answer_ItemFactor = {item:np.random.rand(n_factors)*4 for item in set(item_ids)}
    
    get_rating = lambda u,i: (answer_UserFactor[u]*answer_ItemFactor[i]).sum()
    user_item_ratings  = [ [u, i, get_rating(u, i)] for u,i in user_item_] 
    
    user_attribute_vectors = {u: random.choices([0,1], k=4) for u in user_ids}
    item_attribute_vectors = {i: random.choices([0,1], k=5) for i in item_ids}
    
    UIBMS = User_Item_Bias_and_MFCF_SGD(user_item_ratings,
                                      user_attribute_vectors=user_attribute_vectors,
                                      item_attribute_vectors=item_attribute_vectors)    
    UIBMS.fit()    
    predict_result = UIBMS.predict(user_id='u01', item_id='i02')
    
    UIBMS.lossHistory