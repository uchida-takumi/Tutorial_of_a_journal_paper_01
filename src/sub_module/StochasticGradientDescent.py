# -*- coding: utf-8 -*-
"""
This code is mostly form:
    https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/
"""

import numpy as np


class StochasticGradientDescent:
    
    def __init__(self):
        self.lossHistory = []
        
    def fit(self, user_ids, item_ids, ratings, n_factors=3, epochs=50, alpha=0.001):
        """
        Inputs:
            user_ids : [user01, user02, user03, ...]
            item_ids : [item42, user42, user59, ...]
            user_ids : [  2   ,    4  ,   3   , ...]
                user_ids, item_ids and user_ids must have same length.
        
        Results of fit:
            self.UserFactores : the learned factores of users. {user_id : np.array([factor_1, fctor_2, ...]),}
            self.ItemFactores : the learned factores of items. {item_id : np.array([factor_1, fctor_2, ...]),}
        
        After fit, self.predict(user_ids, item_ids) is available
        """
        
        # Check arguments
        if len(user_ids)==len(item_ids)==len(ratings):
            self.n_sample = len(user_ids)
        else:
            raise Exception("user_ids, item_ids and user_ids must have same length.")        
            
        # Initialize
        self.UserFactores = {id_ : np.random.uniform(size=n_factors) for id_ in set(user_ids)}
        self.ItemFactores = {id_ : np.random.uniform(size=n_factors) for id_ in set(item_ids)}
        
        self.params = {'n_factors':n_factors, 'epochs':epochs, 'alpha':alpha}
        
        # loop over the desired number of epochs
        for each in np.arange(0, epochs):
            # initialize the total loss for the epoch
            epochLoss = []
            
            # loop over our data in batches
            for (user_id, item_id, rating) in zip(user_ids, item_ids, ratings):
                
                # update self.UserFactores and self.ItemFactores
                for update_target in ['UserFactores', 'ItemFactores']:
                    # get predicted values (preds) useing temporaly self.UserFactores and self.ItemFactores.
                    pred = self._predict_of_single_id(user_id, item_id)

                    # get error of this batch.
                    error = pred - rating
                    
                    # given our 'error', we can compute the total loss value on
                    # the batch as the sum of squared loss
                    loss = np.sum(error ** 2)
                    epochLoss.append(loss)
                    
                    # the gradient update is therefore the dot product between
                    # the transpose of our current batch and the error on the # batch
                    gradient = self._get_gradient(user_id, item_id, error, update_target)
                    
                    # use the gradient computed on the current batch to take
                    # a "step" in the corrent direction
                    if update_target=='UserFactores':                        
                        self.UserFactores[user_id] += - alpha * gradient
                    elif update_target=='ItemFactores':
                        self.ItemFactores[item_id] += - alpha * gradient
            
            # update our loss history list by taking the average loss
            # across all batches
            self.lossHistory.append(np.average(epochLoss))
            
    def predict(self, user_ids, item_ids):
        return np.array([self._predict_of_single_id(user_id, item_id) for user_id, item_id in zip(user_ids, item_ids)])
    
    def _predict_of_single_id(self, user_id, item_id):
        return (self.UserFactores[user_id] * self.ItemFactores[item_id]).sum()
    
    def _get_gradient(self, user_id, item_id, error, update_target='UserFactores'):
        if update_target=='UserFactores':
            gradient = self.ItemFactores[item_id] * error
        elif update_target=='ItemFactores':
            gradient = self.UserFactores[user_id] * error
        return gradient


if __name__ == '__main__':
    # INPUT
    n_sample = 100
    n_factors = 3

    user_ids = ['u%02d'%d for d in np.random.randint(1,5,size=n_sample)]
    item_ids = ['i%02d'%d for d in np.random.randint(1,10,size=n_sample)]

    answer_UserFactor = {user:np.random.randint(1,11, size=n_factors) for user in set(user_ids)}
    answer_ItemFactor = {item:np.random.randint(1,11, size=n_factors) for item in set(item_ids)}

    ratings  = [(answer_UserFactor[u]*answer_ItemFactor[i]).sum() for u,i in zip(user_ids, item_ids)]  

    sgd = StochasticGradientDescent()    
    sgd.fit(user_ids, item_ids, ratings, n_factors=n_factors, epochs=100, alpha=0.001)    
    predict_result = sgd.predict(user_ids, item_ids)
    
    print(sgd.lossHistory)
    print(sgd.UserFactores)
    print(sgd.ItemFactores)
        
    [(u,i,p,r) for u,i,p,r in zip(user_ids, item_ids, predict_result, ratings)]
    
    