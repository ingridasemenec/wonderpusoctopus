import numpy as np
import torch

class reduced_ten:
    def __init__(self, dataset, all_variables) -> None:
        self.dataset = dataset  
        self.all_variables = all_variables
        self.feature_name = np.random.choice(all_variables)  # randomly chooses a feature to compare
        self.red_variables = [x for x in self.all_variables if x != self.feature_name]
        self.nan_list = []
        self.ten_shape = (dataset[self.feature_name].sortby('time').shape[1],dataset[
            self.feature_name].sortby('time').shape[2]) #shape of tensor before resize
        self.ten_size = self.ten_shape[0]*self.ten_shape[1]  #total size of tensor before resize

        

    # Randomly choose a variable and find the indices of NaN entries, then check out if indices are same across the dataset.
    # If same, the output of the nan_list will have only one entry. 
    def fit(self):
        self.nan_list = []
        for i in np.arange(self.dataset[self.feature_name].sortby('time').shape[0]):
            self.nan_list.append(np.argwhere(
                np.isnan(self.dataset[self.feature_name].sortby('time')[i].to_numpy().flatten())).flatten())
        
        if np.unique(self.nan_list).ndim == 1:
            self.nan_list = np.unique(self.nan_list) 
            self.redten_size = self.ten_size-len(self.nan_list)
        
            # If the first feature you choose have unique indices, find the rest of the features NaN entries, 
            # check if they are unique and compare with the first one you choose
        
            nan_listvar = []
            for i, names in enumerate(self.red_variables):
                 nan_l = []
            
                 for j in np.arange(self.dataset[names].sortby('time').shape[0]):
                    nan_l.append(np.argwhere(
                        np.isnan(self.dataset[names].sortby('time')[j].to_numpy().flatten())).flatten())
                     
                 nan_listvar.append(np.unique(nan_l))
            
            if np.array(nan_listvar).ndim == 2:
        
                check=0
                for i in range(np.array(nan_listvar).shape[0]):
                    check += ((nan_listvar[i] == np.unique(self.nan_list))*1).mean()
                if check == len(self.red_variables):
                    print(f'There are {len(self.nan_list)} NaNs for each feature at every month and '
                              f'their places are the same.')
            else:
                print(f'The NaN indices are not same across the rest of the dataset.')   
        else:
            print(f'The NaN indices are not same across {self.feature_name} dataset.')   
    
    # Find the divisors of the reduced tensor size so you can write it down as a reduced mxn matrix 
    def find_divisor(self):
        divisors = [] 
        for i in range(1, self.redten_size+1): 
            if self.redten_size % i == 0: 
                divisors.append(i)
        
        divisors = divisors[:-1]
        middle = len(divisors)
        if middle == 1:
            print('Grats you got a prime number!')
        if middle % 2 != 0:
            self.midd_div = divisors[int(np.ceil(middle/2))]
            self.redten_shape = (int(self.redten_size/self.midd_div),self.midd_div)
            self.poolsize = int(np.ceil(np.ceil(self.redten_shape[0]/2)/2)*np.ceil(np.ceil(self.redten_shape[1]/2)/2))

            print(f'Your original tensor of size {self.ten_shape} = {self.ten_size} is reduced '
                  f'to {self.redten_size} = {self.redten_shape}.')
        else:
            self.midd_div =  divisors[int(middle/2)]
            self.redten_shape = (int(self.redten_size/self.midd_div),self.midd_div)
            self.poolsize = int(np.ceil(np.ceil(self.redten_shape[0]/2)/2)*np.ceil(np.ceil(self.redten_shape[1]/2)/2))
            
            print(f'Your original tensor of size {self.ten_shape} = {self.ten_size} is reduced '
                  f'to {self.redten_size} = {self.redten_shape}.')

    # Takes in the reduced matrix plugs back in NaNs and reshapes back to original form
    def revert(self, x_nona):
      t = np.zeros(self.ten_size)
      for i in self.nan_list:
          t[i] = np.nan
      
      j=0
      for i in range(len(t)):
          if not np.isnan(t[i]):
              t[i] = x_nona.flatten()[j]
              j = j+1
              
      return t.reshape(self.ten_shape[0],self.ten_shape[1])
    
    # Takes a feature set, drops the NaNs and forms the reduced matrices
    def nona(self,features,tup):
      self.features = features
      self.tup = tup
      Xnonalist = [None] * len(self.features)
      for j,var in enumerate(self.features):
          Xnona = []
          for i in range(self.dataset[var].sortby('time').to_numpy().shape[0]):
              x = self.dataset[var].sortby('time')[i].to_numpy().flatten()
              x_flatnona = x[~np.isnan(x)]
              Xnona.append(x_flatnona.reshape(self.tup))
              Xnonalist[j] = Xnona
        
      return torch.from_numpy(np.array(Xnonalist))