#Set of functions for the correlation coefficients that are used for feature selection
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def corr_red(df, corr, note = False):
    
    ind = np.where(abs(df.corr().to_numpy())  > corr) #finds the values that are > corr, and gives a tuple

    ndiag = sorted([sorted([ind[0][i], ind[1][i]]) #gives the nondiag tuples
            for i in range(0,len(ind[0])) if ind[0][i]!= ind[1][i]])
    
    ndiag_f = {ndiag[i][0] for i  in range(len(ndiag)) } #takes the first elements of ndiag tuples
    
    if note == True:
        print(f'The features {[list(df.columns)[i] for i in ndiag_f]} are taken out')

    return  list(df.drop(columns = 
                         [list(df.columns)[i] for i in ndiag_f]).columns)
#return reduced df columns 



def corr_plot(df):
    corr_matrix = pd.DataFrame(df.values
                           , columns= df.columns).corr() #correlation matrix and plots
    
    return plt.show(sns.heatmap(corr_matrix, cmap="coolwarm"
                                , center=0, annot=True, fmt=".1g"))

def corr_list(df, start, stop, step):
    c_list = []
    
    for value in np.arange(start, stop, step):
        if value == start:
            c_list.append(corr_red(df, value))
        elif  corr_red(df, value) != c_list[-1] :
            c_list.append(corr_red(df, value))
            
    return c_list #creates a list of features given the bound of correlation and stepsize

#for the iq scores
def drop_outliers(dataframe, column):
    q1 = pd.DataFrame(dataframe[column]).quantile(0.25).values[0]
    q3 = pd.DataFrame(dataframe[column]).quantile(0.75).values[0]

    iqr = q3 - q1 #Interquartile range
    fence_low = q1 - (1.5*iqr)
    fence_high = q3 + (1.5*iqr)
    
    return dataframe.drop(index = dataframe.loc[(dataframe['CHL'] < fence_low) 
                                                | (dataframe['CHL'] > fence_high)].index)

# for the z-scores
def drop_outliers_z(dataframe, column, threshold):
    mean = pd.DataFrame(dataframe[column]).mean().values[0]
    std = pd.DataFrame(dataframe[column]).std().values[0]

    dataframe['Z_Score'] = (dataframe[column]-mean)/std
    
    return dataframe.drop(index = dataframe.loc[(dataframe['Z_Score'] > threshold) 
                                               | (dataframe['Z_Score'] < -threshold)].index)
