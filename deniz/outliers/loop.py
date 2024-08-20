#for looping over models
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error


def cv_loop(model, X_train , y_train, X_test, y_test, n_folds, params, rand):
    
    cv_rmses = np.zeros(n_folds)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state = rand )
    i = 0
    for train_index, test_index in kfold.split(X_train):
        X_train_train = X_train.iloc[train_index]
        X_holdout = X_train.iloc[test_index]
        y_train_train = y_train.iloc[train_index]
        y_holdout = y_train.iloc[test_index]

        reg = model(**params)
        reg.fit(X_train_train, y_train_train)
        pred = reg.predict(X_holdout)
        cv_rmses[i] = root_mean_squared_error(y_holdout,pred)
    i=i+1
    
    reg_val =  model(**params)
    reg_val.fit(X_train, y_train)
    pred_val = reg.predict(X_test)
    val_rmse = root_mean_squared_error(y_test,pred_val)

    return (cv_rmses.mean(), pred_val.mean())

