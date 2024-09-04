import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt

def training_loop(n_epochs, optimizer, model, loss_fn, batchsize, train_loader, val_loader):
    rmse_loss_train = []
    rmse_loss_val = []
    for epoch in range(1, n_epochs + 1):  
        loss_train = 0
        loss_val = 0
        for xtrain, ytrain in train_loader:  
            
            outputs = model(xtrain)  
            loss = loss_fn(outputs, ytrain.view(batchsize,-1)) 
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            loss_train += loss.item()  

            
        for xval, yval in val_loader:
            with torch.no_grad():
                
                model.eval()
                outputs = model(xval)
                loss = loss_fn(outputs, yval.view(batchsize,-1))
                loss_val += loss.item()  
        
        rmse_loss_train.append(np.sqrt(loss_train/len(train_loader)))
        rmse_loss_val.append(np.sqrt(loss_val/len(val_loader)))
        
        if epoch == 1 or epoch % 10 == 0:
            print(' Epoch {}| Training rmse {} | Validation rmse {} '.format(
                epoch,
                rmse_loss_train[-1], rmse_loss_val[-1]))    
              
    H = pd.DataFrame(data= {'loss':rmse_loss_train, 'val_loss':rmse_loss_val})
    # # determine the total number of epochs used for training, then
    #  # initialize the figure
    # n_ep = np.arange(0, len(H["loss"]))
    # plt.style.use("ggplot")
    # (fig, axs) = plt.subplots(1, 1)

    # # plot the *unshifted* training and validation loss
    # plt.style.use("ggplot")
    # axs.plot(n_ep, H["loss"], label="train_loss")
    # axs.plot(n_ep, H["val_loss"], label="val_loss")
    # axs.set_title("Loss Plot {}".format(model.__class__.__name__))
    # axs.set_xlabel("Epoch #")
    # axs.set_ylabel("Loss")
    # axs.legend()
 
    return H  

def validate(model, batchsize, train_loader, val_loader, loss_fn):
    accdict = {}
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        loss_tot = 0
        
        with torch.no_grad():  # <1>
            for features, target in loader:
                outputs = model(features)
                loss = loss_fn(outputs,target.view(batchsize,-1))
                loss_tot += loss.item()

        print("{} RMSE {}: {:.6f}".format(model.__class__.__name__, name , np.sqrt(loss_tot/len(loader))))
        accdict[name] = np.sqrt(loss_tot/len(loader))
    return accdict    