import numpy as np
import torch
from pathlib import Path
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import imageio.v2 as imageio
from matplotlib import pyplot as plt
import os

class grapher:
  
  def __init__(self, model, X_val, y_val , red_ten_shape ) -> None:
   
   self.pred_list = []
   self.model = model
   self.red_ten_shape = red_ten_shape
   self.X_val = X_val
   self.y_val = y_val
   self.model_name = model.__class__.__name__
   self.gifname = self.model_name+'.gif'
   self.path_im = Path.cwd()/'images'/self.model_name
   self.path_gifs = (self.path_im.parent)/'gifs'
   os.makedirs(self.path_im, exist_ok=True)
   os.makedirs(self.path_gifs, exist_ok=True)

 # Takes in Xval then spits out list of predictions
  def pred_lister(self):
   with torch.no_grad():
     for i in np.arange(self.X_val.shape[0]):
        self.model.eval()
        pred = self.model(self.X_val[i].unsqueeze(0)).view(self.red_ten_shape).to(device = 'cpu')
        self.pred_list.append(pred.numpy())
    
   return self.pred_list 
  
  # Takes in prediction list then back transforms them to original data shape with NaNs 
  # and plots the predictions vs yval

  def plotter(self, reverter_func, fps = 3):
    for i in np.arange(len(self.X_val)):
        fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(10, 8),gridspec_kw={'width_ratios': [1, 1.05]})
        im  = ax.imshow( np.flipud(reverter_func(self.pred_list[i])),cmap ='turbo', interpolation='nearest')
        plt.tight_layout()
        im2 = ax2.imshow( np.flipud(reverter_func(self.y_val[i])),cmap ='turbo', interpolation='nearest')
        plt.tight_layout()
        ax.set_title('Predicted')
        ax2.set_title('Validation')
        ax.set_ylabel(i)
    
        divider = make_axes_locatable(ax2)
        im.set_clim(-2, 6)
        im2.set_clim(-2, 6)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax)
        plt.tight_layout()
        fig.suptitle(self.model_name,y=0.8, fontsize = 20)
        plt.savefig( (self.path_im)/(str("{0:02d}".format(i))+'_pred.png'))
        plt.close() 
    ims = [imageio.imread(f) for f in list(sorted((self.path_im).glob('*_pred.png')))]
    return imageio.mimwrite((self.path_gifs)/self.gifname, ims, fps=fps)        