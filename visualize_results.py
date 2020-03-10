import torch
import matplotlib.pyplot as plt
from image_dataset import ImageDataset, imshow
from train import PSNR
import numpy as np
from PIL import Image
import os


results_folder='results4'


psnr=torch.load(os.path.join(results_folder,'images/0_psnr_scores.pkl'))
P=np.array(list(psnr.values()))
print(P.mean(),np.median(P),P.std(),len(P))
psnr={k:v for k,v in sorted(psnr.items(),key=lambda x:x[1])}

# open worst, median and best images
keys=list(psnr.keys())
im1=keys[0]
im2=keys[len(keys)//2]
im3=keys[-1]
print(psnr[im1],psnr[im2],psnr[im3])
Image.open(os.path.join(results_folder,'images',im1)).show()
Image.open(os.path.join(results_folder,'images',im2)).show()
Image.open(os.path.join(results_folder,'images',im3)).show()


#print(PSNR(out,data['clear']))
#imshow(data['rain'],out,data['clear'])

loss_train=torch.load(os.path.join(results_folder,'loss_train.pkl'))
psnr_train=torch.load(os.path.join(results_folder,'psnr_train.pkl'))
loss_test=torch.load(os.path.join(results_folder,'loss_test.pkl'))
psnr_test=torch.load(os.path.join(results_folder,'psnr_test.pkl'))

fig,axs=plt.subplots(2)
axs[0].plot(loss_train,label='Train')
axs[0].plot(loss_test, label='Test')
axs[0].set_title('Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('MSE Loss')
axs[0].legend()
# axs[0].ylim(bottom=0)
axs[1].plot(psnr_train,label='Train')
axs[1].plot(psnr_test, label='Test')
axs[1].set_title('PSNR')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('PSNR')
axs[1].legend()
plt.show()
