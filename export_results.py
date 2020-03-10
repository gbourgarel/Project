import torch
import matplotlib.pyplot as plt
from image_dataset import ImageDataset, imshow, save_image
from train import PSNR
import os
from tqdm import tqdm

# import data
batch_size=1
folder='results4'

device='cuda' if torch.cuda.is_available() else 'cpu'
print('Device:',device)

path='RAIN_DATASET_COMPRESSED/LABELLED'
classes=['LABELLED_REAL_RAIN_INPUTS','LABELLED_CLEAR_INPUTS']
# path='RAIN_DATASET_2_COMPRESSED/train'
# classes=['data','gt']

export_path=os.path.join(folder,'images')
if not os.path.exists(export_path):
    os.mkdir(export_path)

data_loader = torch.utils.data.DataLoader(
    ImageDataset(path,classes[0],classes[1]),
    batch_size=batch_size,
    shuffle=True)

net=torch.load(os.path.join(folder,'net.pkl'),map_location=torch.device(device))
net.eval()

psnr_dict={}
for data in tqdm(data_loader):
    rain=data['rain'].to(device)
    clear=data['clear']
    output=net(rain).detach().cpu()
    psnr=PSNR(output,clear)
    for i in range(psnr.size(0)):
        psnr_dict[data['name'][i]]=psnr[i].item()
        save_image(output[i], os.path.join(export_path,data['name'][i]))
torch.save(psnr_dict,os.path.join(export_path,'0_psnr_scores.pkl'))
sum(psnr_dict.values())/len(psnr_dict.values())
