import torch
import torch.nn as nn
import torch.optim as optim
from image_dataset import ImageDataset, imshow
from u_net import UNet
from train import train

# import data
batch_size=2
lr=1e-5
n_epochs=10

device='cuda' if torch.cuda.is_available() else 'cpu'
print('Device:',device)

path='RAIN_DATASET_COMPRESSED/ALIGNED_PAIRS'
classes=['REAL_DROPLETS','CLEAN']
# path='RAIN_DATASET_2_COMPRESSED/train'
# classes=['data','gt']

dataset = ImageDataset(path,classes[0],classes[1])
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(.8*len(dataset)),len(dataset)-int(.8*len(dataset))])
train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True)
test_data_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True)

net=UNet().to(device)
# net=torch.load('results3/net.pkl').to(device)
#loss_fn=nn.MSELoss()
loss_fn=nn.SmoothL1Loss()
optimizer=optim.Adam(net.parameters(),lr=lr)

loss_train,psnr_train,loss_test,psnr_test=train(net,train_data_loader,test_data_loader,device,loss_fn,optimizer,n_epochs=n_epochs, show=False, export_folder='results4')

torch.save(net,'results/net.pkl')
torch.save(loss_train,'results/loss_train.pkl')
torch.save(psnr_train,'results/psnr_train.pkl')
torch.save(loss_test,'results/loss_test.pkl')
torch.save(psnr_test,'results/psnr_test.pkl')
