import numpy as np
from tqdm import tqdm
import torch
import os



def train(model,train_data_loader,test_data_loader,device,loss_fn,optimizer,n_epochs=1, show=False, export_folder=None):
    model.train(True)
    loss_train = np.zeros(n_epochs)
    psnr_train = np.zeros(n_epochs)
    loss_test = np.zeros(n_epochs)
    psnr_test = np.zeros(n_epochs)

    t=tqdm(total=n_epochs*(len(train_data_loader)+len(test_data_loader)))
    for epoch_num in range(n_epochs):
        running_loss = .0
        running_psnr = .0

        # losses=[]
        model.train()
        for i,data in enumerate(train_data_loader):
            # change_lr(optimizer)
            rain=data['rain'].to(device)
            clear=data['clear'].to(device)
            output=model(rain)
            loss=loss_fn(output,clear)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # losses.append(loss.item())
            running_loss += loss.item()
            running_psnr += PSNR(output, clear).sum().item()/clear.size(0)

            del rain, clear, output, loss
            t.update()

        loss_train[epoch_num] = running_loss / len(train_data_loader)
        psnr_train[epoch_num] = running_psnr / len(train_data_loader)

        running_loss = .0
        running_psnr = .0
        model.eval()
        for i,data in enumerate(test_data_loader):
            rain=data['rain'].to(device)
            clear=data['clear'].to(device)
            output=model(rain)
            loss=loss_fn(output,clear)

            running_loss += loss.item()
            running_psnr += PSNR(output, clear).sum().item()/clear.size(0)

            del rain, clear, output, loss
            t.update()

        loss_test[epoch_num] = running_loss / len(test_data_loader)
        psnr_test[epoch_num] = running_psnr / len(test_data_loader)

        print('Epoch {}/{}: Train - Loss: {:.4f} - PSNR: {:.4f} - Test - Loss: {:.4f} - PSNR: {:.4f}'
              .format(epoch_num, n_epochs, loss_train[epoch_num], psnr_train[epoch_num], loss_test[epoch_num], psnr_test[epoch_num]))

        if export_folder:
            if not os.path.exists(export_folder):
                os.mkdir(export_folder)
            torch.save(model,os.path.join(export_folder,'net.pkl'))
            torch.save(loss_train,os.path.join(export_folder,'loss_train.pkl'))
            torch.save(psnr_train,os.path.join(export_folder,'psnr_train.pkl'))
            torch.save(loss_test,os.path.join(export_folder,'loss_test.pkl'))
            torch.save(psnr_test,os.path.join(export_folder,'psnr_test.pkl'))
    t.close()
    return loss_train,psnr_train,loss_test,psnr_test


def change_lr(optim):
    for p in optim.param_groups:
        p['lr']*=10**.001

def PSNR(predicted, true):
    vector=predicted-true
    vector=vector.view(vector.size(0),-1)
    mse=torch.norm(vector,p=2,dim=1)**2/vector.size(1)
    return 10*torch.log10(1/mse)
