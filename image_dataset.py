import os,glob
import torch
from torch.utils.data import Dataset
from skimage import io



class ImageDataset(Dataset):
    def __init__(self, path, rain_dir, clear_dir, transform=None):
        super(ImageDataset, self).__init__()
        self.clear_dir = os.path.join(path,clear_dir)
        self.rain_dir = os.path.join(path,rain_dir)
        self.transform = transform
        self.images=[path.split('/')[-1] for path in glob.glob(os.path.join(self.clear_dir,'*.*'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Clear image path
        img_name = self.images[idx]
        clear_path = os.path.join(self.clear_dir, img_name)

        # Rain image path
        rain_path = os.path.join(self.rain_dir, img_name)

        clear = io.imread(clear_path)
        rain = io.imread(rain_path)

        clear=torch.from_numpy(clear).permute(2,0,1)
        rain=torch.from_numpy(rain).permute(2,0,1)

        clear=clear.float()/255
        rain=rain.float()/255
        data_pair = {'rain': rain, 'clear': clear, 'name': img_name}
        if self.transform:
          data_pair=self.transform(data_pair)
        return data_pair

def imshow(rain,output,clear):
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision
    rain=np.transpose(torchvision.utils.make_grid(rain).numpy(), (1, 2, 0))
    output=np.transpose(torchvision.utils.make_grid(output).numpy(), (1, 2, 0))
    clear=np.transpose(torchvision.utils.make_grid(clear).numpy(), (1, 2, 0))
    fig,axs=plt.subplots(3)
    axs[0].imshow(rain)
    axs[1].imshow(output)
    axs[2].imshow(clear)
    plt.show()

def save_image(data, path):
    ''' data is a pytorch tensor '''
    import torchvision.transforms as transforms
    from PIL import Image
    image=transforms.ToPILImage(mode='RGB')(data)
    image.save(path)
