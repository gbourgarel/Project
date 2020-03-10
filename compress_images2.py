from PIL import Image, ImageFilter
from tqdm import tqdm
import glob, os

paths=['train/data/',
       'train/gt/',
       'test_b/data/',
       'test_b/gt/']

size=540,480

for path in paths:
    print(path)
    new_path='RAIN_DATASET_2_COMPRESSED/'+path
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for file in tqdm(glob.glob(path+'*.*')):
        im=Image.open(file)
        width,height=im.size
        new_width=size[0]*height/size[1]
        left=(width-new_width)/2
        right=width-(width-new_width)/2
        top=0
        bottom=height
        im=im.crop((left,top,right,bottom))
        im=im.resize(size)
        if im.size != size:
            print(im.size)
        #
        # im.crop(())
        new_file='RAIN_DATASET_2_COMPRESSED/'+file
        # we remove the '_rain' or '_clean' in the file
        im.save('_'.join(new_file.split('_')[:-1])+'.jpg','JPEG')
