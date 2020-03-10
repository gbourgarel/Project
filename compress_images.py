from PIL import Image, ImageFilter
from tqdm import tqdm
import glob, os

paths=['RAIN_DATASET/ALIGNED_PAIRS/CG_DROPLETS/',
       'RAIN_DATASET/ALIGNED_PAIRS/CLEAN/',
       'RAIN_DATASET/ALIGNED_PAIRS/REAL_DROPLETS/',
       'RAIN_DATASET/LABELLED/LABELLED_CG_RAIN_INPUTS/',
       'RAIN_DATASET/LABELLED/LABELLED_CLEAR_INPUTS/',
       'RAIN_DATASET/LABELLED/LABELLED_MASKS/',
       'RAIN_DATASET/LABELLED/LABELLED_REAL_RAIN_INPUTS/']

size=540,480

for path in paths:
    print(path)
    new_path='RAIN_DATASET_COMPRESSED'+path[12:]
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for file in tqdm(glob.glob(path+'*.png')):
        im=Image.open(file)
        # if im.size != (776,690):
        #     print(im.size)
        im.thumbnail(size)
        new_file='RAIN_DATASET_COMPRESSED'+file[12:]
        split=new_file.split('/')
        split[-1]=split[-1].split('_')[-1]
        new_file='/'.join(split)
        new_file=new_file[:-3]+'jpg'
        # we remove the 'left_' or 'right' in the file
        im.save(new_file,'JPEG')
