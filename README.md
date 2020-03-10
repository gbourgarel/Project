# Project

## Import data

Two datasets need to be uploaded.
The first dataset must be uploaded in a folder named RAIN_DATASET.
The second one must be uploaded in a folder named RAIN_DATASET_2.
In these two folders, the structure must be the following:
- two folders (one for training/testing data, one for validating data)
- in each of those, two folders that corresponds to the rain images and their corresponding clean image.
Ensure that each pair of image have the exact same name, or change the code in the compress_images(2).py files in order to do so in teh next step.

## Compress data

These dataset need to be compressed to the format JPG with dimensions 540x480.
To do so, we propose to use compress_images.py and compress_images2.py.
It brings the same structure as before, unless adapted to your own datasets.

## Train network

To train the network, you just have to run the file train_u_net.py.
The ImageDataset class is defined in image_dataset.py.
The u-net code is in the file u_net.py, and the training code is in the file train.py.
When you run the file train_u_net.py, be sure that you specify a correct export_folder (line 38), that is used to export results such as the network. If you specify a folder that already exists, you may overwrite data.
You must also be sure to indicate the correct path and classes of your training/testing data.

## Export results

You should want to export result images via a virtual machine (quicker). To do so, you can use export_results.py. You should specify in line 10, the same folder as the training export_folder. You should also specify where your validation data is located, and the classes.

## Visualize results

In your local machine, you can easily visualize results thanks to visualize_results.py.
You need to import in your local machine the whole results folder, and specify its path in line 10.
Then, you can see the worst, the median and the best image results, and a plot of loss/psnr over training epochs.
