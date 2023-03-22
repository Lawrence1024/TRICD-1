# This file is not supposed to be used. First try at creating tnesor image.

from PIL import Image
from numpy import asarray
import torch
import numpy as np
import os
import torchvision.transforms as transforms

image_array = []
transform = transforms.ToTensor()

directory = "/home/lawrence92/TRICD/train_images"

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        print(f)
        # load the image and convert into
        # numpy array
        img = Image.open(f)
        
        # asarray() class is used to convert
        # PIL images into NumPy arrays
        # numpydata = asarray(img)
        tempTen = transform(img)
        image_array.append(tempTen)
        # <class 'numpy.ndarray'>
        # print(type(numpydata))
        
        #  shape
        # print(numpydata.shape)

# image_array = np.array(image_array)
with open('image_tensor.t','wb+') as f:
  torch.save(image_array, f)