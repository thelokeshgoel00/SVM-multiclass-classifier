#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from pathlib import Path
from keras.preprocessing import image
import cv2


# In[2]:


p = Path("Datasets/dogs-cats-horses-humans-dataset/dataset/")
dirs = p.glob("*")
image_data = []
labels = []
labels_dict = {"cat":0,"dog":1,"horse":2,"human":3}
for folder_dir in dirs:
    
    label = str(folder_dir).split("/")[-1][:-1]
    
    
    
    for img_path in folder_dir.glob("*.jpg"):
        img = image.load_img(img_path,target_size=(250,250))
        img_array = image.img_to_array(img)
        image_data.append(img_array)
        labels.append(labels_dict[label])


# In[3]:





# In[4]:


image_data = np.array(image_data,dtype="float32")/255
labels = np.array(labels)

print(image_data.shape,labels.shape)


# In[5]:


def drawImg(img):
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()
    return 
drawImg(image_data[354])


# In[ ]:




