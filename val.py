# -*- coding: utf-8 -*-

import time
time1 = time.time()
import keras
import numpy as np
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import os
import sys
base_dir = 'F:\\py_experiment\\'
model_path = base_dir+'model.h5'
model = load_model(model_path)
#model = model.load_weights(model_path)

# img_path = base_dir+'test\\2\\155.jpg'
img_path = sys.argv[1]

img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x   = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x   = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)  
#x  = np.expand_dims(x, axis=0)  #作用同上一步
x /= 255

print(x)
# import matplotlib.pyplot as plt  # 可视化单幅图像
# plt.imshow(x[0])
# plt.show()

feature_maps = model.predict(x)
print(feature_maps)
predict_index = np.argmax(feature_maps)
print(predict_index)
print( str( round(max(feature_maps[0])*100,3) )+'%' )
cates = ['分类1','分类2','分类3','分类4']
print(cates[predict_index])
print(str(round( time.time()-time1 ,3))+'s' )