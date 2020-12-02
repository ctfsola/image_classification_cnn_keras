# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image



# 从文件夹读取图片和标签到numpy数组中
# 标签信息在文件夹名中，例如1/100.jpg表示该图片的标签为1
def read_data(data_dir):
    fpaths = []
    for cate_name in os.listdir(data_dir):
    	for fname in os.listdir(data_dir+cate_name):
    		fpath = data_dir+cate_name+'/'+ fname
    		fpaths.append(fpath+'::'+cate_name)
    np.random.shuffle(fpaths)
    return fpaths

def process_map(fpaths):
	image_paths = []
	labels      = []
	for item in fpaths:
		image_path,label = item.split('::')
		image_paths.append(image_path)
		labels.append(label)
	labels = np.array(labels).astype(np.int32)
	return image_paths,labels

#from tensorflow.keras.preprocessing.image import img_to_array, load_img,array_to_img
# def vetor(img_path):
# 	img = load_img(img_path, target_size=(150, 150))  
# 	x   = img_to_array(img)  # Numpy array with shape (150, 150, 3)
# 	x   = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
# 	x /= 255
# 	return x

def vectorize(fnames):
	datas = []
	for img in fnames:
		image = Image.open(img)
		image = image.resize([150, 150])
		data  = np.array(image) / 255.0
		# vetor_pic = array_to_img(vetor(img))
		# vetor_pic.show()
		datas.append(data)
	datas = np.array(datas)
	return datas

#--------------------------------------------------------



def load_data():
	base_dir  = 'F:/py_experiment/'
	data_dir  = base_dir+'data/'
	train_dir = data_dir+'train/'
	val_dir   = data_dir+'val/'
	labels = []
	cate_arr = os.listdir(train_dir)
	for x in cate_arr:
		labels.append(int(x))
	classes = np.array(labels)

	train_fpaths = read_data(train_dir)
	val_fpaths   = read_data(val_dir)

	train_data_path ,train_labels = process_map(train_fpaths)
	val_data_path ,val_labels     = process_map(val_fpaths)
	#print("shape of train_datas: {}\tshape of labels: {}".format(datas.shape,labels.shape))
	return vectorize(train_data_path),train_labels,vectorize(val_data_path),val_labels,classes






