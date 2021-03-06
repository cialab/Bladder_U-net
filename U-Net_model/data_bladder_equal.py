# -*- coding:utf-8 -*-

'''
This code aimed in doind data processsing, which 
'''
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob

import time
from multiprocessing import Process, Manager

class dataProcess(object):
    
    #specifying the location of all the data -Y.Z
    #out_rows and out_cols are pixle of output image/array -Y.Z
    
	def __init__(self, out_rows, out_cols, train_path="Bladder_equal/train", train_label="Bladder_equal/trainannot",
				 val_path="Bladder_equal/val", val_label="Bladder_equal/valannot",
				 test_path="Bladder_equal/test", test_label='Bladder_equal/testannot', npy_path="./npydata", img_type="png"):
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.train_path = train_path
		self.train_label = train_label
		self.img_type = img_type
		self.val_path = val_path
		self.val_label = val_label
		self.test_path = test_path
		self.test_label = test_label
		self.npy_path = npy_path

    
	def label2class(self, label, num_class):
		#256 was 3
		
        x = np.zeros([self.out_rows, self.out_cols, num_class]) #just curious if this can be replaced with a 2D-array instead of a 3-D array -Y.Z
		label_val = 0
		for i in range(self.out_rows):
			for j in range(self.out_cols):
				label_val = label[i][j]
				if label_val > num_class-1:
					label_val = 0 #Set all intersection to background
				x[i, j, int(label_val)] = 1
		return x

	
	def populate_np(self, imgs, labels, x, num_class):
		imgpath = imgs[x]
		labelpath = labels[x]
		
		img = load_img(imgpath, grayscale=False, target_size=[512, 512]) 
		
		label = load_img(labelpath, grayscale=True, target_size=[512, 512])
		label = self.label2class(img_to_array(label),num_class)
	
		return img, label
	
	def create_train_data(self):
		num_class = 8
	
		i = 0
		print('Creating training images...')
		imgs0 = sorted(glob.glob(self.train_path+"/*."+self.img_type))
		imgs1 = sorted(glob.glob(self.test_path+"/*."+self.img_type))
		imgs = imgs0 + imgs1
		labels0 = sorted(glob.glob(self.train_label+"/*."+self.img_type))
		labels1 = sorted(glob.glob(self.test_label + "/*." + self.img_type))
		labels = labels0 + labels1
		imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
		
		#12 is num colors on grayscale mask, each color pertains to a class
		imglabels = np.ndarray((len(labels), self.out_rows, self.out_cols, num_class), dtype=np.uint8)
		print(len(imgs), len(labels))

		for x in range(len(imgs)):
			img, label = self.populate_np(imgs, labels, x, num_class)
			imgdatas[i] = img
			imglabels[i] = label 
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
			
		print('loading done')
		np.save(self.npy_path + '/bladder_equal_train.npy', imgdatas)
		np.save(self.npy_path + '/bladder_equal_mask_train.npy', imglabels)
		print('Saving to .npy files done.')
	
	"""
	def populate_mp(self, imgs, labels, x, num_class, imgdatas, imglabels, index):
		imgpath = imgs[x]
		labelpath = labels[x]
		
		img = load_img(imgpath, grayscale=False, target_size=[512, 512]) 
		
		label = load_img(labelpath, grayscale=True, target_size=[512, 512])
		label = self.label2class(img_to_array(label),num_class)
	
		#return img, label

		imgdatas.append(img)
		imglabels.append(label)
		index.append(x)
		
		if x % 10 == 0:
			print('Done: {0}/{1} images'.format(x, len(imgs)))
			
	def create_train_data(self):
		num_class = 8
	
		i = 0
		print('Creating training images...')
		imgs0 = sorted(glob.glob(self.train_path+"/*."+self.img_type))
		imgs1 = sorted(glob.glob(self.test_path+"/*."+self.img_type))
		imgs = imgs0 + imgs1
		labels0 = sorted(glob.glob(self.train_label+"/*."+self.img_type))
		labels1 = sorted(glob.glob(self.test_label + "/*." + self.img_type))
		labels = labels0 + labels1
		imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
		
		#12 is num colors on grayscale mask, each color pertains to a class
		imglabels = np.ndarray((len(labels), self.out_rows, self.out_cols, num_class), dtype=np.uint8)
		print(len(imgs), len(labels))

	
		print("running FIRST")
		with Manager() as manager:
			imgdat = manager.list()  # <-- can be shared between processes.
			imglab = manager.list()  # <-- can be shared between processes.
			idx = manager.list()
			processes = []
			for x in range(0,5000):
				p = Process(target=self.populate_mp, args=(imgs, labels, x, num_class, imgdat, imglab, idx,))
				processes.append(p)
				p.start()
				
			print("waiting for processes to finish")
			for p in processes:
				p.join()
				p.terminate()
			
			print("zipping")
			datind = zip(imgdat, idx)
			labind = zip(imglab, idx)
			print("Moving imgdat")
			for dat, ind in datind:
				imgdatas[ind] = dat
			print("Moving imglab")	
			for lab, ind in labind:
				imglabels[ind] = lab
		
		
		print("running SECOND")
		with Manager() as manager:
			imgdat = manager.list()  # <-- can be shared between processes.
			imglab = manager.list()  # <-- can be shared between processes.
			idx = manager.list()
			processes = []
			for x in range(5000,10000):
				p = Process(target=self.populate_mp, args=(imgs, labels, x, num_class, imgdat, imglab, idx,))
				processes.append(p)
				p.start()
				
			print("waiting for processes to finish")
			for p in processes:
				p.join()
				p.terminate()
			
			print("zipping")
			datind = zip(imgdat, idx)
			labind = zip(imglab, idx)
			print("Moving imgdat")
			for dat, ind in datind:
				imgdatas[ind] = dat
			print("Moving imglab")	
			for lab, ind in labind:
				imglabels[ind] = lab

		print("running THIRD")
		with Manager() as manager:
			imgdat = manager.list()  # <-- can be shared between processes.
			imglab = manager.list()  # <-- can be shared between processes.
			idx = manager.list()
			processes = []
			for x in range(10000,len(imgs)):
				p = Process(target=self.populate_mp, args=(imgs, labels, x, num_class, imgdat, imglab, idx,))
				processes.append(p)
				p.start()
				
			print("waiting for processes to finish")
			for p in processes:
				p.join()
				p.terminate()
			
			print("zipping")
			datind = zip(imgdat, idx)
			labind = zip(imglab, idx)
			print("Moving imgdat")
			for dat, ind in datind:
				imgdatas[ind] = dat
			print("Moving imglab")	
			for lab, ind in labind:
				imglabels[ind] = lab

		print('loading done')
		np.save(self.npy_path + '/bladder_equal_train_mp.npy', imgdatas)
		np.save(self.npy_path + '/bladder_equal_mask_train_mp.npy', imglabels)
		print('Saving to .npy files done.')
	"""

	def create_test_data(self):
		i = 0
		print('Creating test images...')
		imgs = glob.glob(self.val_path + "/*." + self.img_type)
		imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
		testpathlist = []

		for imgname in imgs:
			testpath = imgname
			testpathlist.append(testpath)
			img = load_img(testpath, grayscale=False, target_size=[512, 512])
			img = img_to_array(img)
			imgdatas[i] = img
			i += 1

		txtname = './results/bladder_equal.txt'
		with open(txtname, 'w') as f:
			for i in range(len(testpathlist)):
				f.writelines(testpathlist[i] + '\n')
		print('loading done')
		np.save(self.npy_path + '/bladder_equal_test.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

	def create_val_test_data(self):
		i = 0
		print('Creating test images...')
		imgs = glob.glob("Bladder_Val/val" + "/*." + self.img_type)
		imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
		testpathlist = []

		for imgname in imgs:
			testpath = imgname
			testpathlist.append(testpath)
			img = load_img(testpath, grayscale=False, target_size=[512, 512])
			img = img_to_array(img)
			imgdatas[i] = img
			i += 1

		txtname = './results_val/bladder_equal.txt'
		with open(txtname, 'w') as f:
			for i in range(len(testpathlist)):
				f.writelines(testpathlist[i] + '\n')
		print('loading done')
		np.save(self.npy_path + '/bladder_equal_test_val.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

	def load_train_data(self):
		print('load train images...')
		imgs_train = np.load(self.npy_path + "/bladder_equal_train.npy").astype('float32') / 255
		print("done loading train, now loading mask")
		imgs_mask_train = np.load(self.npy_path + "/bladder_equal_mask_train.npy").astype('float32') /255
		print("done loading mask")
		#imgs_train = imgs_train.astype('float32')
		#imgs_mask_train = imgs_mask_train.astype('float32')
		#imgs_train /= 255
		#imgs_mask_train /= 255
		return imgs_train, imgs_mask_train

######################the two below are pretty much the same#############Y.Z
	def load_test_data(self):
		print('-' * 30)
		print('load test images...')
		print('-' * 30)
		imgs_test = np.load(self.npy_path + "/bladder_equal_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		return imgs_test
		
	def load_test_data_val(self):
		print('-' * 30)
		print('load test images...')
		print('-' * 30)
		imgs_test = np.load(self.npy_path + "/bladder_equal_test_val.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		return imgs_test


if __name__ == "__main__":
	mydata = dataProcess(512, 512)
	mydata.create_train_data()
	mydata.create_test_data()
