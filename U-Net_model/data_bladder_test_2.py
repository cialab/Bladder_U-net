# -*- coding:utf-8 -*-

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob


class dataProcess(object):
	def __init__(self, out_rows, out_cols, train_path="Bladder_test_2/train", train_label="Bladder_test_2/trainannot",
				 val_path="Bladder_test_2/val", val_label="Bladder_test_2/valannot",
				 test_path="Bladder_test_2/test", test_label='Bladder_test_2/testannot', npy_path="./npydata", img_type="png"):
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
		x = np.zeros([self.out_rows, self.out_cols, num_class])
		label_val = 0
		for i in range(self.out_rows):
			for j in range(self.out_cols):
				label_val = label[i][j]
				if label_val > num_class-1:
					label_val = 1 #Set all intersection to unlabelled other
				x[i, j, int(label_val)] = 1
		return x

	def create_train_data(self):
		num_class = 10
	
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
			imgpath = imgs[x]
			labelpath = labels[x]
			img = load_img(imgpath, grayscale=False, target_size=[512, 512]) 
			label = load_img(labelpath, grayscale=True, target_size=[512, 512]) #set to true if it doesnt work
			img = img_to_array(img)
			#print("Calling 'label2class' on: " + str(img_to_array(label)))
			testy= img_to_array(label)
			#print("DIMMY: " + str(len(testy)))
			label = self.label2class(img_to_array(label),num_class)
			#break
			imgdatas[i] = img
			imglabels[i] = label # <-- new issue
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1

		print('loading done')
		np.save(self.npy_path + '/bladder_test_2_train.npy', imgdatas)
		np.save(self.npy_path + '/bladder_test_2_mask_train.npy', imglabels)
		print('Saving to .npy files done.')

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

		txtname = './results_test/bladder_test_2.txt'
		with open(txtname, 'w') as f:
			for i in range(len(testpathlist)):
				f.writelines(testpathlist[i] + '\n')
		print('loading done')
		np.save(self.npy_path + '/bladder_test_2_test.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

	def create_val_test_data(self):
		i = 0
		print('Creating test images...')
		imgs = glob.glob("Bladder_test_2_Val/val" + "/*." + self.img_type)
		imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
		testpathlist = []

		for imgname in imgs:
			testpath = imgname
			testpathlist.append(testpath)
			img = load_img(testpath, grayscale=False, target_size=[512, 512])
			img = img_to_array(img)
			imgdatas[i] = img
			i += 1

		txtname = './results_val/test/bladder_test_2.txt'
		with open(txtname, 'w') as f:
			for i in range(len(testpathlist)):
				f.writelines(testpathlist[i] + '\n')
		print('loading done')
		np.save(self.npy_path + '/bladder_test_2_test_val.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

	def load_train_data(self):
		print('load train images...')
		imgs_train = np.load(self.npy_path + "/bladder_test_2_train.npy")
		imgs_mask_train = np.load(self.npy_path + "/bladder_test_2_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		imgs_mask_train /= 255
		return imgs_train, imgs_mask_train

	def load_test_data(self):
		print('-' * 30)
		print('load test images...')
		print('-' * 30)
		imgs_test = np.load(self.npy_path + "/bladder_test_2_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		return imgs_test
		
	def load_test_data_val(self):
		print('-' * 30)
		print('load test images...')
		print('-' * 30)
		imgs_test = np.load(self.npy_path + "/bladder_test_2_test_val.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		return imgs_test



if __name__ == "__main__":
	mydata = dataProcess(512, 512)
	mydata.create_train_data()
	mydata.create_test_data()
