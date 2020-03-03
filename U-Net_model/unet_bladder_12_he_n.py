# -*- coding:utf-8 -*-
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
#from data_bladder import *
from data_bladder_equal import *

from sklearn.utils import shuffle

class myUnet(object):
	def __init__(self, img_rows=512, img_cols=512):
		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):
		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test
		
	def load_val_data(self):
		mydata = dataProcess(self.img_rows, self.img_cols)
		mydata.create_val_test_data()
		imgs_test = mydata.load_test_data_val()
		return imgs_test

	def get_unet(self):
		inputs = Input((self.img_rows, self.img_cols, 3))
		
		kern_init = 'he_normal'
		num_class = 8

		conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = kern_init)(inputs)
		# print((conv1)
		print( "conv1 shape:", str(conv1.shape))
		print( "conv1:"+ str(conv1))
		
		conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = kern_init)(conv1)
		print( "conv1 shape:", str(conv1.shape))
		print( "conv1:"+ str(conv1))
		
		conv1 = Dropout(0.2)(conv1)
		print( "drop1 shape:", str(conv1.shape))
		print( "drop1:"+ str(conv1))
		
		
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print( "pool1 shape:", str(pool1.shape))
		print( "pool1:"+ str(pool1))

		conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = kern_init)(pool1)
		print( "conv2 shape:", str(conv2.shape))
		print( "conv2:"+ str(conv2))
		
		conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = kern_init)(conv2)
		print( "conv2 shape:", str(conv2.shape))
		print( "conv2:"+ str(conv2))
		
		conv2 = Dropout(0.2)(conv2)
		print( "drop2 shape:", str(conv2.shape))
		print( "drop2:"+ str(conv2))
		
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print( "pool2 shape:", str(pool2.shape))
		print( "pool2:"+ str(pool2))

		conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = kern_init)(pool2)
		print( "conv3 shape:", str(conv3.shape))
		print( "conv3:"+ str(conv3))
		
		conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = kern_init)(conv3)
		print( "conv3 shape:", str(conv3.shape))
		print( "conv3:"+ str(conv3))
		
		conv3 = Dropout(0.2)(conv3)
		print( "drop3 shape:", str(conv3.shape))
		print( "drop3:"+ str(conv3))
		
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print( "pool3 shape:", str(pool3.shape))
		print( "pool3:"+ str(pool3))

		conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = kern_init)(pool3)
		print( "conv4 shape:", str(conv4.shape))
		print( "conv4:"+ str(conv4))
		conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = kern_init)(conv4)
		print( "conv4 shape:", str(conv4.shape))
		print( "conv4:"+ str(conv4))
		drop4 = Dropout(0.2)(conv4)
		print( "drop4:"+ str(drop4))
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
		print( "pool4 shape:", str(pool4.shape))
		print( "pool4:"+ str(pool4))

		conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer = kern_init)(pool4)
		print( "conv5 shape:", str(conv5.shape))
		print( "conv5:"+ str(conv5))
		conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer = kern_init)(conv5)
		print( "conv5 shape:", str(conv5.shape))
		print( "conv5:"+ str(conv5))
		drop5 = Dropout(0.5)(conv5)
		print( "drop5:"+ str(drop5))
		pool5 = MaxPooling2D(pool_size=(2, 2))(drop5)
		print( "pool5 shape:", str(pool5.shape))
		print( "pool5:"+ str(pool5))
		
		conv6 = Conv2D(2048, 3, activation='relu', padding='same', kernel_initializer = kern_init)(pool5)
		print( "conv6 shape:", str(conv6.shape))
		print( "conv6:"+ str(conv6))
		conv6 = Conv2D(2048, 3, activation='relu', padding='same', kernel_initializer = kern_init)(conv6)
		print( "conv6 shape:", str(conv6.shape))
		print( "conv6:"+ str(conv6))
		drop6 = Dropout(0.5)(conv6)
		
		up7 = Conv2D(1024, 2, activation='relu', padding='same', kernel_initializer = kern_init)(UpSampling2D(size=(2, 2))(drop6))
		concatenate7 = concatenate([drop5, up7], axis=3)
		print("up6:"+str(up7))
		print("concat6:"+str(concatenate7))
		conv7 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = kern_init)(concatenate7)
		print("conv7:"+ str(conv7))
		conv7 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer = kern_init)(conv7)
		print("conv7:"+ str(conv7))
		
		
		up8 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer = kern_init)(UpSampling2D(size=(2, 2))(conv7))
		concatenate8 = concatenate([conv4, up8], axis=3)
		print("up8:"+str(up8))
		print("concat8:"+str(concatenate8))
		conv8 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = kern_init)(concatenate8)
		print("conv8:"+ str(conv8))
		conv8 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = kern_init)(conv8)
		print("conv8:"+ str(conv8))
		
		up9 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer = kern_init)(UpSampling2D(size=(2, 2))(conv8))
		concatenate9 = concatenate([conv3, up9], axis=3)
		print("up9:"+str(up9))
		print("concat9:"+str(concatenate9))
		conv9 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = kern_init)(concatenate9)
		print("conv9:"+ str(conv9))
		conv9 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = kern_init)(conv9)
		print("conv9:"+ str(conv9))
		
		up10 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer = kern_init)(
			UpSampling2D(size=(2, 2))(conv9))
		concatenate10 = concatenate([conv2, up10], axis=3)
		print("up10:"+str(up10))
		print("concaten10:"+str(concatenate10))
		conv10 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = kern_init)(concatenate10)
		print("conv10:"+ str(conv10))
		conv10 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = kern_init)(conv10)
		print("conv10:"+ str(conv10))
		
		up11 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer = kern_init)(
			UpSampling2D(size=(2, 2))(conv10))
		concatenate11 = concatenate([conv1, up11], axis=3)
		print("up11:"+str(up11))
		print("concaten11:"+str(concatenate11))
		conv11 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = kern_init)(concatenate11)
		print("conv11:"+ str(conv11))
		conv11 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = kern_init)(conv11)
		print("conv11:"+ str(conv11))
		conv11 = Conv2D(num_class, 3, activation='relu', padding='same', kernel_initializer = kern_init)(conv11)
		print( "conv11 shape:", str(conv11.shape))
		print("conv11:"+ str(conv11))
		
		conv12 = Conv2D(num_class, 1, activation='softmax')(conv11)
		print("conv12: " + str(conv12))
		model = Model(input=inputs, output=conv12)

		model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

		return model

	def train(self):
		print(("loading data"))
		imgs_train, imgs_mask_train, imgs_test = self.load_data()
		print(("loading data done"))
		model = self.get_unet()
		
		print(("got unet"))
		"""
		model_checkpoint = ModelCheckpoint('unet_+"bladder_equal"+_12_he_n.hdf5', monitor='val_acc', verbose=1, save_best_only=True)
		print(('Fitting model...'))
		#set epochs back to 50
		model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=50, verbose=1,
				  validation_split=0.1, shuffle=True, callbacks=[model_checkpoint])
		"""
		
		#EarlyStopping(monitor='val_acc', patience=3),
		callback = [ModelCheckpoint('unet_bladder_equal_12_he_n.hdf5', monitor='val_acc', verbose=1, save_best_only=True)]
		
		print(('Fitting model...'))
		#set epochs back to 50
		#imgs_train, imgs_mask_train = shuffle(imgs_train, imgs_mask_train)
		model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=25, verbose=1,
				  validation_split=0.1, shuffle=True, callbacks=callback)
		
		print(('predict test data'))
		imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		np.save('./results/12/he_n_eq/bladder_equal_mask_test.npy', imgs_mask_test)

	def save_img(self):
		print(("array to image"))
		imgs = np.load('./results/12/he_n_eq/bladder_equal_mask_test.npy')
		piclist = []
		for line in open("./results/bladder_equal.txt"):
			line = line.strip()
			picname = line.split('/')[-1]
			piclist.append(picname)
		for i in range(imgs.shape[0]):
			path = "./results/12/he_n_eq/" + piclist[i]
			img = np.zeros((imgs.shape[1], imgs.shape[2], 3), dtype=np.uint8)
			for k in range(len(img)):
				for j in range(len(img[k])):
					num = np.argmax(imgs[i][k][j])
					if num == 0:
						img[k][j] = [128, 128, 128]
					elif num == 1:
						img[k][j] = [0, 128, 255]
					elif num == 2:
						img[k][j] = [0, 0, 128]
					elif num == 3:
						img[k][j] = [0, 128, 0]
					elif num == 4:
						img[k][j] = [128, 0, 0]
					elif num == 5:
						img[k][j] = [128, 0, 255]
					elif num == 6:
						img[k][j] = [64, 0, 64]	
					elif num == 7:
						img[k][j] = [0, 255, 128]	
					elif num == 8:
						img[k][j] = [0, 128, 128]
					elif num == 9:
						img[k][j] = [128, 255, 0]
			#img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_CUBIC)
			cv2.imwrite(path, img)
			
			
	def save_val_img(self):
		print(("array to image"))
		imgs = np.load('./results_val/12/he_n_eq/bladder_equal_mask_test.npy')
		piclist = []
		for line in open("./results_val/bladder_equal.txt"):
			line = line.strip()
			picname = line.split('/')[-1]
			piclist.append(picname)
		for i in range(imgs.shape[0]):
			path = "./results_val/12/he_n_eq/" + piclist[i]
			img = np.zeros((imgs.shape[1], imgs.shape[2], 3), dtype=np.uint8)
			for k in range(len(img)):
				for j in range(len(img[k])):
					num = np.argmax(imgs[i][k][j])
					if num == 0:
						img[k][j] = [128, 128, 128]
					elif num == 1:
						img[k][j] = [0, 128, 255]
					elif num == 2:
						img[k][j] = [0, 0, 128]
					elif num == 3:
						img[k][j] = [0, 128, 0]
					elif num == 4:
						img[k][j] = [128, 0, 0]
					elif num == 5:
						img[k][j] = [128, 0, 255]
					elif num == 6:
						img[k][j] = [64, 0, 64]	
					elif num == 7:
						img[k][j] = [0, 255, 128]	
					elif num == 8:
						img[k][j] = [0, 128, 128]
					elif num == 9:
						img[k][j] = [128, 255, 0]
						
			#img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_CUBIC)
			cv2.imwrite(path, img)


if __name__ == '__main__':
	myunet = myUnet()
	print("Setting model to unet")
	model = myunet.get_unet()
	# model.summary()
	# plot_model(model, to_file='model.png')
	print("Training unet")
	myunet.train()
	print("Saving unet img")
	myunet.save_img()
