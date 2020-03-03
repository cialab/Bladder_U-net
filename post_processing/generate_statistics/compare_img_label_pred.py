import numpy as np
from PIL import Image
import os

def combine_images(path_i, path_l, path_p, image, label, pred):
	print("combining")
	img_i = Image.open(path_i+image)
	pic_i = np.array(img_i)
	
	img_l = Image.open(path_l+label)
	pic_l = np.array(img_l)
	
	img_p = Image.open(path_p+pred)
	pic_p = np.array(img_p)
	
	row_i, col_i, channel_i = pic_i.shape
	row_l, col_l, channel_l = pic_l.shape
	row_p, col_p, channel_p = pic_p.shape
	
	combined_image = np.zeros((row_i,col_i+col_l+col_p,3))
	
	combined_image[0:row_i,0:col_i] = pic_i
	combined_image[0:row_i,col_i:col_i+col_l] = pic_l
	combined_image[0:row_i,col_i+col_l:col_i+col_l+col_p] = pic_p
	
	return combined_image
	
	

def main():

	path_img = "img_label/test/"
	path_label =  "img_label/test-colored/"
	path_compare = "results/test/12/he_n/"
	dest = "side_by_side/12_test/he_n/"
	
	image_files = []
	label_files = []
	pred_files = []
	
	for file in os.listdir(path_img):
		file_name, file_extension = os.path.splitext(file)
		if file_extension == ".png":
			image_files.append(file)

	for file in os.listdir(path_label):
		file_name, file_extension = os.path.splitext(file)
		if file_extension == ".png":
			label_files.append(file)
			
	for file in os.listdir(path_compare):
		file_name, file_extension = os.path.splitext(file)
		if file_extension == ".png":
			pred_files.append(file)
			
	triple_list = list(zip(image_files, label_files, pred_files))
	print(len(triple_list))
	for triple in triple_list:
		combined_image = combine_images(path_img, path_label, path_compare, triple[0], triple[1], triple[2])
		c_i = Image.fromarray(np.uint8(combined_image))
		c_i.save(dest+triple[0])
			
if __name__ == '__main__':
	main()	