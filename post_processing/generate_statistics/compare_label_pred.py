import numpy as np
from PIL import Image
import os

def combine_images(path_l, path_p, label, pred):
	print("combining")
	img_l = Image.open(path_l+label)
	pic_l = np.array(img_l)
	
	img_p = Image.open(path_p+pred)
	pic_p = np.array(img_p)
	
	row_l, col_l, channel_l = pic_l.shape
	row_p, col_p, channel_p = pic_p.shape
	
	combined_image = np.zeros((row_l,col_l+col_p,3))
	
	combined_image[0:row_l,0:col_l] = pic_l
	combined_image[0:row_l,col_l:col_l+col_p] = pic_p
	
	return combined_image
	
	

def main():

	path_label = "labels_colors_resized/"
	path_compare = "compare/"
	dest = "label_pred/"
	
	label_files = []
	pred_files = []

	for file in os.listdir(path_label):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			label_files.append(file)
			
	for file in os.listdir(path_compare):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			pred_files.append(file)
			
	pair_list = list(zip(label_files, pred_files))
	print(len(pair_list))
	for pair in pair_list:
		combined_image = combine_images(path_label, path_compare, pair[0],pair[1])
		c_i = Image.fromarray(np.uint8(combined_image))
		c_i.save(dest+pair[0])
			
if __name__ == '__main__':
	main()	