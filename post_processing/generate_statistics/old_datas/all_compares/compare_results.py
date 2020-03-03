import numpy as np
from PIL import Image
import os

def combine_images(path_gn, path_gu, path_hn, path_hu, gn, gu, hn, hu):
	print("combining")
	img_gu = Image.open(path_gu+gu)
	pic_gu = np.array(img_gu)
	
	img_gn = Image.open(path_gn+gn)
	pic_gn = np.array(img_gn)
	
	img_hu = Image.open(path_hu+hu)
	pic_hu = np.array(img_hu)
	
	img_hn = Image.open(path_hn+hn)
	pic_hn = np.array(img_hn)
	
	row_gu, col_gu, channel_gu = pic_gu.shape
	row_gn, col_gn, channel_gn = pic_gn.shape
	row_hn, col_hn, channel_hn = pic_hn.shape
	row_hu, col_hu, channel_hu = pic_hu.shape
	
	combined_image = np.zeros((row_gu+row_gn+row_hn+row_hu,col_gu,3))
	
	combined_image[0:row_gn,0:col_gu] = pic_gn
	combined_image[row_gn:row_gn+row_gu,0:col_gu] = pic_gu
	combined_image[row_gn+row_gu:row_gn+row_gu+row_hn,0:col_gu] = pic_hn
	combined_image[row_gn+row_gu+row_hn:row_gn+row_gu+row_hn+row_hu,0:col_gu] = pic_hu
	
	
	return combined_image
	
	

def main():

	
	path_gn = "glorot_normal/"
	path_gu = "glorot_uniform/"
	path_hn = "he_normal/"
	path_hu = "he_uniform/"
	
	dest = "all_compared/"
	
	gn_files = []
	gu_files = []
	hn_files = []
	hu_files = []
	
	for file in os.listdir(path_gn):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			gn_files.append(file)
			
	for file in os.listdir(path_gu):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			gu_files.append(file)
	
	for file in os.listdir(path_hn):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			hn_files.append(file)
			
	for file in os.listdir(path_hu):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			hu_files.append(file)
			
	quad_list = list(zip(gn_files, gu_files, hn_files, hu_files))
	print(len(quad_list))
	for quad in quad_list:
		combined_image = combine_images(path_gn, path_gu, path_hn, path_hu, quad[0], quad[1], quad[2], quad[3])
		c_i = Image.fromarray(np.uint8(combined_image))
		c_i.save(dest+quad[0])
			
if __name__ == '__main__':
	main()	