import numpy as np
from PIL import Image
import os

def combine_images(path_8_he_n,path_12_he_n,path_8_gl_u,path_10_gl_u,path_12_gl_u, he_n_8, he_n_12, gl_u_8, gl_u_10, gl_u_12):
	print("combining")
	img_he_n_8 = Image.open(path_8_he_n+he_n_8)
	pic_he_n_8 = np.array(img_he_n_8)
	
	img_he_n_12 = Image.open(path_12_he_n+he_n_12)
	pic_he_n_12 = np.array(img_he_n_12)
	
	img_gl_u_8 = Image.open(path_8_gl_u+gl_u_8)
	pic_gl_u_8 = np.array(img_gl_u_8)
	
	img_gl_u_10 = Image.open(path_10_gl_u+gl_u_10)
	pic_gl_u_10 = np.array(img_gl_u_10)
	
	img_gl_u_12 = Image.open(path_12_gl_u+gl_u_12)
	pic_gl_u_12 = np.array(img_gl_u_12)
	
	row_he_n_8, col_he_n_8, channel_he_n_8 = pic_he_n_8.shape
	row_he_n_12, col_he_n_12, channel_he_n_12 = pic_he_n_12.shape
	row_gl_u_8, col_gl_u_8, channel_gl_u_8 = pic_gl_u_8.shape
	row_gl_u_10, col_gl_u_10, channel_gl_u_10 = pic_gl_u_10.shape
	row_gl_u_12, col_gl_u_12, channel_gl_u_12 = pic_gl_u_12.shape
	
	combined_image = np.zeros((row_he_n_8+row_he_n_12+row_gl_u_8+row_gl_u_10+row_gl_u_12,col_he_n_8,3))
	
	combined_image[0:row_he_n_8,0:col_he_n_8] = pic_he_n_8
	combined_image[row_he_n_8:row_he_n_8+row_he_n_12,0:col_he_n_8] = pic_he_n_12
	combined_image[row_he_n_8+row_he_n_12:row_he_n_8+row_he_n_12+row_gl_u_8,0:col_he_n_8] = pic_gl_u_8
	combined_image[row_he_n_8+row_he_n_12+row_gl_u_8:row_he_n_8+row_he_n_12+row_gl_u_8+row_gl_u_10,0:col_he_n_8] = pic_gl_u_10
	combined_image[row_he_n_8+row_he_n_12+row_gl_u_8+row_gl_u_10:row_he_n_8+row_he_n_12+row_gl_u_8+row_gl_u_10+row_gl_u_12,0:col_he_n_8] = pic_gl_u_12
	
	
	return combined_image
	
	

def main():

	
	path_8_he_n = "side_by_side/8_test/he_n/"
	path_12_he_n = "side_by_side/12_test/he_n/"
	path_8_gl_u = "side_by_side/8_test/gl_u/"
	path_10_gl_u = "side_by_side/10_test/gl_u/"
	path_12_gl_u = "side_by_side/12_test/gl_u/"
	
	dest = "side_by_side/all_together_test/"
	
	he_n_8_files = []
	he_n_12_files = []
	gl_u_8_files = []
	gl_u_10_files = []
	gl_u_12_files = []
	
	for file in os.listdir(path_8_he_n):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			he_n_8_files.append(file)
			
	for file in os.listdir(path_12_he_n):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			he_n_12_files.append(file)
	
	for file in os.listdir(path_8_gl_u):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			gl_u_8_files.append(file)
	
	for file in os.listdir(path_10_gl_u):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			gl_u_10_files.append(file)
	
	for file in os.listdir(path_12_gl_u):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			gl_u_12_files.append(file)
	
	penta_list = list(zip(he_n_8_files, he_n_12_files, gl_u_8_files, gl_u_10_files, gl_u_12_files))
	
	print(len(penta_list))
	for pent in penta_list:
		try:
			combined_image = combine_images(path_8_he_n,path_12_he_n,path_8_gl_u,path_10_gl_u,path_12_gl_u, pent[0], pent[1], pent[2], pent[3], pent[4])
			c_i = Image.fromarray(np.uint8(combined_image))
			c_i.save(dest+pent[0])
		except:
			print("minor error, skipping")
			
if __name__ == '__main__':
	main()	