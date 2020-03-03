import numpy as np
from PIL import Image
import os


#taken from: https://www.blog.pythonlibrary.org/2017/10/12/how-to-resize-a-photo-with-python/
def resize_image(input_image_path,
				 size):
	original_image = Image.open(input_image_path)
	width, height = original_image.size
	#print('The original image size is {wide} wide x {height} '
	#	  'high'.format(wide=width, height=height))
 
	resized_image = original_image.resize(size)
	width, height = resized_image.size
	#print('The resized image size is {wide} wide x {height} '
	#	  'high'.format(wide=width, height=height))
	#resized_image.show()
	#resized_image.save(output_image_path)
	return resized_image

def resizeImage(img_path, image, filename):
	rotation_vals = [0, 90, 180, 270]
	rotations = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
	
	main_pic_name, tile_i, tile_j, tile_number, rotation = filename.split('-')
	
	print("rotating: " + filename + ": " + rotation)

	rotation = int(rotation)
	
	img = image
	if rotation_vals.index(rotation) == 1 or rotation_vals.index(rotation) == 3:
		img = resize_image(input_image_path=img_path,size=(480,360))
	
	return img
	
def main():

	path = "labels_colors/"
	dest = "labels_colors_resized/"
	#path = "images/"
	#dest = "images_resized/"

	for file in os.listdir(path):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			print(file)
			image = Image.open(path+file)
			rot_img = resizeImage(path+file, image, file_name)
			rot_img.save(dest+file)
			
if __name__ == '__main__':
	main()	