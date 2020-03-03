import numpy as np
import cv2
from PIL import Image

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 17483324430

import os
import argparse
import random

from shutil import copyfile


def mostly_blank(pic):
	img = Image.fromarray(np.uint8(pic))
	colors = img.getcolors(img.size[0]*img.size[1])
	total_colors, blank_colors = 0,0
	img.close()
	for c in colors:
		#if c[0] > max_occurence:
		#	(max_occurence, most_present) = c
			
		color = c[1]
		
		if color[0] >= 235 and color[1] >= 235 and color[2] >= 235:
			blank_colors = blank_colors + c[0]
		
			
		total_colors = total_colors + c[0]

	return blank_colors/total_colors > 0.9

def check_blank(img_p, img):

	#print("overlaying: " + mask)

	img_i = Image.open(img_p+img)
	pic_i = np.array(img_i)
	
	if not mostly_blank(pic_i):
		return False
		
	return True
	
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))	

def get_name(file_name):
	name_data = file_name.split('-')
	main_pic_name_list = name_data[:-2]
	main_pic_name = '-'.join(main_pic_name_list)
	return main_pic_name

def main():
	#parse commandline args
	parser = argparse.ArgumentParser(description='segmentation test')
	parser.add_argument('--i', required=False, dest='path', metavar='NAME', help='name of image directory')
	parser.add_argument('--m', required=False, dest='masks', metavar='NAME', help='name of mask directory')
	parser.add_argument('--o', required=False, dest='output', metavar='NAME', help='name of output directory')
	parser.add_argument('--s', required=False, dest='tile_stride', metavar='NAME', help='magnitude of stride')
	
	args = parser.parse_args()
	
	#path = "inputs/image/"
	path = "04/" #"val_tiles_512/"
	if args.path is not None:
		path = args.path
	
	#output = "train/"
	output = "04_predict/" #"val_tiles_predict_512/"
	#output = "test_out/"
	if args.output is not None:
		output = args.output

	
	images = []
	
	for file in os.listdir(path):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			images.append(file)
			
	image_parts = list(split(images, 5))
	use_part = 4
		
	print("using part: " + str(use_part))
			
	curr_dir = None		
	for file in images:#image_parts[use_part]:
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			img_name = get_name(file_name)
			curr_dir = output + img_name + "/"
			if not os.path.isdir(curr_dir):
				os.mkdir(curr_dir)
				
			exists = os.path.isfile(curr_dir+file)
			if not exists:
				if not check_blank(path, file):
					copyfile(path+file,curr_dir+file)
	
if __name__ == '__main__':
	main()