import numpy as np
import cv2
from PIL import Image

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 17483324430

import os
import argparse
import random

from shutil import copyfile

def split(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
	
def get_tissues(path, image):

	img = Image.open(path + image)
	colors = img.getcolors(img.size[0]*img.size[1])
	img.close()
	
	colors_seen = []
	
	for c in colors:
		color = c[1]
		if color != 0 and color not in colors_seen:
			colors_seen.append(color)
			
	#print("Saw colors: " + str(colors_seen) + " in " + path + image)
	return colors_seen

def main():
	path_mask = "Bladder/valannot/"
	
	tissues = {
		'LP': [],
		'MP': [],
		'M': [],
		'RBC': [],
		'INF': [],
		'CT': [],
		'MM': [],
	}
	
	tissues_names = {
		'LP': [],
		'MP': [],
		'M': [],
		'RBC': [],
		'INF': [],
		'CT': [],
		'MM': [],
	}
	
	masks = []
	
	for file in os.listdir(path_mask):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			masks.append(file)
		
	#mask_parts = list(split(masks, 3))
	#use_part = 2
		
	#print("using part: " + str(use_part))
		
	print("Determining tissue presence in Test Set")
		
	total_tiles = 0
	for file in masks: #mask_parts[use_part]:
		total_tiles += 1
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			colors_seen = get_tissues(path_mask, file)
			
			name_data = file_name.split('-')
			main_pic_name_list = name_data[:-2]
			main_pic_name = '-'.join(main_pic_name_list)
			
			for color in colors_seen:
				if color == 1:
					tissues['LP'].append(file)
					if main_pic_name not in tissues_names['LP']:
						tissues_names['LP'].append(main_pic_name)
				if color == 2:
					tissues['MP'].append(file)
					if main_pic_name not in tissues_names['MP']:
						tissues_names['MP'].append(main_pic_name)
				if color == 3:
					tissues['M'].append(file)
					if main_pic_name not in tissues_names['M']:
						tissues_names['M'].append(main_pic_name)
				if color == 4:
					tissues['RBC'].append(file)
					if main_pic_name not in tissues_names['RBC']:
						tissues_names['RBC'].append(main_pic_name)
				if color == 5:
					tissues['INF'].append(file)
					if main_pic_name not in tissues_names['INF']:
						tissues_names['INF'].append(main_pic_name)
				if color == 6:
					tissues['CT'].append(file)
					if main_pic_name not in tissues_names['CT']:
						tissues_names['CT'].append(main_pic_name)
				if color == 7:
					tissues['MM'].append(file)
					if main_pic_name not in tissues_names['MM']:
						tissues_names['MM'].append(main_pic_name)
	
	
	print("Num tiles with Lamina Propria: " + str(len(tissues['LP'])))
	print("Slides with Lamina Propria: " + str((tissues_names['LP'])))
	print("Num tiles with Muscularis Propria: " + str(len(tissues['MP'])))
	print("Slides with Muscularis Propria: " + str((tissues_names['MP'])))
	print("Num tiles with Mucosa: " + str(len(tissues['M'])))
	print("Slides with Mucosa: " + str((tissues_names['M'])))
	print("Num tiles with Red Blood Cells: " + str(len(tissues['RBC'])))
	print("Slides with Red Blood Cells: " + str((tissues_names['RBC'])))
	print("Num tiles with Inflamation: " + str(len(tissues['INF'])))
	print("Slides with Inflamation: " + str((tissues_names['INF'])))
	print("Num tiles with Cautery: " + str(len(tissues['CT'])))
	print("Slides with Cautery: " + str((tissues_names['CT'])))
	print("Num tiles with Muscularis Mucosa: " + str(len(tissues['MM'])))
	print("Slides with Muscularis Mucosa: " + str((tissues_names['MM'])))
	print("Total Number of tiles in set: " + str(total_tiles))
	
	
if __name__ == '__main__':
	main()	