import numpy as np
import cv2
from PIL import Image

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 17483324430

import os
import argparse
import random

import time
import multiprocessing 

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

	return blank_colors/total_colors > 0.8
	
	
def mergeTiles(tile):
	row, col, channels = tile.shape;
	merge_tile = np.zeros((row,col))
		
	for j in range(0, col):
		for i in range(0, row):
			color = tile[i][j]
			
			if color[0] < 235 or color[1] < 235 or color[2] < 235:
				merge_tile[i][j] = 1
	
	return merge_tile

image_tiles = []
	
def overlay_mask(img_p, mask_p, img, mask, dest):
	"""
	img = image_tiles[ind]
	
	img_p = "mtlb_images_tiled/"
	mask_p = "mask/"
	mask = img
	dest = "mask_overlay_tiled_proc/"

	#print("overlaying: " + mask)
	"""
	img_i = Image.open(img_p+img)
	pic_i = np.array(img_i)
	img_i.close()
	
	if not mostly_blank(pic_i):
	
		img_m = Image.open(mask_p+mask)
		pic_m = np.array(img_m)
		img_m.close()
		
		overlay = mergeTiles(pic_i)
		
		row, col = pic_m.shape;
		merge_mask = np.zeros((row,col))
			
		for j in range(0, col):
			for i in range(0, row):
				if pic_m[i][j] == 0:
					merge_mask[i][j] = overlay[i][j]
				else:
					merge_mask[i][j] = pic_m[i][j]
	
		mask_image = Image.fromarray(np.uint8(merge_mask))
		mask_image.save(dest+mask)
		mask_image.close()
		
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))	
	
def main():
	#parse commandline args
	parser = argparse.ArgumentParser(description='segmentation test')
	parser.add_argument('--i', required=False, dest='path', metavar='NAME', help='name of image directory')
	parser.add_argument('--m', required=False, dest='masks', metavar='NAME', help='name of mask directory')
	parser.add_argument('--o', required=False, dest='output', metavar='NAME', help='name of output directory')
	parser.add_argument('--s', required=False, dest='tile_stride', metavar='NAME', help='magnitude of stride')
	
	args = parser.parse_args()
	
	path = "image_tiles/"
	if args.path is not None:
		path = args.path
		
	masks = "mask/"
	if args.masks is not None:
		masks = args.masks
		
	#output = "train/"
	output = "mask_tiles_overlay/"
	#output = "test_out/"
	if args.output is not None:
		output = args.output
	
	#image_tiles = []
	mask_tiles = []
	
	for file in os.listdir(masks):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			mask_tiles.append(file)

	
	mask_parts = list(split(mask_tiles, 10))
		
	use_part = 2
	print("using: " + str(use_part))
	num_cpu = multiprocessing.cpu_count()
	processes = []
	for image in mask_parts[use_part]:
	

		p = multiprocessing.Process(target=overlay_mask, args=(path, masks, image, image, output,))
		processes.append(p)
		p.start()

	for p in processes:
		p.join()
	
if __name__ == '__main__':
	main()	