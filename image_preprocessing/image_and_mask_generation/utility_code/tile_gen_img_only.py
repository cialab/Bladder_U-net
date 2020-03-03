import numpy as np
import cv2
from PIL import Image

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 17483324430000000000000

import os
import argparse
import random

def tile(pic, t_row, t_col, stride, mask):
	
	print('##########################################')
	print('################ Tiling ##################')
	print('##########################################')
	
	row, col, channels = 0,0,1
	
	if not mask:
		print("TILING PNG:")
		row, col, channels = pic.shape
	else:
		print("TILING MASK")
		row, col = pic.shape
		
	tile_pic = None
	#tiles = [[0 for x in range(0,int(row/t_row)+1)] for y in range (0,int(col/t_col)+1)] 
	tiles = [[0 for x in range(0,int(row/stride[1])+1)] for y in range (0,int(col/stride[0])+1)] 
	row_ctr = 0
	col_ctr = 0
	col_end = 0

	total_pic_area = row*col
	total_area = 0
	
	
	#divide image into segments
	for i in range (0,int(col/stride[0])+1):
		#handle cutoff bits at bottom
		if col_ctr+int(t_col) < col :
			col_end = col_ctr+int(t_col)
		else:
			col_end = col

		#horizontal tile sliding
		for j in range (0,int(row/stride[1])+1):
		
			#print("tile #: (" + str(i) +", " +str(j) + ")")
			if row_ctr+int(t_row) < row :
				tile_pic = pic[row_ctr:row_ctr+int(t_row),col_ctr:col_end]
				tiles[i][j] = tile_pic
				
				#scoot it horizontally right
				row_ctr = row_ctr+int(stride[1])
			
			else:
				####don't care about corners. Set them to black to be weeded out later
				#tile_pic = np.zeros((int(t_row), int(t_col), 3))
				tile_pic = pic[row_ctr:row,col_ctr:col_end]
				#to_add = pic[row_ctr:row,col_ctr:col_end]
				#tile_pic[0:to_add.shape[0],0:to_add.shape[1]] = to_add
				
				tiles[i][j] = tile_pic
				#print("resultant tile is: (" + str(row_ctr) + ", " + str(col_ctr) + ") & (" + str(row) + ", " + str(col_end) + ")")
			
				
		#scoot it vertically down and back to the beginning horizontally
		col_ctr = col_ctr+int(stride[0])
		row_ctr = 0

	return tiles

def mostly_blank(img):
	colors = img.getcolors(img.size[0]*img.size[1])
	total_colors, blank_colors = 0,0
	
	for c in colors:
		#if c[0] > max_occurence:
		#	(max_occurence, most_present) = c
			
		color = c[1]
		if color[0] > 230 and color[1] > 230 and color[2] > 230:
			blank_colors = blank_colors + c[0]
		
			
		total_colors = total_colors + c[0]

	return blank_colors/total_colors > 0.95
	
def operate_image(pic, stride, output, file):
	row, col, channels = pic.shape
	ratio = row/col 
	output = output# + "image/"
	#tile (subsection of img)
	t_row, t_col = (360, 480)
	
	
	tiles = tile(pic, t_row, t_col, stride, False)
	#print("DIMENSION OF TILES: " + str(len(tiles)) + ", " + str(len(tiles[0])))
	#print("DIMENSION OF GOOD_TILES: " + str(len(good_tiles)) + ", " + str(len(good_tiles[0])))

	counter = 1

	file_name, file_extension = os.path.splitext(file)

	for i in range (0,int(col/stride[0])+1):
		for j in range (0,int(row/stride[1])+1):
		
			pic = tiles[i][j]
			
			img = Image.fromarray(np.uint8(pic*255))
			
			img.save(output+file_name+"-"+str(i)+"-"+str(j)+".png")
			
			counter = counter + 1

def main():
	#parse commandline args
	parser = argparse.ArgumentParser(description='segmentation test')
	parser.add_argument('--i', required=False, dest='path', metavar='NAME', help='name of image directory')
	parser.add_argument('--m', required=False, dest='masks', metavar='NAME', help='name of mask directory')
	parser.add_argument('--o', required=False, dest='output', metavar='NAME', help='name of output directory')
	parser.add_argument('--s', required=False, dest='tile_stride', metavar='NAME', help='magnitude of stride')
	
	args = parser.parse_args()
	
	#path = "inputs/image/"
	path = "image/"
	if args.path is not None:
		path = args.path
	
	#output = "train/"
	output = "image_tiled/"
	if args.output is not None:
		output = args.output
	
	stride = (480,360)
	#if args.tile_stride is not None:
	#	stride = int(args.tile_stride)

	print("FINISHED GENERATING MASK TILES, MOVING TO IMAGES")
	print("FINISHED GENERATING MASK TILES, MOVING TO IMAGES")
	print("FINISHED GENERATING MASK TILES, MOVING TO IMAGES")
	print("FINISHED GENERATING MASK TILES, MOVING TO IMAGES")
	print("FINISHED GENERATING MASK TILES, MOVING TO IMAGES")
	print("FINISHED GENERATING MASK TILES, MOVING TO IMAGES")
	print("FINISHED GENERATING MASK TILES, MOVING TO IMAGES")
	
	path_check = "image_tiled/"
	
	already_done = []
	
	for file in os.listdir(path_check):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
		
			name_data = file_name.split('-')

			main_pic_name_list = name_data[:-2]
			main_pic_name = '-'.join(main_pic_name_list)
		
			if main_pic_name not in already_done:
				already_done.append(main_pic_name)
				print("ALREADY tiled: " + main_pic_name)
			
	for file in os.listdir(path):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db" and file_name not in already_done:
			print(file)
			image = Image.open(path+file)
			pic = np.array(image)/255

			operate_image(pic, stride, output, file)
			
	
if __name__ == '__main__':
	main()	