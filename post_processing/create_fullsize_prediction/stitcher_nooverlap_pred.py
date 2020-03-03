import numpy as np
import cv2
from PIL import Image

import PIL.Image
import PIL.ImageOps
PIL.Image.MAX_IMAGE_PIXELS = 1748332443000000000000000

import os
import argparse
import random

import ast


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

def dim_file_gen(dir, is_gray):
	out_dims = {}
	files = []
	print("gen dims")
	for file in os.listdir(dir):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			print(file)
			if is_gray:
				image = Image.open(dir+file)
			else:
				image = Image.open(dir+file)#/255

				
			pic = np.array(image)
			out_dims[file_name] = pic.shape
			files.append(file_name)
			
	return out_dims, files

def fill_tiles(out_dims, stride, out_name, input_dir, is_gray):
	row, col, channels = 0,0,1
	if is_gray:
		row, col = out_dims[out_name]
	else:
		row, col, channels = out_dims[out_name]
		
	print("Image dims: " + str((row,col,channels)))
	tiles = [[0 for x in range(0,int(row/stride[1])+1)] for y in range (0,int(col/stride[0])+1)] 

	for file in os.listdir(input_dir):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			
			name_data = file_name.split('-')
			main_pic_name_list = name_data[:-2]
			tile_i, tile_j = name_data[-2:]
			main_pic_name = '-'.join(main_pic_name_list)
			tile_i = int(tile_i)
			tile_j = int(tile_j)
			
			if file_extension != ".db" and main_pic_name == out_name:
				print("Image: " + file + " is at: (" + str(tile_i) + ", " + str(tile_j) + ") ")
				image = Image.open(input_dir+file)
				pic = np.array(image)
				try:
					tiles[tile_i][tile_j] = pic
				except:
					print("weirdly, found something at: " + str(((tile_i*stride[1]),(tile_j*stride[0]))))
					print("even though dims are: " + str((row,col)))
				
				

	return tiles
	
def stitch(out_dims, out_name, tiles, stride, is_gray):	
	print('##########################################')
	print('############### Stitching ################')
	print('##########################################')
	
	row, col, channels = 0,0,1
	img_stitched = None
	if is_gray:
		row, col = out_dims[out_name]
		img_stitched = np.zeros((row, col))
	else:
		row, col, channels = out_dims[out_name]
		img_stitched = np.zeros((row, col, channels))
		
	
		
	print("Created image with dims: " + str((row,col)))
	#set base color to gray
	img_stitched[...] = (128, 128, 128)
	
	print("Color at 00: " + str((int(img_stitched[0][0][0]),int(img_stitched[0][0][1]),int(img_stitched[0][0][2]))))
	print("Color at 00  (r) : " + str(img_stitched[0][0]))
	
	row_ctr = 0
	col_ctr = 0
	col_end = 0
	
	t_row = 512
	t_col = 512
	
	common_colors = {}
	
	for i in range (0, int(col/stride[0])+1):
	
		#handle cutoff bits at bottom
		if col_ctr+int(t_col) < col :
			col_end = col_ctr+int(t_col)
		else:
			col_end = col
		
		for j in range (0, int(row/stride[1])+1):
			#print("stitching tile #: (" + str(i) +", " +str(j) + ")")
			tile = tiles[i][j]
	
			if row_ctr+int(t_row) < row:
				
				if isinstance(tile, int):
					tile = np.zeros((t_row, col_end - col_ctr, channels))
					tile[...] = (128, 128, 128)
				
				#img_stitched[row_ctr:row_ctr+int(t_row),col_ctr:col_end] = tile
				dim_r = (row_ctr+int(t_row))-row_ctr
				dim_c = col_end-col_ctr
				try:
					img_stitched[row_ctr:row_ctr+int(t_row),col_ctr:col_end] = tile
				except:
					print("Tile shape: " + str(tile.shape))
					print("Img stch shape: " + str(((row_ctr+int(t_row))-row_ctr,col_end-col_ctr)))
					print("dimC: " + str(dim_c) + " dimR: " + str(dim_r))
					chopped_tile = tile[0:dim_r,0:dim_c]
					print("Chopped tile shape: " + str(chopped_tile.shape))
					img_stitched[row_ctr:row_ctr+int(t_row),col_ctr:col_end] = chopped_tile
					
				row_ctr = row_ctr+int(stride[1])#+1
			else:
				
				if isinstance(tile, int):
					tile = np.zeros((row - row_ctr, col_end - col_ctr, channels))
					tile[...] = (128, 128, 128)

				#img_stitched[row_ctr:row,col_ctr:col_end] = tile
				dim_r = row-row_ctr
				dim_c = col_end-col_ctr
				try:
					img_stitched[row_ctr:row,col_ctr:col_end] = tile
				except:
					print("Tile shape: " + str(tile.shape))
					print("Img stch shape: " + str((row-row_ctr,col_end-col_ctr)))
					chopped_tile = tile[0:dim_r,0:dim_c]
					print("Chopped tile shape: " + str(chopped_tile.shape))
					img_stitched[row_ctr:row,col_ctr:col_end] = chopped_tile

		col_ctr = col_ctr+int(stride[0])#+1
		row_ctr = 0
	
	
	print("converting to img")
	if is_gray:
		img = Image.fromarray(np.uint8(img_stitched))
	else:
		try:
			img = Image.fromarray(np.uint8(img_stitched))
		except:
			print("Step 1: creating blank image")
			img = Image.new('RGB', (row, col))
			print("Step 2: modifying pixels")
			for i in range(img.size[0]): # for every pixel:
				for j in range(img.size[1]):
					img.putpixel((i,j),(int(img_stitched[i][j][0]),int(img_stitched[i][j][1]),int(img_stitched[i][j][2])))
				if i % 100 == 0: print("Completed: " + str(i) + " rows")
			img = img.transpose(Image.ROTATE_90)
			img = img.transpose(Image.FLIP_TOP_BOTTOM)
			
	return img, row, col
	
def main():
	#parse commandline args
	parser = argparse.ArgumentParser(description='segmentation test')
	parser.add_argument('--i', required=False, dest='path', metavar='NAME', help='name of image directory')
	parser.add_argument('--d', required=False, dest='dim_masks', metavar='NAME', help='name of full mask directory')
	parser.add_argument('--f', required=False, dest='dim_images', metavar='NAME', help='name of full image directory')
	parser.add_argument('--m', required=False, dest='masks', metavar='NAME', help='name of mask directory')
	parser.add_argument('--o', required=False, dest='output', metavar='NAME', help='name of output directory')
	parser.add_argument('--s', required=False, dest='tile_stride', metavar='NAME', help='magnitude of stride')
	
	args = parser.parse_args()
	
	#path = "inputs/image/"
	path = "test_out/image_pred/"
	if args.path is not None:
		path = args.path
		
	#masks = "inputs/mask/"
	masks = "test_out/mask/"
	if args.masks is not None:
		masks = args.masks
	
	#masks = "inputs/mask/"
	dim_masks = "test_in/mask/"
	if args.dim_masks is not None:
		dim_masks = args.dim_masks
		
	dim_images = "test_in/image_pred/"
	if args.dim_images is not None:
		dim_images = args.dim_images
	
	#output = "train/"
	output = "test_stitched/"
	if args.output is not None:
		output = args.output
	
	stride = (512,512)
	#if args.tile_stride is not None:
	#	stride = int(args.tile_stride)
	
	#return dictionary of dimensions of each file in dim_masks
	#return list of names of files in dim_masks
	#out_dims_mask, mask_files = dim_file_gen(dim_masks, False)

	#for the file we're stitching, fill a 2d array of tiles
	#with the corresponding tiles, then return this array
	
	#for file in mask_files:
	#	print(out_dims_mask[file])
	
	out_dims_images, image_files = dim_file_gen(dim_images, False)
	#for file in mask_files:
	#	print(out_dims_mask[file])
		
	is_gray = False
	
	for out_name in image_files:
		print("stitching: " + out_name)
		tiles = fill_tiles(out_dims_images, stride, out_name, path, is_gray)
		stitched, row, col = stitch(out_dims_images, out_name, tiles, stride, is_gray)
		#stitched = PIL.ImageOps.invert(stitched)
		stitched.save(output+"image_pred/"+out_name+".png")
		
		row_small = 4000
		col_small = 4000 * (col/row)
		
		print("resizing image to: " + str((int(col_small),row_small)))
		preview = resize_image(input_image_path=output+"image_pred/"+out_name+".png",size=(int(col_small),row_small))
		preview.save(output+"preview_pred/"+out_name+".png")
		
		row_small = 800
		col_small = 800 * (col/row)
		
		print("resizing image to: " + str((int(col_small),row_small)))
		thumb = resize_image(input_image_path=output+"image_pred/"+out_name+".png",size=(int(col_small),row_small))
		thumb.save(output+"thumb_pred/"+out_name+".png")
		

if __name__ == '__main__':
	main()	