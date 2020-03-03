import numpy as np
import cv2
from PIL import Image

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 17483324430
import random

import os
from shutil import copyfile

import time
import multiprocessing 

def removedUnlabelled(path_m, path_i, file, out_m, out_i):
	img_i = Image.open(path_i+file)
	pic_i = np.array(img_i)
	
	img_m = Image.open(path_m+file)
	pic_m = np.array(img_m)
	
	row_m, col_m = pic_m.shape
	row_i, col_i, channel_i = pic_i.shape
	
	image_onlylabels = np.zeros((row_i,col_i,channel_i))
	mask_onlylabels = np.zeros((row_m,col_m))
	
	save_mask = False
	
	all_px = 0
	labelled_px = 0
	
	for j in range(0, col_m):
		for i in range(0, row_m):
			label = pic_m[i][j]
			color = pic_i[i][j]
			if label == 0 or label == 1 or label == 9:
				label = 0
				#color = [245+random.randint(-2,2),245+random.randint(-2,2),245+random.randint(-2,2)]
				color = [245,245,245]
			else:
				label -= 1
				if label == 9:
					print("there seems to be a problem here officer: " + file)
				#save_mask = True
				labelled_px +=1
				
				
			mask_onlylabels[i][j] = label
			image_onlylabels[i][j] = color
			
			all_px += 1
				
	percent_labelled = labelled_px/all_px * 100
	if percent_labelled >= 25:
		#print("all px: " + str(all_px) + " labelled_px: " + str(labelled_px) + " for img: " + file)
		save_mask = True
			
	if(save_mask):
		new_mask = Image.fromarray(np.uint8(mask_onlylabels))
		new_mask.save(out_m+ file)
		
		new_image = Image.fromarray(np.uint8(image_onlylabels))
		new_image.save(out_i+ file)

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
	
def main():
	path_img = "image_labelled/"
	path_mask = "mask_labelled/"

	output_mask = "mask_tiles_labelsonly/"
	output_img = "image_tiles_labelsonly/"

	masks = []


	for file in os.listdir(path_mask):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			masks.append(file)
			
	mask_parts = list(split(masks, 6))
		
	use_part = 4
	#print("REM UNLAB section: " + str(use_part))
	print("multiproc: " + str(use_part))
	processes = []
		
	for file in mask_parts[use_part]:
		p = multiprocessing.Process(target=removedUnlabelled, args=(path_mask, path_img, file, output_mask, output_img,))
		processes.append(p)
		p.start()
	
	for p in processes:
		p.join()
	
		#removedUnlabelled(path_mask, path_img, file, output_mask, output_img)
	
if __name__ == '__main__':
	main()	
	
