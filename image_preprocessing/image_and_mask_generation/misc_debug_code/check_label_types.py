import os
import numpy as np
import cv2
from shutil import copyfile
import random
from PIL import Image
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 17483324430


def main():
	path_mask = "mask_tiles_labelsonly/"

	for file in os.listdir(path_mask):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			img_m = Image.open(path_mask+file)
			pic_m = np.array(img_m)
			if 9 in pic_m[:, :]:
				print("found 9 in: " + file)
			if 2 in pic_m[:, :]:
				print("found 2 in: " + file)
				
if __name__ == '__main__':
	main()	