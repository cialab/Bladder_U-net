import numpy as np
from PIL import Image
import os


def colorImage(pic):
	print("Coloring tile")
	row, col, channels = 0,0,1;
	colored_image = None
	try:
		row, col, channels = pic.shape
	except:
		row, col = pic.shape
	
	img = np.zeros((row,col,3))

	for j in range(0, col):
		for k in range(0, row):
			num = pic[k][j]
			if num == 0:
				img[k][j] = [128, 128, 128]
			elif num == 1:
				img[k][j] = [255, 128, 0]
			elif num == 2:
				img[k][j] = [128, 0, 0]
			elif num == 3:
				img[k][j] = [0, 128, 0]
			elif num == 4:
				img[k][j] = [0, 0, 128]
			elif num == 5:
				img[k][j] = [255, 0, 128]
			elif num == 6:
				img[k][j] = [64, 0, 64]	
			elif num == 7:
				img[k][j] = [128, 255, 0]	
			elif num == 8:
				img[k][j] = [128, 128, 0]
			elif num == 9:
				img[k][j] = [0, 255, 128]

	colored_image = img

	return colored_image

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
	

def main():

	path = "img_label/valannot/"
	dest = "img_label/colored/"
	
	masks = []
	
	for file in os.listdir(path):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			masks.append(file)

	mask_parts = list(split(masks, 5))
	use_part = 4
	
	print("Coloring part: " + str(use_part))

	for file in mask_parts[use_part]:
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			print(file)
			image = Image.open(path+file)
			pic = np.array(image)
			colored_image = colorImage(pic)
			c_i = Image.fromarray(np.uint8(colored_image))
			c_i.save(dest+file)
	
if __name__ == '__main__':
	main()	