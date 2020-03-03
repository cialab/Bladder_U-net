import numpy as np
from PIL import Image


file_to_read = "bladder_mask_train.npy"

a = np.load(file_to_read)
dir = "view_npy/"
subdir = "mask/"

print("begin saving")

num = 0
num_class = 8
for image in a:
	print(num)
	image = image.reshape(512, 512,num_class)
	row, col, chan = image.shape
	valid_image = np.zeros((row,col))
	for i in range(0, row-1):
		for j in range(0, col-1):
			list_of_colors = [a for a in image[i,j]]
			index = list_of_colors.index(1)
			#print("index is: " + str(index))
			if index:
				valid_image[i,j] = index
			else:
				valid_image[i,j] = 0
	
	#img_stitched = np.zeros((row, col))
	img = Image.fromarray(np.uint8(valid_image))
	img.save(dir+subdir+str(num)+".png")
	num += 1
	if num % 100 == 0: print( str(num) + " done")

"""
def fillIt(imgdatas, i):
	x = np.ones((5,5,3), dtype=np.uint8)
	imgdatas[i] = x

imgdatas = np.ndarray((10, 5, 5, 3), dtype=np.uint8)
print(str(imgdatas))
i = 0
for x in range(0,10):
	fillIt(imgdatas,i)
	i+=1
	
#print(str(imgdatas))
"""