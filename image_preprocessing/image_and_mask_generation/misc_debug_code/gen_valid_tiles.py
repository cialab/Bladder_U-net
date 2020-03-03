import os
from shutil import copyfile

path_img = "mtlb_images_tiled/"
path_mask = "mask_overlay_tiled/"
output = "valid_image_tiles/" #"mtlb_images_tiled/"


masks = []

for file in os.listdir(path_mask):
	file_name, file_extension = os.path.splitext(file)
	if file_extension != ".db":
		masks.append(file_name)
		
print("generated list of image tiles to add. Size is: " + str(len(masks)))
		
for file in os.listdir(path_image):
	file_name, file_extension = os.path.splitext(file)
	if file_extension != ".db" and file_name in masks:
		print("Copying: " + path_img + file + " to " + output + file)
		copyfile(path_img + file, output + file)