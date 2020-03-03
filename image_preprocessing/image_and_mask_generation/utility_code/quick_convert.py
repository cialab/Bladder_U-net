import os

path_img = "bl_image_tiff/"

for file in os.listdir(path_img):
	file_name, file_extension = os.path.splitext(file)
	if  file_extension != ".db":
		os.rename(path_img+file, path_img+file_name+".tiff")

