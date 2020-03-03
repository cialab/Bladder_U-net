import os

path_img = "val/"

for file in os.listdir(path_img):
	file_name, file_extension = os.path.splitext(file)
	if file_extension != ".db" and file_extension != ".tiff":
		os.rename(path_img+file, path_img+file_name+".tiff")

