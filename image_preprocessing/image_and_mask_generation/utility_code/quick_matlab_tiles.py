import os

path_img = "image/"

for file in os.listdir(path_img):
	file_name, file_extension = os.path.splitext(file)
	if file_extension != ".db":
		print("tileImages(\"image/"+file+"\",\"image_tiles/"+file_name+".png\")")

